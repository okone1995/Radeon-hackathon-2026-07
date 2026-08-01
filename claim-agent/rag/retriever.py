# -*- coding: utf-8 -*-
"""
rag/retriever.py — 药品目录检索器

实现《设计文档.md》第 6.3 节的检索策略：
  1. 精确/模糊名称匹配（规范化后比对 drug_name / generic_name），命中则高置信直接返回；
  2. 语义召回：未精确命中时，用本地 bge embedding + Chroma 相似度检索 top_k；
  3. 阈值过滤：最高分低于阈值且未命中商保创新药目录 → 判定为「目录外」(in_catalog=false)。

底层向量库为 Chroma（余弦空间），embedding 为本地 sentence-transformers（bge），
均本地执行、不调用远程 API。
"""

import os
import re
import json
import threading
from functools import lru_cache

import rag  # noqa: F401  触发 HF_ENDPOINT 设置与 sys.path 注入
import config as cfg

# 无封顶的哨兵值（Chroma metadata 不允许 None）
NO_CAP = -1.0


# ----------------------------------------------------------------------------
# 名称规范化与目录加载
# ----------------------------------------------------------------------------
def normalize_name(name: str) -> str:
    """去除空格、括号及规格/剂量尾巴，便于精确匹配。"""
    if not name:
        return ""
    s = str(name)
    # 去掉规格/剂量片段，如 0.25g*24粒、100mg、250ml 等
    s = re.sub(r"\d+(\.\d+)?\s*(mg|g|ml|ug|µg|iu|万单位|单位|粒|片|支|袋|瓶|盒|丸|贴|ml/支)", "", s, flags=re.I)
    s = re.sub(r"[\s\(\)（）\[\]【】*×xX:：,，。\-—/]", "", s)
    return s.strip().lower()


def load_catalog(path: str = None) -> list:
    """读取药品目录 JSON，返回 drugs 列表。"""
    path = path or cfg.DRUG_CATALOG_PATH
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, dict):
        return data.get("drugs", [])
    return data  # 兼容纯数组格式


def doc_text(entry: dict) -> str:
    """将可检索字段拼成文档文本（用于建库与语义检索）。"""
    parts = [entry.get("drug_name", ""), entry.get("generic_name", ""),
             entry.get("spec", ""), entry.get("dosage_form", "")]
    return " ".join(p for p in parts if p).strip()


def to_metadata(entry: dict) -> dict:
    """将目录条目转为 Chroma 可存的 metadata（无 None）。"""
    cap = entry.get("cap", None)
    return {
        "drug_name": entry.get("drug_name", "") or "",
        "generic_name": entry.get("generic_name", "") or "",
        "spec": entry.get("spec", "") or "",
        "dosage_form": entry.get("dosage_form", "") or "",
        "category": entry.get("category", "") or "",
        "commercial_innovative": bool(entry.get("commercial_innovative", False)),
        "self_pay_1": float(entry.get("self_pay_1", 0.0) or 0.0),
        "self_pay_2": float(entry.get("self_pay_2", 0.0) or 0.0),
        "reimburse_ratio": float(entry.get("reimburse_ratio", 0.0) or 0.0),
        "cap": float(cap) if cap is not None else NO_CAP,
        "note": entry.get("note", "") or "",
        "catalog_source": entry.get("catalog_source", "") or "",
    }


def _metadata_to_match(md: dict, score: float, in_catalog: bool) -> dict:
    """将 Chroma metadata 还原为对外的 match 结构（cap 哨兵还原为 None）。"""
    cap = md.get("cap", NO_CAP)
    return {
        "matched_name": md.get("drug_name", ""),
        "generic_name": md.get("generic_name", ""),
        "spec": md.get("spec", ""),
        "dosage_form": md.get("dosage_form", ""),
        "category": md.get("category", ""),
        "commercial_innovative": bool(md.get("commercial_innovative", False)),
        "self_pay_1": float(md.get("self_pay_1", 0.0)),
        "self_pay_2": float(md.get("self_pay_2", 0.0)),
        "reimburse_ratio": float(md.get("reimburse_ratio", 0.0)),
        "cap": None if cap == NO_CAP else float(cap),
        "note": md.get("note", ""),
        "catalog_source": md.get("catalog_source", ""),
        "score": round(float(score), 4),
        "in_catalog": in_catalog,
    }


# ----------------------------------------------------------------------------
# Embedding 模型（进程内单例，懒加载）
# ----------------------------------------------------------------------------
_embed_lock = threading.Lock()
_embedder = None


def get_embedder():
    """懒加载并缓存 SentenceTransformer 实例。"""
    global _embedder
    if _embedder is None:
        with _embed_lock:
            if _embedder is None:
                from sentence_transformers import SentenceTransformer
                _embedder = SentenceTransformer(cfg.EMBEDDING_MODEL, device=cfg.EMBEDDING_DEVICE)
    return _embedder


def embed_documents(texts):
    """文档向量化（不加查询前缀），归一化。返回 list[list[float]]。"""
    model = get_embedder()
    vecs = model.encode(list(texts), normalize_embeddings=True, convert_to_numpy=True)
    return vecs.tolist()


def embed_query(text: str):
    """查询向量化（加 bge 查询前缀），归一化。返回 list[float]。"""
    model = get_embedder()
    q = f"{cfg.BGE_QUERY_PREFIX}{text}" if cfg.BGE_QUERY_PREFIX else text
    vec = model.encode([q], normalize_embeddings=True, convert_to_numpy=True)[0]
    return vec.tolist()


# ----------------------------------------------------------------------------
# 检索器
# ----------------------------------------------------------------------------
class DrugCatalogRetriever:
    """药品目录检索器：精确匹配 + 语义召回 + 阈值判定。"""

    def __init__(self, chroma_dir: str = None, collection: str = None,
                 top_k: int = None, score_threshold: float = None):
        self.chroma_dir = chroma_dir or cfg.CHROMA_DIR
        self.collection_name = collection or cfg.CHROMA_COLLECTION
        self.top_k = top_k or cfg.RAG_TOP_K
        self.score_threshold = score_threshold if score_threshold is not None else cfg.RAG_SCORE_THRESHOLD
        self._client = None
        self._collection = None
        # 精确匹配索引：normalize(name) -> entry（drug_name 与 generic_name 均入表）
        self._exact = {}
        self._build_exact_index()

    # ---- 精确匹配索引 ----
    def _build_exact_index(self):
        try:
            entries = load_catalog()
        except Exception:
            entries = []
        for e in entries:
            for key in (e.get("drug_name"), e.get("generic_name")):
                nk = normalize_name(key)
                if nk:
                    self._exact.setdefault(nk, e)

    # ---- Chroma 连接（懒加载）----
    @property
    def collection(self):
        if self._collection is None:
            import chromadb
            self._client = chromadb.PersistentClient(path=self.chroma_dir)
            self._collection = self._client.get_collection(self.collection_name)
        return self._collection

    def _exact_lookup(self, drug_name: str):
        nq = normalize_name(drug_name)
        if not nq:
            return None
        if nq in self._exact:
            return self._exact[nq]
        # 包含关系：查询名包含某通用名（或反之）
        for nk, e in self._exact.items():
            if len(nk) >= 3 and (nk in nq or nq in nk):
                return e
        return None

    def search(self, drug_name: str, top_k: int = None) -> dict:
        """检索药品目录，返回 {query, matches:[...]}。"""
        top_k = top_k or self.top_k

        # 第一层：精确/模糊名称匹配
        hit = self._exact_lookup(drug_name)
        if hit is not None:
            md = to_metadata(hit)
            in_catalog = md["category"] in ("甲类", "乙类")
            match = _metadata_to_match(md, score=1.0, in_catalog=in_catalog)
            match["match_type"] = "exact"
            return {"query": drug_name, "matches": [match]}

        # 第二层：语义召回
        try:
            q_emb = embed_query(drug_name)
            res = self.collection.query(
                query_embeddings=[q_emb],
                n_results=top_k,
                include=["metadatas", "documents", "distances"],
            )
        except Exception as e:
            return {"query": drug_name, "matches": [], "error": f"检索失败: {e}"}

        metadatas = (res.get("metadatas") or [[]])[0]
        distances = (res.get("distances") or [[]])[0]

        matches = []
        for md, dist in zip(metadatas, distances):
            score = max(0.0, 1.0 - float(dist))  # 余弦距离 → 相似度
            # 第三层：阈值判定。低于阈值且非商保创新药 → 目录外
            in_catalog = (score >= self.score_threshold) and (md.get("category") in ("甲类", "乙类"))
            m = _metadata_to_match(md, score=score, in_catalog=in_catalog)
            m["match_type"] = "semantic"
            matches.append(m)

        return {"query": drug_name, "matches": matches}


@lru_cache(maxsize=1)
def get_retriever() -> "DrugCatalogRetriever":
    """进程内单例检索器。"""
    return DrugCatalogRetriever()


if __name__ == "__main__":
    # 简单自测：python -m rag.retriever 阿莫西林
    import sys
    q = sys.argv[1] if len(sys.argv) > 1 else "阿莫西林胶囊"
    r = get_retriever()
    out = r.search(q)
    print(json.dumps(out, ensure_ascii=False, indent=2))

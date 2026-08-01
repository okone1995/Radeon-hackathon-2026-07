# 项目启动说明

## 环境概述

| 组件 | 说明 |
|------|------|
| GPU | AMD Radeon gfx1100 (RX 7900 系列), 48GB VRAM |
| ROCm | 7.2.4 |
| 大模型 | Qwen3.6-27B Q8_K_XL GGUF (~34GB) |
| Python | 3.12, 虚拟环境 `/opt/vllm-env` |
| 项目路径 | `/workspace/fake_ocr_test` |

---

## 一、启动大模型（llama-server）

这是**最核心的一步**，所有后端服务都依赖它提供推理能力。

```bash
# 启动（160K 上下文、视觉多模态、MTP 加速）
/opt/llama.cpp/llama-server \
  -m /workspace/models/Qwen3.6-27B-UD-Q8_K_XL.gguf \
  --mmproj /workspace/models/mmproj-BF16.gguf \
  -ngl 99 \
  -c 163840 \
  -ctk q4_0 -ctv q4_0 \
  -fa on \
  --spec-type draft-mtp \
  -np 1 \
  --host 0.0.0.0 --port 8080 &
```

### 参数说明

| 参数 | 值 | 作用 |
|------|-----|------|
| `-m` | GGUF 模型路径 | 加载模型 |
| `--mmproj` | mmproj-BF16.gguf | **视觉多模态投影器**（必填，否则图片输入报错）|
| `-ngl 99` | 99 | 所有层卸载到 GPU（ROCm）|
| `-c 163840` | 160K tokens | 上下文长度 |
| `-ctk / -ctv` | `q4_0` | KV cache 4bit 量化（160K 时 Q8 显存不够，必须 Q4）|
| `-fa` | `on` | Flash Attention，省显存并加速 |
| `--spec-type draft-mtp` | `draft-mtp` | MTP 多 token 预测，生成速度翻倍 |
| `-np 1` | 1 | 单并发槽（显存有限）|

### 显存参考

| 配置 | KV cache | 显存占用 |
|------|----------|---------|
| 128K + MTP + 视觉 | Q8 量化 | ~46GB（极限）|
| **160K + MTP + 视觉** | **Q4 量化** | **~40.5GB（稳定）**|

### 验证

```bash
curl http://127.0.0.1:8080/health
# 返回 {"status":"ok"} 表示就绪
```

### 注意

- 容器内 DNS 不通，`localhost` 无法解析，连接大模型必须用 `127.0.0.1:8080`
- 模型加载约需 15-30 秒，期间返回 503
- 48GB VRAM 下 256K 上下文+MTP 会撑满，建议 128K

---

## 二、启动 Python 后端服务

三个服务均需设置环境变量 `MODEL_HOST=127.0.0.1 MODEL_PORT=8080` 来指向大模型。

```bash
cd /workspace/fake_ocr_test
source /opt/vllm-env/bin/activate

# 理赔后端 :8001
nohup env MODEL_HOST=127.0.0.1 MODEL_PORT=8080 \
  python3 -u -m backend.main > /tmp/backend8001.log 2>&1 &

# 核保后端 :8002
nohup env MODEL_HOST=127.0.0.1 MODEL_PORT=8080 \
  python3 -u -m underwriting.backend > /tmp/backend8002.log 2>&1 &

# Gradio Web 界面 :7865
nohup env MODEL_HOST=127.0.0.1 MODEL_PORT=8080 \
  python3 -u -c "
import sys; sys.path.insert(0, '.')
import app as a
import threading
threading.Thread(target=a._warmup, daemon=True).start()
a.demo.queue(default_concurrency_limit=4)
a.demo.launch(server_name='0.0.0.0', server_port=7865, share=False)
" > /tmp/fake_ocr.log 2>&1 &
```

### 健康检查

```bash
curl http://127.0.0.1:8001/api/health   # 理赔
curl http://127.0.0.1:8002/api/health   # 核保
curl -s http://127.0.0.1:7865/ | head   # Gradio
```

---

## 三、Cloudflare 隧道（公网反代）

免费创建临时公网地址：

```bash
# 理赔后端
nohup cloudflared tunnel --url http://127.0.0.1:8001 \
  --metrics 0.0.0.0:0 > /tmp/cloudflared8001.log 2>&1 &

# 核保后端
nohup cloudflared tunnel --url http://127.0.0.1:8002 \
  --metrics 0.0.0.0:0 > /tmp/cloudflared8002.log 2>&1 &
```

查看公网地址：

```bash
grep -oP 'https://[a-z0-9-]+\.trycloudflare\.com' /tmp/cloudflared8001.log
grep -oP 'https://[a-z0-9-]+\.trycloudflare\.com' /tmp/cloudflared8002.log
```

---

## 四、服务架构图

```
用户/客户端
    │
    ├── 浏览器 ── Cloudflare 隧道 ──► :8001 理赔后端 (FastAPI)
    │                                    │
    │                                    └──► :8080 llama-server (Qwen3.6-27B)
    │
    ├── 浏览器 ── Cloudflare 隧道 ──► :8002 核保后端 (FastAPI)
    │                                    │
    │                                    └──► :8080 llama-server
    │
    └── 浏览器 ────────────────────► :7865 Gradio 前端 (直连内网)
                                         │
                                         └──► :8080 llama-server
```

所有业务服务均通过 `127.0.0.1:8080` 调用大模型。

---

## 五、常用运维命令

```bash
# 查看所有进程
ps aux | grep -E "llama-server|backend.main|underwriting.backend|cloudflared"

# 查看日志
tail -f /tmp/llama-server.log
tail -f /tmp/backend8001.log
tail -f /tmp/backend8002.log
tail -f /tmp/fake_ocr.log

# 重启所有服务
kill $(pgrep -f "llama-server" 2>/dev/null)
kill $(pgrep -f "backend.main\|underwriting.backend\|python3 -u -c" 2>/dev/null)

# 然后按上文顺序重新启动
```

## 六、环境变量说明

| 变量 | 默认值 | 说明 |
|------|--------|------|
| `MODEL_HOST` | `localhost` | 大模型 API 主机（容器内必须 `127.0.0.1`）|
| `MODEL_PORT` | `8000` | 大模型 API 端口（实际 `8080`）|
| `MODEL_BASE_URL` | `http://{MODEL_HOST}:{MODEL_PORT}/v1` | 完整 API 地址 |
| `MODEL_ID` | `/workspace/models/Qwen3.6-27B-UD-Q8_K_XL.gguf` | 模型标识 |

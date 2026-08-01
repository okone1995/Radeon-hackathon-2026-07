# Demo Script — Intelligent Claim Agent · 4:30

================================================================================
PRE-FLIGHT
================================================================================
□ llama-server :8080 running   □ Backend :8001/:8002 running
□ fapiao.jpg ready             □ rocm-smi visible
□ All terminals: clear history  □ Recording: 1920×1080, EN audio

================================================================================
0:00–0:20  OPENING — Hardware Proof
================================================================================
[SCREEN] Terminal · rocm-smi
[SUBTITLE] Intelligent Claim Agent — AMD Radeon W7900 · ROCm 7.2.4

SPEAK: "Intelligent Claim Agent for AMD Hackathon Track 2. 
All inference runs on AMD Radeon W7900 with ROCm. No cloud APIs."

[ACTION] rocm-smi --showgpu → GPU name, VRAM, ROCm version

================================================================================
0:20–0:50  ★ 权限控制与隐私保护
================================================================================
[SUBTITLE] Capability 1: Permission Control & Privacy Protection
[SCREEN] Terminal · curl commands

SPEAK: "JWT Bearer authentication. Without a token — 401 denied."

[ACTION]
$ curl -X POST 127.0.0.1:8001/api/followup -H "Content-Type: application/json" \
  -d '{"message":"hi","session_id":"x"}'
  → {"detail":"Missing Authorization header"}

$ curl -X POST 127.0.0.1:8001/api/login \
  -H "Content-Type: application/json" \
  -d '{"username":"admin","password":"claim123"}'
  → {"token":"...","expires_in":43200}

SPEAK: "sha256 salted hash. JWT issued. All inference local."

================================================================================
0:50–2:10  ★ 工具调用 + 多步骤规划 + RAG (3-in-1)
================================================================================
[SCREEN] Browser · localhost:8001/docs · upload fapiao.jpg
[SUBTITLE] Capabilities 2–4: Tool Calling · Multi-Step Planning · RAG

SPEAK: "Upload invoice. The agent executes a 4-step pipeline."

[ACTION] Upload → show OCR extracting fields
SPEAK: "Step 1 — Multimodal OCR. Vision LLM extracts fields from image."

[ACTION] Show verify result
SPEAK: "Step 2 — Verification. Cross-check against official tax API."

[ACTION] Show RAG matches — drug category, reimbursement ratio
SPEAK: "Step 3 — RAG retrieval. Chroma + bge on GPU. Three-tier: exact
match, semantic search, threshold for out-of-catalog."

[ACTION] Show conclusion card — total, reimbursable, item breakdown
SPEAK: "Step 4 — Deterministic decision. Pure Python rules. Every amount
auditable — not LLM-generated."

================================================================================
2:10–2:45  ★ 多轮记忆
================================================================================
[SUBTITLE] Capability 5: Multi-Turn Memory
[SCREEN] Same browser · follow-up chat

SPEAK: "Multi-turn memory. No re-extraction needed."

[ACTION] Type: "为什么这个药没报销？"
SPEAK: "Answers from cached RAG result. LangGraph session memory."

[ACTION] Type: "那按乙类算能报多少？"
SPEAK: "Recalculates from stored context. UUID session isolation."

================================================================================
2:45–3:15  Underwriting Risk Agent — Second Agent
================================================================================
[SUBTITLE] Second Agent: Underwriting Risk Assessment
[SCREEN] Browser · localhost:8002/docs or switch terminal

SPEAK: "Beyond claims, insurance companies have another critical, high-volume
workflow: underwriting. Every application requires reviewing medical reports,
detecting lab abnormalities, scoring disease risk, and searching medical
literature — tasks that are repetitive, manual, and error-prone.

We built a second agent specifically for this. Both run on the same AMD GPU."

[ACTION]
Switch to underwriting: open http://localhost:8002/docs
OR if terminal-only: curl the /api/underwriting/process endpoint with a sample report

SPEAK: "Meet the Underwriting Risk Agent. Same architecture — four tools:
medical report extraction, abnormality detection, risk scoring, and
dual-backend medical research search. Independent pipeline. Independent memory.
Same GPU. Two agents for the two most repetitive departments in insurance."

[ACTION]
Show underwriting pipeline: extract → abnormality → risk → report
Point at risk level output (Low/Medium/High)

SPEAK: "Together, the Claims Agent and Underwriting Agent cover the full
insurance value chain — all local, all private, all on AMD Radeon."

================================================================================
3:15–4:05  ★ AMD GPU Performance
================================================================================
[SUBTITLE] AMD Radeon GPU Optimization Evidence
[SCREEN] Split: rocm-smi (left) + terminal (right)

SPEAK: "Live benchmarks on W7900. MTP speculative decoding: 50 tok/s,
100% draft acceptance. Embedding GPU migration: 92.5× speedup."

[ACTION] Run bench.py:
$ cd /workspace/insurance_claim_agent && \
  MODEL_HOST=127.0.0.1 MODEL_PORT=8080 \
  MODEL_ID=/workspace/models/Qwen3.6-27B-UD-Q8_K_XL.gguf \
  /opt/vllm-env/bin/python bench.py --n 3 --max-tokens 128

SPEAK: "10 GPU optimizations, all measured on this hardware. Full AMD
ROCm pipeline: LoRA fine-tune → GGUF quantize → inference on W7900."

================================================================================
4:05–4:30  CLOSING
================================================================================
[SCREEN] Run: echo "W7900 · ROCm 7.2.4 · 48GB · Qwen3.6-27B · 5/5 · All-Local"
[SUBTITLE] Track 2 · Intelligent Claim Agent

SPEAK: "Five of five capabilities. Ten GPU optimizations. Full AMD pipeline.
All local, all private, all auditable. Thank you."

================================================================================
SRT SUBTITLES
================================================================================
1
00:00:00,000 --> 00:00:20,000
Intelligent Claim Agent — AMD Radeon W7900 · ROCm 7.2.4 · Fully Local

2
00:00:20,000 --> 00:00:50,000
★ Permission Control: JWT auth · sha256 hash · 401 without token

3
00:00:50,000 --> 00:02:10,000
★ Tool Calling + Multi-Step Planning + RAG
OCR → Verify → RAG (Chroma + GPU bge) → Deterministic Decision

4
00:02:10,000 --> 00:02:45,000
★ Multi-Turn Memory: session-isolated follow-up with cached context

5
00:02:45,000 --> 00:03:15,000
★ Second Agent: Underwriting Risk Assessment
Medical reports · Abnormality · Risk scoring · Research search
Two agents covering the full insurance value chain on one AMD GPU

6
00:03:15,000 --> 00:04:05,000
★ AMD GPU Optimization: 50 tok/s MTP · 92.5× embedding · 10 optimizations
Full ROCm pipeline: LoRA on W7900 → GGUF → Inference

7
00:04:05,000 --> 00:04:30,000
5/5 capabilities · Dual-agent · All-local · Private · Auditable
Track 2 · Intelligent Claim Agent

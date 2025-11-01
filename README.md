# nano-go-vllm

Minimal CPU‑only Go implementation inspired by nano‑vLLM, with:

- Safetensors loader (F32/F16/BF16) for Hugging Face checkpoints
- HF ByteLevel BPE tokenizer (loads `tokenizer.json`) and proper byte decode
- Qwen‑style Transformer scaffold with RoPE, MHA, MLP, RMSNorm
- Minimal per‑layer KV cache for prefill/decode
- Top‑k / Top‑p sampling with temperature
- Simple CLI for offline text generation

This is a learning/reference implementation — correctness first, then speed. It runs on CPU and is slow for large models.

## Quickstart

Prereqs:
- Go 1.22+ (repo includes a local toolchain under `.tools/go` if needed)
- Python venv to download a model with `huggingface_hub`

Build:

```bash
export PATH="$PWD/.tools/go/bin:$PATH"   # optional: use local Go
go build -o bin/nanovllm cmd/main.go
```

Download a small Qwen model (example: Qwen2.5‑0.5B‑Instruct):

```bash
python3 -m venv .venv
./.venv/bin/pip install -q --upgrade pip huggingface_hub
./.venv/bin/hf download Qwen/Qwen2.5-0.5B-Instruct \
  --include "*.json" --include "*.safetensors" \
  --local-dir models/qwen2_5_0_5b
```

Run a short generation (CPU):

```bash
./bin/nanovllm -max-tokens=16 -temperature=0.2 -top-p=0.9 -top-k=40 \
  models/qwen2_5_0_5b "Hello, how are you?"
```

Flags:
- `-max-tokens` (default 64) — number of new tokens to generate
- `-temperature` (default 0.7)
- `-top-p` (default 0.95)
- `-top-k` (default 50)

Tips:
- Use proper chat formatting for instruct models, e.g.:

```text
<|im_start|>system You are a helpful assistant.<|im_end|>
<|im_start|>user How are you today?<|im_end|>
<|im_start|>assistant
```

## What’s inside

- `pkg/safetensors`: parses safetensors header, reads and converts weights
- `pkg/tokenizer`: minimal ByteLevel BPE compatible loader/decoder
- `internal/models`: Qwen‑style model wiring + safetensors weight loading
- `internal/layers`: Embedding, RMSNorm, Linear, MLP (SiLU‑gate), Attention (RoPE)
- `internal/engine`: basic scheduler, runner, and sampling glue
- `cmd/main.go`: CLI with sampling flags

## Limitations

- CPU only, single sequence at a time; slow for >0.5B models
- Minimal KV cache; no paged cache or tensor parallel
- Math kernels are naive loops; BLAS/vectorization would speed things up

## Roadmap

- BLAS/AVX vectorized matmul and attention kernels
- Streaming token output
- Repetition penalty, frequency/presence penalties
- Batched generation

## License

MIT


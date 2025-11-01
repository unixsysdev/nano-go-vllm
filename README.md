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
- Go 1.22+
- Python venv to download a model with `huggingface_hub`

Build:

```bash
go build -o bin/nanovllm cmd/main.go
```

Download a model (script)

Use the helper to fetch a small public model. Examples:

```bash
# Qwen3 0.6B (base)
scripts/download_model.sh qwen3-0.6b models/qwen3_0_6b

# Qwen2.5 0.5B Instruct
scripts/download_model.sh qwen2.5-0.5b-instruct models/qwen2_5_0_5b

# Or any HF repo id you have access to:
scripts/download_model.sh Qwen/Qwen2.5-0.5B-Instruct models/qwen2_5_0_5b
```

Notes:
- Public models work without auth. For gated/private repos, set `HF_TOKEN`.
- The script prefers the `hf` CLI if present, else uses a local venv with `huggingface_hub`.

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
 - `-repetition-penalty` (default 1.1)
 - `-presence-penalty`, `-frequency-penalty`
- `-stream` (stream tokens as they are generated)

Verification / Debugging

- `-verify` prints the top logits for the last token (no sampling). Useful to check parity with a reference implementation.
  Example: `./bin/nanovllm -verify models/qwen3_0_6b "Hello"`


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

## Status

- CPU only. Minimal per‑layer KV cache.
- Streaming output (`-stream`) supported.
- Sampling: Top‑k / Top‑p, repetition / presence / frequency penalties.
- Tokenizer + weights: ByteLevel BPE tokenizer and safetensors loader (F32/F16/BF16).
- Vectorized math: Attention core uses BLAS‑backed GEMM (Q·Kᵀ and probs·V); linear layers ride BLAS via gorgonia tensor.
- RoPE: rope_theta read from `config.json` and applied.

## Roadmap

- Repro‑checked RoPE: implement rope_scaling variants and verify numerical parity with HF for Qwen.
- Batched generation with shared kernels (multi‑seq forward with shared GEMM calls).
- Typical sampling / min_p and deterministic seeding.
- Parity tests: 1‑token logits checks against transformers; micro‑benchmarks for kernels.

## License

MIT

# Realtime Demo Commands

Use these commands for a quick demo setup on this machine.

## 1) Install system libraries (one-time)

```bash
apt-get update
apt-get install -y libnuma1 libibverbs1
```

## 2) Ensure sgl-kernel is up to date (in the project venv)

```bash
uv pip install --upgrade sgl-kernel
```

## 3) Start realtime demo (2 GPUs, validated)

This is the command for demo mode that matched the successful startup.

Important: for this specific startup failure (`Not enough memory ... increase --mem-fraction-static`),
you must use a higher `mem_fraction_static` value.

```bash
CUDA_VISIBLE_DEVICES=0,1 ./playground/realtime/start.sh \
  --model-path Qwen/Qwen3-Omni-30B-A3B-Instruct \
  --stages.4.executor.args.mem_fraction_static 0.90 \
  --stages.4.executor.args.max_running_requests 1 \
  --stages.4.executor.args.thinker_max_seq_len 512 \
  --stages.6.executor.args.talker_max_seq_len 512
```

## 4) If startup still fails, run backend directly (same settings)

This bypasses `start.sh` and uses the same flags directly.

```bash
CUDA_VISIBLE_DEVICES=0,1 ./.venv/bin/python -m sglang_omni.cli.cli serve \
  --model-path Qwen/Qwen3-Omni-30B-A3B-Instruct \
  --host 0.0.0.0 \
  --port 8000 \
  --stages.4.executor.args.mem_fraction_static 0.90 \
  --stages.4.executor.args.max_running_requests 1 \
  --stages.4.executor.args.thinker_max_seq_len 512 \
  --stages.6.executor.args.talker_max_seq_len 512
```

# Install Wizard

After cloning the repo, run:

```
uv run llamactl init
```

The wizard:

1. Asks whether to overwrite an existing `config.toml` (backs it up to
   `config.toml.bak.<timestamp>` if you say yes).
2. Locates `llama-server.exe` in three places:
   `../llama.cpp/build/bin/Release/`, `../llamacpp-bin/`, then your
   `PATH`. If none are found, offers to download a pinned Windows CUDA
   release of llama.cpp into `./llamacpp-bin/`.
3. Detects your GPU's VRAM via `nvidia-smi` and recommends a GGUF
   model:
   - ≥ 24 GB → Qwen3-Coder-30B-A3B-Instruct-UD-Q4_K_XL
   - 14–23 GB → DeepSeek-R1-Distill-Qwen-14B-Q6_K_L
   - 8–13 GB → Llama-3.2-3B-Instruct-Q5_K_M
   - unknown / < 8 GB → no recommendation; pick manually
4. Scans `./GGUF/`, `../GGUF/`, and `~/GGUF/` for the chosen file. If
   absent, offers to download it from HuggingFace into `./GGUF/`.
   (Requires `huggingface_hub` — install with
   `uv add huggingface_hub` if the wizard says it's missing.)
5. Collects `allowed_dirs`. The repo root is added automatically; you
   can add more paths in a loop.
6. Picks `ctx_size` and `n_parallel` for your VRAM tier
   (24 GB → 65536/2, 14 GB → 32768/2, 8 GB → 8192/1). Accept the
   defaults or override.
7. Writes `config.toml` and prints the next commands.

## Flags

- `--force` — overwrite an existing `config.toml` without prompting
  (still backs up first).

## When to re-run

Re-run any time you change GPU, add a new model, or move the repo. The
wizard is deterministic given the same answers, so you can re-run it
to regenerate `config.toml` after editing helpers.

## Edge cases

- **No NVIDIA GPU / no `nvidia-smi`** → wizard skips VRAM detection
  and the tier becomes "unknown". You can pick any model from the
  full list.
- **No `huggingface_hub` installed** → the model-download prompt
  errors with a clear instruction. Install with
  `uv add huggingface_hub` or download the GGUF manually into
  `./GGUF/`.
- **Pinned llama.cpp release moved** → if the auto-download fails
  (404 or sha mismatch), the wizard tells you, and you can supply a
  path manually.

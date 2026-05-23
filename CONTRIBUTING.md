# Contributing

Noeris has two useful local setup paths:

- **CPU-only source-tree development** for docs, CLI parsing, artifact checks,
  import checks, and most non-GPU unit tests.
- **Linux CUDA development** for full package installs, Triton kernels, local
  GPU validation, and `scripts/ci_local.sh` parity.

Use Python 3.11 unless a workflow says otherwise.

## CPU-Only Setup

Use this path on macOS arm64, laptops without NVIDIA GPUs, and any machine
where you only need to edit docs, CLI code, tests, or non-kernel plumbing.

```bash
python3.11 -m venv .venv
. .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install numpy scikit-learn pytest pytest-timeout

export PYTHONPATH="$PWD/src:$PWD${PYTHONPATH:+:$PYTHONPATH}"
```

Validate that the source tree is importable:

```bash
python -c "import noeris; import research_engine; print(noeris.__version__)"
python -m research_engine.cli status
python scripts/check_public_claim_artifacts.py
```

Run a small CPU-safe test slice:

```bash
python -m unittest tests.test_cli tests.test_ci_local_script tests.test_codex_config tests.test_llm
```

Targeted `unittest` invocations under the `tests.*` package apply the
source-tree test path setup automatically. CLI commands and scripts still need
the `PYTHONPATH` export above unless the package is installed editable.

Preview the full local CI command list without running the GPU-adjacent
benchmark steps:

```bash
CI_LOCAL_DRY_RUN=1 PYTHON_BIN=python ./scripts/ci_local.sh
```

### macOS arm64 and `uv`

The project package depends on Triton because the Linux CUDA path needs it.
The checked-in `uv.lock` currently resolves Triton from Linux wheels; it does
not provide a native macOS arm64 Triton wheel. On Apple Silicon, commands such
as `uv sync`, `uv run`, or `pip install -e .` can fail while resolving or
installing Triton.

For macOS arm64, use the `PYTHONPATH` source-tree path above for CPU-safe work.
Use a Linux CUDA machine, container, or Modal for full Triton validation.

## Linux CUDA Setup

Use this path on a Linux host with an NVIDIA GPU and CUDA-compatible PyTorch.
This is the closest local match to GitHub CI plus kernel validation.

```bash
python3.11 -m venv .venv
. .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[dev]" pytest-timeout

python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
PYTHON_BIN=python ./scripts/ci_local.sh
```

`scripts/ci_local.sh` sets `PYTHONPATH` for the source tree, runs the unit test
suite, checks public artifact references, runs two deterministic
`matmul-speedup` benchmark records, exports history, and checks the exported
history regression gate.

For local GPU kernel search, add `--local`:

```bash
python -m research_engine.cli triton-iterate \
  --operator rmsnorm \
  --gpu A100 \
  --configs-per-run 8 \
  --local
```

## Modal and API Keys

Most contributor work does not need credentials.

| Work | Modal token | LLM/API key |
|---|---:|---:|
| Imports, docs, CLI help/status | No | No |
| CPU-safe tests and artifact checks | No | No |
| `scripts/ci_local.sh` local parity | No | No |
| `triton-iterate --local` | No | Optional with `--llm` |
| `triton-iterate` without `--local` | Yes | Optional with `--llm` |
| Modal benchmark scripts under `scripts/modal_*` | Yes | No |
| Research or benchmark iteration with `--llm` / `--live-execution` | Depends on runner | Yes |

For Modal-backed runs:

```bash
python -m pip install modal
modal token new
python -m research_engine.cli triton-iterate \
  --operator rmsnorm \
  --gpu A100 \
  --configs-per-run 8
```

GitHub workflows use `MODAL_TOKEN_ID` and `MODAL_TOKEN_SECRET` secrets for
Modal. LLM-backed workflows use `AZURE_OPENAI_API_KEY`,
`AZURE_OPENAI_BASE_URL`, `AZURE_OPENAI_MODEL`, and `AZURE_OPENAI_WIRE_API`
when running through Azure OpenAI. Local LLM-backed commands can also use
`OPENAI_API_KEY` or a Codex provider config that targets the Responses API.

## Free GPU Validation

For quick CUDA validation without paid compute, use Kaggle or Colab T4.

```bash
git clone https://github.com/0sec-labs/noeris
cd noeris
python -m pip install -e . numpy scikit-learn
python scripts/colab_validate_all.py
```

Those environments provide the CUDA GPU; they are not a substitute for A100 or
H100 performance claims.

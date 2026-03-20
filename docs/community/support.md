---
description: >
  How to get support for vLLM — choosing the right channel, writing effective
  bug reports, and understanding what to expect from the community.
---

# Getting support

vLLM is an open-source project maintained by a community of volunteers and
contributors. Support is provided on a best-effort basis through the channels
described below. Choosing the right channel and providing good information
upfront is the fastest way to get a helpful answer.

---

## :compass: Choosing the right channel

| Situation | Best channel |
|---|---|
| You have a **usage question** (how do I…?) | [User Forum](https://discuss.vllm.ai) |
| You found a **bug** | [GitHub Issues](https://github.com/vllm-project/vllm/issues/new/choose) |
| You want to **request a feature** | [GitHub Issues](https://github.com/vllm-project/vllm/issues/new/choose) |
| You want to **propose a major change** | [RFC process](rfcs.md) |
| You need **real-time help** from contributors | [Developer Slack](https://slack.vllm.ai) |
| You discovered a **security vulnerability** | [GitHub Security Advisories](https://github.com/vllm-project/vllm/security/advisories) |
| You want to **collaborate or partner** | [collaboration@vllm.ai](mailto:collaboration@vllm.ai) |

!!! tip "Search before posting"
    Before opening an issue or posting on the forum, search existing issues and
    discussions. Your question may already have an answer.

---

## :material-forum: User forum

The [vLLM User Forum](https://discuss.vllm.ai) is the best place for:

- Questions about installation, configuration, and usage
- Sharing deployment experiences and tips
- Discussing model compatibility
- Long-form technical discussions that benefit from a permanent, searchable record

The forum is indexed by search engines, so a well-written question helps future
users with the same problem.

### Writing a good forum post

1. **Use a descriptive title.** "vLLM crashes on startup" is less useful than
   "ImportError: cannot import name 'LLM' after upgrading to vLLM 0.7.0 on CUDA 12.4".
1. **Describe what you expected and what happened.** Include the exact error
   message or unexpected output.
1. **Include your environment.** Run `vllm collect-env` and paste the output
   (see [below](#collecting-environment-information)).
1. **Provide a minimal reproduction.** A short Python script or `curl` command
   that reproduces the issue is far more useful than a description alone.
1. **Format code with code blocks.** Use triple backticks (` ``` `) so code is
   readable.

---

## :fontawesome-brands-github: GitHub issues

Use [GitHub Issues](https://github.com/vllm-project/vllm/issues) for:

- **Bug reports** — something that worked before and no longer does, or
  behavior that contradicts the documentation.
- **Feature requests** — a capability you need that vLLM does not currently
  provide.

!!! important "Security vulnerabilities"
    Do **not** open a public GitHub Issue for security vulnerabilities. Use
    [GitHub Security Advisories](https://github.com/vllm-project/vllm/security/advisories/new)
    instead. See the [security policy](https://github.com/vllm-project/vllm/blob/main/SECURITY.md)
    for details.

### Writing an effective bug report

A good bug report includes:

1. **vLLM version** — output of `python -c "import vllm; print(vllm.__version__)"`.
1. **Environment information** — output of `vllm collect-env` (see [below](#collecting-environment-information)).
1. **Steps to reproduce** — the exact commands or code needed to trigger the bug.
1. **Expected behavior** — what you expected to happen.
1. **Actual behavior** — what actually happened, including the full error
   traceback.
1. **Relevant logs** — paste logs with `--log-level debug` if the default output
   is not informative enough.

!!! tip "Use the issue template"
    When you click **New Issue** on GitHub, select the appropriate template
    (Bug Report, Feature Request, or RFC). The template prompts you for the
    information maintainers need.

### What happens after you file an issue

- A maintainer or community member will triage the issue, usually within a few
  days.
- Issues may be labeled `needs-reproduction`, `needs-info`, `good first issue`,
  `help wanted`, or `rfc-required`.
- If your issue is labeled `needs-info`, please respond with the requested
  information within a reasonable time frame. Issues that remain inactive may
  be closed.

---

## :fontawesome-brands-slack: Developer Slack

The [vLLM Developer Slack](https://slack.vllm.ai) is a real-time chat workspace
for contributors and power users. It is best for:

- Quick clarifying questions while working on a contribution
- Coordinating PR reviews
- Discussing ongoing development work
- Joining working groups (`#sig-*` and `#feat-*` channels)

!!! note
    Slack is not a substitute for GitHub Issues. If you find a bug or want to
    request a feature, please open a GitHub Issue so it is tracked and
    searchable. Slack messages are not indexed and may be lost.

### Recommended channels

| Channel | Purpose |
|---|---|
| `#contributors` | General contributor discussion |
| `#pr-reviews` | PR review requests and coordination |
| `#sig-ci` | CI infrastructure working group |
| `#feat-*` | Feature-specific working groups |
| `#sig-*` | Special interest groups (e.g., `#sig-torch-compile`) |

---

## :material-console: Collecting environment information

The `vllm collect-env` command gathers system and package information that
maintainers need to diagnose issues. Always include this output in bug reports.

```bash
vllm collect-env
```

This prints information such as:

- vLLM version
- Python version
- PyTorch version and CUDA version
- GPU model and driver version
- Operating system
- Installed packages relevant to vLLM

If vLLM is not installed or the command fails, you can collect similar
information manually:

```bash
python -c "import vllm; print(vllm.__version__)"
python -c "import torch; print(torch.__version__, torch.version.cuda)"
nvidia-smi
uname -a
pip list | grep -E "vllm|torch|transformers|accelerate"
```

---

## :material-frequently-asked-questions: Common issues

### Installation problems

- **CUDA version mismatch** — ensure your PyTorch CUDA version matches your
  system CUDA driver. See the [installation guide](../getting_started/installation/index.md).
- **`pip install vllm` fails on non-CUDA platforms** — use the platform-specific
  installation instructions (e.g., [ROCm](../getting_started/installation/gpu-rocm.md),
  [CPU](../getting_started/installation/cpu.md)).
- **Build errors from source** — see [Building from source](../getting_started/installation/from-source.md)
  and ensure all build dependencies are installed.

### Runtime errors

- **Out-of-memory (OOM) errors** — reduce `--gpu-memory-utilization` (default
  `0.9`) or use a smaller model. See [engine arguments](../configuration/engine_args.md).
- **Slow first request** — vLLM compiles CUDA graphs on the first request.
  Subsequent requests are faster. Use `--enforce-eager` to disable graph
  compilation if startup latency is critical.
- **Model not found** — ensure the model ID is correct and you have a valid
  Hugging Face token if the model is gated.

### API server issues

- **Port already in use** — change the port with `--port <PORT>`.
- **Authentication errors** — set `VLLM_API_KEY` and pass
  `--api-key $VLLM_API_KEY` to the server.
- **Timeout on long requests** — increase the client timeout; vLLM does not
  impose a server-side request timeout by default.

---

## :material-book-open-variant: Additional resources

- [Quickstart guide](../getting_started/quickstart.md) — get up and running in minutes
- [Configuration reference](../configuration/index.md) — all engine arguments explained
- [Deployment guide](../deployment/index.md) — Docker, Kubernetes, and production setup
- [Benchmarking guide](../benchmarking/index.md) — measure and optimize performance
- [Contributing guide](../contributing/index.md) — how to contribute code or docs
- [RFC process](rfcs.md) — how to propose major changes

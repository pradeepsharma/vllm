# vLLM Documentation — Markdown Style Guide

**Version:** 1.0  
**Date:** 2026-03-19  
**Applies to:** All Markdown files under `docs/`

This guide defines the writing style, formatting conventions, and admonition
patterns for the vLLM documentation set. All contributors and reviewers should
follow these conventions to ensure a consistent, high-quality reading experience.

---

## Table of Contents

1. [General Principles](#1-general-principles)
2. [File and Directory Conventions](#2-file-and-directory-conventions)
3. [Front Matter](#3-front-matter)
4. [Headings](#4-headings)
5. [Prose Style](#5-prose-style)
6. [Code Blocks](#6-code-blocks)
7. [Links](#7-links)
8. [Lists](#8-lists)
9. [Tables](#9-tables)
10. [Admonitions](#10-admonitions)
11. [Tabs](#11-tabs)
12. [Collapsible Sections](#12-collapsible-sections)
13. [Images and Figures](#13-images-and-figures)
14. [Emoji and Icons](#14-emoji-and-icons)
15. [Grid Cards](#15-grid-cards)
16. [Math](#16-math)
17. [API Cross-References](#17-api-cross-references)
18. [Accessibility](#18-accessibility)
19. [Common Mistakes to Avoid](#19-common-mistakes-to-avoid)

---

## 1. General Principles

- **Clarity first.** Write for a reader who is intelligent but unfamiliar with
  vLLM internals. Avoid jargon without explanation.
- **One idea per sentence.** Keep sentences short (≤ 25 words where possible).
- **Active voice.** Prefer "vLLM loads the model" over "the model is loaded by vLLM".
- **Present tense.** Prefer "vLLM supports…" over "vLLM will support…".
- **Second person.** Address the reader as "you". Avoid "the user" or "one".
- **Consistent terminology.** Use the [Glossary](#) for canonical terms. Do not
  alternate between synonyms (e.g., always "tensor parallelism", never "model
  parallelism" unless quoting).

---

## 2. File and Directory Conventions

| Convention | Rule |
|---|---|
| **File names** | `kebab-case.md` (e.g., `openai_compatible_server.md`) |
| **Section indexes** | Always named `index.md` (preferred) or `README.md` |
| **Include files** | Suffix `.inc.md` — excluded from nav by `mkdocs.yaml` |
| **Template files** | Suffix `.template.md` — excluded from nav |
| **Directory names** | `snake_case` or `kebab-case` (match existing convention per section) |
| **Asset files** | Place in `docs/assets/<section>/` |

---

## 3. Front Matter

Every page **should** include YAML front matter. Minimum required fields:

```yaml
---
description: >
  One or two sentence description used in search results and social previews.
  Keep it under 160 characters.
---
```

Optional fields:

```yaml
---
description: >
  Short description for SEO and search.
hide:
  - navigation   # Use only on full-width landing pages (e.g., index.md)
  - toc          # Use only when the page has no meaningful headings
toc_depth: 3     # Limit TOC depth (useful for very long pages)
---
```

**Rules:**
- `hide: navigation` and `hide: toc` are reserved for top-level landing pages
  (`docs/index.md`). Do not use them on content pages.
- `description` should be a complete sentence, not a fragment.

---

## 4. Headings

```markdown
# Page Title        ← H1: exactly one per page, matches the nav label
## Major Section    ← H2: primary content divisions
### Subsection      ← H3: sub-topics within a section
#### Detail         ← H4: use sparingly; prefer prose or lists instead
```

**Rules:**
- Use **sentence case** for all headings (capitalize only the first word and
  proper nouns). Example: `## Getting started with vLLM` ✅ not `## Getting
  Started With vLLM` ❌.
- Do not skip heading levels (e.g., H2 → H4).
- Do not use bold or italic inside headings.
- Headings should be descriptive enough to stand alone in the TOC.
- Avoid ending headings with punctuation.

---

## 5. Prose Style

### Terminology

| Use | Avoid |
|---|---|
| vLLM | VLLM, vllm (in prose) |
| Hugging Face | HuggingFace, huggingface |
| PagedAttention | Paged Attention, paged attention |
| OpenAI-compatible | OpenAI compatible, openai-compatible |
| LoRA | lora, LORA |
| GPU | gpu |
| CPU | cpu |
| Python | python (in prose) |

### Numbers and Units

- Spell out numbers one through nine; use numerals for 10 and above.
- Use SI prefixes for memory: GB, MB, TB (not GiB unless specifically binary).
- Use `ms` for milliseconds, `s` for seconds in performance contexts.

### Emphasis

- Use **bold** for UI labels, key terms on first use, and critical warnings.
- Use *italic* for titles of external works, introducing new terms, and light
  emphasis.
- Do not use bold or italic for decoration.
- Do not use ALL CAPS for emphasis — use bold instead.

---

## 6. Code Blocks

### Fenced Code Blocks

Always specify the language for syntax highlighting:

````markdown
```bash
vllm serve Qwen/Qwen2.5-1.5B-Instruct
```
````

````markdown
```python
from vllm import LLM, SamplingParams
llm = LLM(model="facebook/opt-125m")
```
````

````markdown
```yaml
# mkdocs.yaml
site_name: vLLM
```
````

### Language Tags

| Content | Tag |
|---|---|
| Shell commands | `bash` |
| Python code | `python` |
| YAML config | `yaml` |
| JSON | `json` |
| Dockerfile | `dockerfile` |
| Plain text / output | `text` |
| Console output | `console` |
| Environment variables | `bash` |

### Inline Code

Use backticks for:
- Command names: `` `vllm serve` ``
- File paths: `` `docs/index.md` ``
- Parameter names: `` `--tensor-parallel-size` ``
- Python identifiers: `` `LLM` ``, `` `SamplingParams` ``
- Environment variables: `` `VLLM_API_KEY` ``
- Values: `` `true` ``, `` `null` ``, `` `8000` ``

Do **not** use backticks for product names (vLLM, PyTorch) or general terms.

### Shell Command Conventions

- Use `$` prefix for user-level shell commands only when showing a prompt is
  necessary for clarity. Omit `$` in most cases so commands are easy to copy.
- Use `\` for line continuation in long commands:

```bash
docker run --runtime nvidia --gpus all \
    -p 8000:8000 \
    vllm/vllm-openai:latest \
    --model mistralai/Mistral-7B-Instruct-v0.1
```

---

## 7. Links

### Internal Links

Use **relative paths** for all internal links:

```markdown
[Quickstart](../getting_started/quickstart.md)          ✅
[Quickstart](https://docs.vllm.ai/en/latest/...)        ❌ (absolute, breaks versioning)
```

### API Cross-References

Use MkDocs `mkdocstrings` cross-reference syntax for Python API links:

```markdown
[LLM][vllm.LLM]
[SamplingParams][vllm.SamplingParams]
```

### External Links

External links open in the same tab by default. Do not add `{target="_blank"}`
unless there is a strong UX reason (e.g., a link that would interrupt a tutorial).

### Link Text

- Use descriptive link text: `[OpenAI-compatible server guide](...)` ✅
- Avoid bare URLs in prose: `See https://...` ❌
- Avoid "click here" or "here": `See [here](...)` ❌

---

## 8. Lists

### Unordered Lists

Use `-` as the bullet character (not `*` or `+`):

```markdown
- First item
- Second item
    - Nested item (4-space indent)
```

### Ordered Lists

Use `1.` for all items (MkDocs auto-numbers):

```markdown
1. First step
1. Second step
1. Third step
```

### Rules

- Use lists for three or more parallel items.
- Keep list items grammatically parallel (all noun phrases, or all imperative
  sentences, etc.).
- End list items with a period if they are complete sentences; omit punctuation
  for fragments.
- Do not use lists for two items that read naturally as prose with "and".

---

## 9. Tables

Use tables for structured comparisons. Always include a header row:

```markdown
| Column A | Column B | Column C |
|---|---|---|
| Value 1  | Value 2  | Value 3  |
```

**Rules:**
- Left-align text columns; center-align status/icon columns.
- Keep table cells concise — move long explanations to prose below the table.
- Use `—` (em dash) for "not applicable" cells, not `N/A` or blank.
- Avoid tables with more than 6 columns — consider splitting or using a list.

---

## 10. Admonitions

vLLM docs use MkDocs Material admonitions. The following types are approved and
their usage is defined below.

### Approved Admonition Types

#### `note` — Supplementary information

Use for information that is helpful but not critical to the main flow.

```markdown
!!! note
    By default, vLLM downloads models from Hugging Face.
```

#### `tip` — Helpful suggestions

Use for best practices, shortcuts, or recommendations.

```markdown
!!! tip
    Use `uv` instead of `pip` for faster dependency resolution.
```

#### `important` — Must-read information

Use for information the reader must understand to avoid problems. Use sparingly.

```markdown
!!! important
    By default, the server applies `generation_config.json` from the model
    repository. Pass `--generation-config vllm` to disable this behavior.
```

#### `warning` — Potential for data loss or breakage

Use when an action could cause data loss, security issues, or hard-to-reverse
consequences.

```markdown
!!! warning
    vLLM does not support Windows natively. Use WSL2 for Windows deployments.
```

#### `danger` — Severe risk

Use only for actions that could cause irreversible damage (e.g., data deletion,
security vulnerabilities). Use extremely sparingly.

```markdown
!!! danger
    Disabling authentication exposes your API server to the public internet.
```

#### `example` — Code examples and walkthroughs

Use to frame a complete worked example.

```markdown
!!! example "Serving Llama 3 with tensor parallelism"
    ```bash
    vllm serve meta-llama/Llama-3-8B --tensor-parallel-size 2
    ```
```

#### `announcement` — Release notes and deprecation notices

Use for version-specific announcements (e.g., deprecation warnings).

```markdown
!!! announcement
    V0 has been fully deprecated. See [RFC #18571](https://github.com/...) for details.
```

#### `question` (collapsible) — FAQ items

Use inside FAQ sections. Always collapsible (`???`).

```markdown
??? question "Can I run vLLM without a GPU?"
    Yes! vLLM supports CPU-only inference on x86, ARM, Apple Silicon, and IBM Z.
```

### Collapsible vs. Always-Open

- Use `!!!` for admonitions that should always be visible.
- Use `???` for admonitions that should be collapsed by default (FAQ, long code
  examples, optional details).
- Use `???+` for admonitions that are expanded by default but collapsible.

### Admonition Titles

- Omit the title to use the default type name: `!!! note` → renders as "Note".
- Provide a custom title in quotes for context: `!!! note "Python version"`.
- Use sentence case for custom titles.

### Nesting

Do not nest admonitions inside other admonitions. If you need nested structure,
use a collapsible section (`??? code`) inside a regular admonition.

### Frequency

- **`note`**: Use freely, but no more than 2–3 per page.
- **`tip`**: Use freely.
- **`important`**: Maximum 1–2 per page.
- **`warning`**: Maximum 1–2 per page.
- **`danger`**: Maximum 1 per page; prefer `warning` unless truly severe.

---

## 11. Tabs

Use MkDocs Material content tabs for platform-specific or alternative
instructions. Always use the `===` syntax:

```markdown
=== "NVIDIA CUDA"
    ```bash
    uv pip install vllm --torch-backend=auto
    ```

=== "AMD ROCm"
    ```bash
    uv pip install vllm --extra-index-url https://wheels.vllm.ai/rocm/
    ```
```

**Rules:**
- Tab labels use title case.
- Keep tab content parallel — each tab should cover the same topic for its
  platform.
- Do not use tabs for fewer than two alternatives.
- Do not nest tabs.

---

## 12. Collapsible Sections

Use `??? code` for long code examples that are supplementary to the main flow:

```markdown
??? code "Full Python example"
    ```python
    from openai import OpenAI
    client = OpenAI(api_key="EMPTY", base_url="http://localhost:8000/v1")
    # ...
    ```
```

Use `???+` to start expanded:

```markdown
???+ note "Advanced configuration"
    This section covers advanced options for power users.
```

---

## 13. Images and Figures

Use the MkDocs Material figure syntax for captioned images:

```markdown
<figure markdown="span">
  ![Alt text describing the image](../assets/design/arch-overview.png){ width="80%" }
  <figcaption>Architecture overview of the vLLM V1 engine.</figcaption>
</figure>
```

**Rules:**
- Always provide meaningful `alt` text (not "image" or the file name).
- Store images in `docs/assets/<section>/`.
- Use PNG for diagrams, JPEG for photographs.
- Specify `width` as a percentage to ensure responsive scaling.
- Use `{ align="center" }` for centered standalone images.

---

## 14. Emoji and Icons

Use MkDocs Material emoji syntax for icons in headings and cards:

```markdown
## :rocket: Getting Started
## :warning: Known Limitations
```

Use sparingly in prose — emoji in headings and card titles are acceptable;
emoji in body text should be rare and purposeful.

**Approved emoji for section headings:**

| Emoji | Code | Use for |
|---|---|---|
| 🚀 | `:rocket:` | Getting started, quickstart |
| ⚡ | `:zap:` | Performance, speed |
| 🧠 | `:brain:` | Models, AI concepts |
| 🌐 | `:globe_with_meridians:` | API, networking |
| 📊 | `:bar_chart:` | Benchmarking, metrics |
| 🔄 | `:arrows_counterclockwise:` | Distributed, parallelism |
| 🔌 | `:electric_plug:` | Hardware, plugins |
| 📦 | `:package:` | Installation |
| 🐳 | `:whale:` | Docker |
| 📚 | `:books:` | Documentation sections |
| ✅ | `:white_check_mark:` | Requirements, compatibility |
| ❓ | `:question:` | FAQ |
| 🔗 | `:link:` | Related resources |
| 🤝 | `:handshake:` | Community |
| 🧭 | `:compass:` | Navigation, decision trees |
| 📜 | `:scroll:` | Citation, legal |

---

## 15. Grid Cards

Use MkDocs Material grid cards for section overviews and feature lists.
Requires `attr_list` and `md_in_html` extensions (both enabled in `mkdocs.yaml`).

```markdown
<div class="grid cards" markdown>

-   :material-rocket-launch: **Getting Started**

    ---

    Brief description of this card's content.

    [:octicons-arrow-right-24: Link text](path/to/page.md)

-   :material-server: **Serving**

    ---

    Brief description of this card's content.

    [:octicons-arrow-right-24: Link text](path/to/page.md)

</div>
```

**Rules:**
- Use 2–6 cards per grid. More than 6 cards becomes hard to scan.
- Each card should have: an icon, a bold title, a `---` divider, a 1–2 sentence
  description, and a single call-to-action link.
- Use `material/` icons for section cards and `octicons/` for action links.

---

## 16. Math

Use MkDocs Material's MathJax integration for mathematical expressions.
The `pymdownx.arithmatex` extension is enabled.

Inline math:

```markdown
The temperature parameter $T$ controls the sharpness of the distribution.
```

Block math:

```markdown
$$
P(x_i) = \frac{\exp(z_i / T)}{\sum_j \exp(z_j / T)}
$$
```

---

## 17. API Cross-References

Use `mkdocstrings` cross-reference syntax to link to Python API symbols:

```markdown
[LLM][vllm.LLM]
[SamplingParams][vllm.SamplingParams]
[RequestOutput][vllm.RequestOutput]
```

This generates a hyperlink to the auto-generated API reference page for that
symbol. Always prefer this over manual links to API pages.

---

## 18. Accessibility

- All images must have descriptive `alt` text.
- Do not convey information through color alone (e.g., in tables, use text
  labels in addition to colored icons).
- Use `<abbr>` tags for abbreviations on first use:
  ```markdown
  <abbr title="Key-Value Cache">KV cache</abbr>
  ```
- Ensure heading hierarchy is logical (no skipped levels).
- Code blocks should be self-contained — do not rely on surrounding prose to
  explain what a code block does; use comments inside the code block.

---

## 19. Common Mistakes to Avoid

| ❌ Avoid | ✅ Prefer |
|---|---|
| `# vLLM Documentation` as H1 on every page | Use the page's specific title |
| Bare URLs: `See https://github.com/...` | `See [GitHub](https://github.com/...)` |
| `Click here` link text | Descriptive link text |
| Mixing `-` and `*` bullets in the same file | Use `-` consistently |
| `!!! NOTE` (uppercase type) | `!!! note` (lowercase) |
| Skipping heading levels (H2 → H4) | Use sequential levels |
| Title Case In All Headings | Sentence case in headings |
| Trailing whitespace | Clean line endings |
| Hard-coded absolute doc URLs | Relative paths |
| `vllm` in prose (lowercase) | `vLLM` in prose |
| `Huggingface` | `Hugging Face` |
| Tabs for indentation in YAML | Spaces (2-space indent) |
| Long lines > 100 chars in prose | Wrap at 80–100 characters |

---

*This style guide is a living document. Propose changes via a pull request to
`.sasva/style/markdown-style-guide.md`.*

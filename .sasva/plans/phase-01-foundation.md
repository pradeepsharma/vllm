# Phase 01 — Foundation & Directory Setup

**Status:** ✅ Complete  
**Date:** 2026-03-19  
**Owner:** SASVA AI

---

## Objective

Establish the documentation foundation: landing page, getting-started overview,
MkDocs configuration verification, `.sasva/` directory structure, and a Markdown
style guide.

---

## Tasks

| # | Task | Status | Output |
|---|---|---|---|
| 1 | Create `docs/index.md` — project landing page | ✅ Done | `docs/index.md` |
| 2 | Create `docs/getting_started/index.md` — overview with decision tree | ✅ Done | `docs/getting_started/index.md` |
| 3 | Verify MkDocs configuration covers all new pages | ✅ Done | See notes below |
| 4 | Create `.sasva/` directory structure | ✅ Done | `.sasva/` tree |
| 5 | Establish Markdown style guide and admonition conventions | ✅ Done | `.sasva/style/markdown-style-guide.md` |

---

## MkDocs Verification Notes

- **Plugin:** `awesome-nav` is configured in `mkdocs.yaml`. This plugin
  auto-discovers all Markdown files under `docs/` — no explicit `nav:` block
  is required.
- **`docs/index.md`:** Will be served as the site root (`/`). The `hide:
  navigation` and `hide: toc` front-matter matches the pattern used in the
  existing `docs/README.md` landing page.
- **`docs/getting_started/index.md`:** Will be served as the section index for
  the Getting Started tab. The `navigation.indexes` MkDocs Material feature is
  enabled, so `index.md` files are automatically used as section landing pages.
- **`exclude_docs`:** The `mkdocs.yaml` `exclude_docs` block excludes `*.inc.md`
  and `*.template.md` — our new files do not match these patterns and will be
  included correctly.
- **No redirects needed:** The existing `docs/README.md` is the current landing
  page. `docs/index.md` takes precedence as the canonical index when both exist
  (MkDocs prefers `index.md`). The old `README.md` should be reviewed in a
  future phase for consolidation.

---

## Key Decisions

1. **`docs/index.md` vs `docs/README.md`:** Both exist. MkDocs Material with
   `awesome-nav` treats `index.md` as the canonical section index. The existing
   `README.md` will remain for GitHub rendering compatibility; `index.md` is the
   authoritative docs landing page.

2. **Decision tree format:** Used ASCII art tree in a fenced code block for the
   installation decision tree in `getting_started/index.md`. This renders
   correctly in all browsers without JavaScript and is copy-paste friendly.

3. **Grid cards:** Used MkDocs Material `grid cards` admonition pattern for
   feature and section overviews. This requires `attr_list` and `md_in_html`
   extensions, both of which are already enabled in `mkdocs.yaml`.

---

## Dependencies on Future Phases

- Phase 2 should consolidate `docs/README.md` and `docs/index.md` or add a
  redirect from `README.md` to `index.md`.
- The `getting_started/index.md` links to `features/quantization/README.md` —
  verify this path exists in a future phase.

# ADR-001 — Navigation and Page Structure Strategy

**Status:** Accepted  
**Date:** 2026-03-19  
**Deciders:** SASVA AI (Phase 1)

---

## Context

vLLM's MkDocs site uses the `awesome-nav` plugin, which auto-discovers pages
from the filesystem without requiring an explicit `nav:` block in `mkdocs.yaml`.
The `navigation.indexes` feature of MkDocs Material is enabled, meaning `index.md`
files serve as clickable section landing pages in the sidebar.

The existing site has `docs/README.md` as the landing page and no `docs/index.md`.
Several section directories (e.g., `getting_started/`) have no `index.md`.

---

## Decision

1. **Use `index.md` as the canonical section index** for all top-level and
   important sub-sections. This is the MkDocs Material convention and enables
   the `navigation.indexes` feature (section titles are clickable in the nav).

2. **Retain `docs/README.md`** for GitHub repository rendering. Do not delete
   it. In a future phase, consider adding a front-matter redirect or a note
   pointing to the docs site.

3. **Do not add an explicit `nav:` block** to `mkdocs.yaml`. The `awesome-nav`
   plugin handles ordering automatically. If custom ordering is needed in a
   specific section, add a `.nav.yml` file in that directory.

4. **Section overview pages** (`index.md`) should follow the template:
   - Brief description of the section
   - Quick-start or decision-tree content
   - Grid cards linking to sub-pages
   - FAQ (where applicable)

---

## Consequences

- **Positive:** No need to maintain a central `nav:` block — new pages are
  automatically included.
- **Positive:** Section landing pages are navigable and informative, not just
  auto-generated lists.
- **Negative:** Page ordering within sections is alphabetical by default.
  Sections that need custom ordering will require `.nav.yml` files.
- **Risk:** If `docs/README.md` and `docs/index.md` both exist, MkDocs may
  include both in the nav. Monitor and resolve in Phase 2.

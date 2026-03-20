# .sasva — Plan Artifacts Directory

This directory contains planning artifacts, style guides, and tracking documents
produced by SASVA AI during the vLLM documentation project.

## Directory Structure

```
.sasva/
├── README.md                   ← This file
├── plans/                      ← Phase plans and task tracking
│   └── phase-01-foundation.md  ← Phase 1: Foundation & Directory Setup
├── style/                      ← Writing and formatting standards
│   └── markdown-style-guide.md ← Markdown style guide and admonition conventions
└── decisions/                  ← Architecture and content decisions log
    └── adr-001-nav-strategy.md ← ADR: Navigation and page structure strategy
```

## Usage

- **plans/** — One file per documentation phase. Each file tracks tasks, decisions,
  and completion status for that phase.
- **style/** — Canonical style references. All contributors should read
  `markdown-style-guide.md` before writing or reviewing documentation.
- **decisions/** — Architecture Decision Records (ADRs) for significant choices
  made during the documentation project.

## Conventions

- File names use `kebab-case`.
- Phase plan files are named `phase-NN-<slug>.md`.
- ADR files are named `adr-NNN-<slug>.md` and are never deleted (only superseded).

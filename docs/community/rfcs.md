---
description: >
  The vLLM RFC (Request for Comments) process — when to write an RFC, how to
  structure it, and how it moves from proposal to merged code.
---

# RFC process

An **RFC** (Request for Comments) is a design document that proposes a
significant change to vLLM. RFCs create a shared record of the problem being
solved, the alternatives considered, and the rationale for the chosen approach.
They are the primary mechanism for community input on major features before
implementation begins.

---

## When to write an RFC

Not every change needs an RFC. Use the following guidelines:

| Change type | RFC required? |
|---|---|
| Bug fix | No |
| Documentation improvement | No |
| Minor API addition (backward-compatible) | No |
| New quantization format or model architecture | Recommended |
| New major feature (>500 LOC excluding tests/data/config) | **Yes** |
| Changes to the engine core, scheduler, or memory manager | **Yes** |
| New hardware backend | **Yes** |
| Breaking API change | **Yes** |
| Changes to the governance or RFC process itself | **Yes** |

!!! tip
    When in doubt, write an RFC. A short, informal RFC is better than a large
    PR that surprises reviewers. You can always expand the RFC as the design
    matures.

If you submit a large PR without a prior RFC, maintainers may label it
`rfc-required` and ask you to write one before the PR is reviewed.

---

## RFC lifecycle

```
Draft → Open for comment → Accepted / Rejected → Implementation → Closed
```

1. **Draft** — you write the RFC document and open a GitHub Issue.
1. **Open for comment** — the community and maintainers discuss the proposal.
   The RFC author incorporates feedback and updates the document.
1. **Accepted or rejected** — a committer (the assigned DRI) makes a final
   decision, documented in the issue.
1. **Implementation** — the author (or another contributor) implements the
   accepted design and opens a PR referencing the RFC issue.
1. **Closed** — the RFC issue is closed when the implementation is merged or
   the proposal is withdrawn.

---

## How to submit an RFC

### Step 1: Open a GitHub Issue

Go to [github.com/vllm-project/vllm/issues/new/choose](https://github.com/vllm-project/vllm/issues/new/choose)
and select the **RFC** template.

Fill in the template sections described [below](#rfc-template). Give the issue
a clear, descriptive title such as:

> RFC: Disaggregated prefill with KV cache transfer over RDMA

### Step 2: Announce in Slack

Post a link to your RFC in the `#contributors` channel on
[Developer Slack](https://slack.vllm.ai). Tag relevant area owners (see
[committers](../governance/index.md#committers-and-area-owners)) to make sure
the right people see it.

### Step 3: Engage with feedback

Respond to comments on the issue. Update the RFC document as the design
evolves. Mark resolved discussions as resolved to keep the thread readable.

### Step 4: Assignment and decision

For high-interest RFCs, the committer group nominates a **DRI** (Directly
Responsible Individual) — a committer who guides the RFC to a decision and
shepherds the implementation. The DRI is reflected in the issue's **Assignee**
field.

If the RFC is contentious, the lead maintainers make the final call after
hearing from all stakeholders.

### Step 5: Implement and reference

Once the RFC is accepted, open a PR with the implementation. Reference the RFC
issue in the PR description:

```
Implements RFC #<issue-number>
```

---

## RFC template

When you open an RFC issue, use the following structure. You do not need to
answer every section perfectly on day one — mark incomplete sections with
`TBD` and fill them in as the design matures.

```markdown
## Summary

One paragraph describing the proposed change and why it matters.

## Motivation

What problem does this RFC solve? Who is affected and how?
Include concrete use cases and, where possible, quantitative impact
(e.g., latency reduction, memory savings, new hardware support).

## Proposed change

Describe the technical design in enough detail that a committer can evaluate
it without reading the implementation. Include:

- Key data structures and interfaces
- Changes to existing APIs (with before/after examples)
- Interaction with other subsystems (scheduler, memory manager, API server, etc.)
- Configuration options exposed to users

## Alternatives considered

List the alternatives you evaluated and explain why you chose the proposed
approach over each alternative. This section is important — it shows that you
have thought carefully about the design space.

## Compatibility and migration

- Is this change backward-compatible?
- Does it require changes to existing configuration files, CLI flags, or APIs?
- What is the migration path for existing users?

## Testing plan

How will the change be tested? Include:

- Unit tests
- Integration tests
- Performance benchmarks (if applicable)

## Open questions

List any unresolved design questions. These can be answered during the comment
period.

## References

Links to related issues, PRs, papers, or external documentation.
```

---

## Review criteria

Committers evaluate RFCs against vLLM's [design values](../governance/index.md#design-values):

1. **Top performance** — does the change maintain or improve system performance?
   Does it introduce overhead that cannot be justified?
1. **Ease of use** — is the proposed API or configuration intuitive? Does it
   follow existing conventions?
1. **Wide coverage** — does the change work across the supported hardware and
   model landscape, or does it introduce platform-specific behavior?
1. **Production readiness** — can the feature be operated reliably in
   production? Does it degrade gracefully?
1. **Extensibility** — does the design allow future improvements without
   breaking changes?

---

## Roles in the RFC process

| Role | Responsibility |
|---|---|
| **RFC author** | Writes the RFC, responds to feedback, updates the document |
| **Community reviewers** | Anyone who comments on the issue with questions or suggestions |
| **Area owners** | Committers responsible for the affected subsystem; their approval carries extra weight |
| **DRI** (Directly Responsible Individual) | A committer assigned to guide the RFC to a decision and shepherd the implementation |
| **Lead maintainers** | Make the final call on contentious RFCs |

---

## Tips for a successful RFC

- **Start early.** Open an RFC before writing significant code. It is much
  easier to change a design document than to refactor a large implementation.
- **Be concrete.** Vague proposals are hard to evaluate. Include interface
  definitions, configuration examples, and performance estimates where possible.
- **Engage proactively.** Ping area owners directly. Post updates in Slack.
  RFCs that receive no engagement are harder to move forward.
- **Separate concerns.** If your proposal touches multiple subsystems, consider
  splitting it into smaller, focused RFCs.
- **Iterate.** It is normal for an RFC to go through several revisions. Each
  revision should be clearly marked (e.g., with a changelog at the top of the
  document).

---

## Finding open RFCs

Browse open RFCs on GitHub using the `RFC` label:

[github.com/vllm-project/vllm/issues?q=label%3ARFC](https://github.com/vllm-project/vllm/issues?q=label%3ARFC+is%3Aopen)

The [project roadmap](https://roadmap.vllm.ai) also lists high-priority RFCs
that the core team is actively working on.

---

## Related pages

- [Collaboration policy](../governance/collaboration.md) — how vLLM works with model providers and hardware vendors
- [Governance model](../governance/index.md) — decision-making process and maintainer hierarchy
- [Contributing guide](../contributing/index.md) — how to set up your development environment and submit PRs

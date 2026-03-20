---
description: >
  vLLM's governance model — project values, maintainer hierarchy, decision-making
  process, and how to become a committer.
---

# Governance

vLLM's success comes from a strong, open-source community. The project favors
informal, meritocratic norms over rigid formal policies. This page describes
the governance philosophy, the maintainer hierarchy, and the processes used to
make decisions and evolve the project.

---

## Values

vLLM aims to be the fastest and easiest-to-use LLM inference and serving
engine. The project stays current with advances in the field, enables
innovation, and supports diverse models, modalities, and hardware.

### Design values

1. **Top performance** — System performance is the top priority. The team
   monitors overheads, optimizes kernels, and publishes benchmarks. Performance
   is never left on the table.
1. **Ease of use** — vLLM must be simple to install, configure, and operate.
   Clear documentation, fast startup, clean logs, helpful error messages, and
   monitoring guides are all part of this commitment.
1. **Wide coverage** — vLLM supports frontier models and high-performance
   accelerators. Adding new models and hardware should be straightforward.
   vLLM and PyTorch together form a simple interface that avoids unnecessary
   complexity.
1. **Production ready** — vLLM runs 24/7 in production. It must be easy to
   operate and monitor for health issues.
1. **Extensibility** — vLLM serves as fundamental LLM infrastructure. The
   codebase cannot cover every use case, so it is designed for easy forking
   and customization.

### Collaboration values

1. **Tightly knit and fast-moving** — The maintainer team is aligned on vision,
   philosophy, and roadmap. Members work closely to unblock each other and move
   quickly.
1. **Individual merit** — No one buys their way into governance. Committer
   status belongs to individuals, not companies. Contribution, maintenance, and
   project stewardship are what matter.

---

## Maintainer hierarchy

Maintainers form a hierarchy based on sustained, high-quality contributions
and alignment with the project's design philosophy.

### Lead maintainers

Lead maintainers are responsible for the overall direction and strategy of the
project. They make decisions where consensus among core maintainers cannot be
reached, adopt changes to technical governance, and organize the voting process
for new committers.

**Current lead maintainers:**

- Woosuk Kwon ([@WoosukKwon](https://github.com/WoosukKwon))
- Zhuohan Li ([@zhuohan123](https://github.com/zhuohan123))
- Simon Mo ([@simon-mo](https://github.com/simon-mo))
- Kaichao You ([@youkaichao](https://github.com/youkaichao))
- Robert Shaw ([@robertgshaw2-redhat](https://github.com/robertgshaw2-redhat))

### Core maintainers (project leads)

Core maintainers function as a technical steering committee. They meet weekly
to coordinate roadmap priorities and allocate engineering resources.

**Responsibilities:**

- Author quarterly roadmaps and own each development effort.
- Make major changes to the technical direction or scope of vLLM.
- Define the project's release strategy.
- Work with model providers, hardware vendors, and key users to keep the
  project on the right track.

**Current project leads** (in addition to lead maintainers above):

- Tyler Michael Smith ([@tlrmchlsmth](https://github.com/tlrmchlsmth))
- Michael Goin ([@mgoin](https://github.com/mgoin))
- Nick Hill ([@njhill](https://github.com/njhill))
- Roger Wang ([@ywang96](https://github.com/ywang96))
- Lu Fang ([@houseroad](https://github.com/houseroad))
- Ye (Charlotte) Qi ([@yeqcharlotte](https://github.com/yeqcharlotte))
- Yihua Cheng ([@ApostaC](https://github.com/ApostaC))

### Committers and area owners

Committers have write access and merge rights to the repository. They typically
have deep expertise in specific areas and help the community by reviewing PRs,
triaging issues, and improving documentation.

**Responsibilities:**

- Review PRs and provide constructive feedback.
- Address issues and questions from the community.
- Own specific areas of the codebase: reviewing PRs, addressing issues,
  answering questions, and improving documentation.

Almost all committers are also area owners. They author subsystems, review PRs,
refactor code, monitor tests, and ensure compatibility with other areas.

For the full list of committers and their areas, see the
[committers page](committers.md).

---

## Committer proposal process

Committership is highly selective and merit-based. Any committer can nominate
a candidate via the private committer mailing list.

### Selection criteria

A committer candidate typically satisfies at least two of the following:

- Author of an accepted RFC or design that materially shaped project direction
- Measurable, widely adopted performance or reliability improvement in core paths
- Long-term ownership of a subsystem with demonstrable quality and stability gains
- Significant cross-project compatibility or ecosystem enablement work (models,
  hardware, tooling)

While there is no strict quantitative bar, past committers have typically:

- Submitted approximately 30+ PRs of substantial quality and scope
- Provided high-quality reviews of approximately 10+ substantial external
  contributor PRs
- Addressed multiple issues and questions from the community in issues, forums,
  and Slack
- Led concentrated efforts on RFCs and their implementation, or significant
  performance or reliability improvements adopted project-wide

### Nomination steps

1. **Nominate** — A committer sends an email to the committer group nominating
   a candidate, with links to PRs, reviews, RFCs, issues, benchmarks, and
   adoption evidence.
1. **Discuss and vote** — The committer group discusses the nomination, votes,
   and voices any concerns. Shared concerns can pause the process. Most cases
   are decided by consensus; contentious cases are resolved by lead maintainers.
1. **Feedback period** — After a two-week feedback period, if no blocking
   concerns remain, the nominator confirms with the lead maintainer group and
   sends an invitation to the candidate.
1. **Permissions and onboarding** — Lead maintainers assign GitHub permissions
   and add the new member to the committer mailing list, the committer-only
   Slack channel, and other communication channels.
1. **Finalize** — The candidate opens a PR to update `CODEOWNERS` and the
   committers list. Once merged, the new committer is officially welcomed.

---

## Working groups

vLLM runs informal working groups for focused areas of work. These are tracked
via `#sig-` (special interest group) and `#feat-` (feature) channels in
[Developer Slack](https://slack.vllm.ai). Some groups hold regular sync
meetings.

Examples of active working groups:

- CI infrastructure (`#sig-ci`)
- `torch.compile` integration (`#sig-torch-compile`)
- Startup UX
- Structured outputs

---

## Advisory board

Project leads consult with an informal advisory board composed of model
providers, hardware vendors, and ecosystem partners. This manifests as a
collaboration channel in Slack and frequent communications.

---

## Decision-making process

### Project roadmap

Project leads publish quarterly roadmaps as GitHub issues. These clarify
current priorities. Topics not listed are not excluded but may receive less
review attention.

[:octicons-arrow-right-24: roadmap.vllm.ai](https://roadmap.vllm.ai)

### Technical decisions

Technical decisions are made in Slack and GitHub using
[RFCs](../community/rfcs.md) and design documents. Discussion may happen
elsewhere, but significant changes must have a public record of the problem
statement, rationale, and alternatives considered.

### Merging code

- PRs require at least one committer review and approval.
- If the code is covered by `CODEOWNERS`, the PR must be reviewed by the
  relevant code owners.
- Trivial changes and hotfixes may be merged directly by lead maintainers.
- If CI fails for reasons unrelated to the PR, lead maintainers may use the
  "force merge" option to override CI checks.

---

## Communication channels

| Channel | Purpose |
|---|---|
| [GitHub Issues](https://github.com/vllm-project/vllm/issues) | Bug reports, feature requests, RFCs |
| [GitHub Pull Requests](https://github.com/vllm-project/vllm/pulls) | Code review and merging |
| [Developer Slack](https://slack.vllm.ai) | Real-time contributor coordination |
| [User Forum](https://discuss.vllm.ai) | Usage questions and community discussion |
| [standup.vllm.ai](https://standup.vllm.ai) | Weekly contributor sync notes and joining instructions |
| [roadmap.vllm.ai](https://roadmap.vllm.ai) | Quarterly roadmap |

### Slack channels for contributors

- `#contributors` — general contributor discussion
- `#pr-reviews` — PR review coordination
- `#sig-*` — special interest groups
- `#feat-*` — feature working groups

---

## Collaboration with external partners

vLLM has a defined process for collaborating with model providers, hardware
vendors, and other stakeholders. See the
[collaboration policy](collaboration.md) for details on:

- Adding new major features via the RFC process
- Working with model providers on new model architectures
- Adding new hardware backends via the plugin system

---

## Governance pages

| Page | Description |
|---|---|
| [Committers](committers.md) | Full list of active committers and area ownership |
| [Collaboration policy](collaboration.md) | How vLLM works with model providers and hardware vendors |
| [Governance process](process.md) | Detailed governance philosophy and process documentation |

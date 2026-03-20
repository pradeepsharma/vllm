# Release Process

This document describes how vLLM releases are planned, built, tested, and published. It is primarily intended for maintainers and contributors who want to understand the release lifecycle.

---

## Release Philosophy

vLLM releases provide a stable, tested snapshot of the codebase packaged as a Python wheel on [PyPI](https://pypi.org/project/vllm). Releases serve as key milestones for communicating new features, improvements, and breaking changes to the community.

---

## Versioning

vLLM uses a versioning scheme similar to [SemVer](https://semver.org/): `vX.Y.Z`

| Component | Meaning | Example |
|---|---|---|
| `X` (Major) | Architectural milestones with sweeping API changes | `v1.0.0` |
| `Y` (Minor) | Regular releases with new features and bug fixes | `v0.9.0` |
| `Z` (Patch) | New model support, emergency fixes | `v0.9.1` |

### Compatibility Guarantee

Backwards compatibility is guaranteed for a limited number of minor releases. See the [deprecation policy](https://docs.vllm.ai/en/latest/contributing/deprecation_policy) for details.

### Release Cadence

- **Regular (minor) releases**: Every 2 weeks
- **Patch releases**: As needed for critical fixes or new model support
- **Major releases**: Reserved for significant architectural milestones

Past releases are listed at [vllm.ai/releases](https://vllm.ai/releases).

---

## Release Types

### Minor Releases (`vX.Y.0`)

Regular bi-weekly releases that include:
- New features
- Bug fixes
- Performance improvements
- New model support
- Backwards-compatible API changes

### Patch Releases (`vX.Y.Z`, Z > 0)

Special releases for:
- New model support (without other changes)
- Critical performance regressions
- Critical functionality bugs
- Security issues

### Major Releases (`vX.0.0`)

Reserved for sweeping architectural changes (e.g., similar to PyTorch 2.0). These are rare and involve extensive planning.

---

## Release Branch

Each release is built from a dedicated release branch named `releases/vX.Y.Z`.

### Branch Cut

- **Minor releases**: Branch cut 1–2 days before the release goes live
- **Patch releases**: Reuse the previously cut release branch

After the branch cut, the team monitors `main` for reverts and applies them to the release branch as needed.

### Cherry-Pick Policy

After branch cut, only specific types of changes are allowed into the release branch:

| Allowed | Description |
|---|---|
| ✅ Regression fixes | Fixes for functional/performance regressions vs. the previous release |
| ✅ Critical fixes | Silent incorrectness, backwards compatibility, crashes, deadlocks, large memory leaks |
| ✅ New feature fixes | Fixes for features introduced in the most recent release |
| ✅ Documentation improvements | Doc fixes and improvements |
| ✅ Release-specific changes | Version identifiers, CI fixes |
| ❌ Feature work | **Not allowed** — all features must land on `main` first |

All cherry-pick candidates must be merged on `main` first (except release-specific changes).

---

## Release Build Process

### Step 1: Create the Release Branch

```bash
# Create the release branch from main
git checkout main
git pull upstream main
git checkout -b releases/v0.9.0
git push upstream releases/v0.9.0
```

### Step 2: Push an RC Tag

Release builds are triggered by pushing an RC (release candidate) tag:

```bash
git tag v0.9.0-rc1
git push upstream v0.9.0-rc1
```

This triggers the Buildkite release pipeline (`.buildkite/release-pipeline.yaml`), which builds Python wheels for all supported platforms.

### Step 3: Wheel Build Matrix

The release pipeline builds wheels for:

| Architecture | CUDA Version | Platform Tag |
|---|---|---|
| x86_64 | 12.9 | `manylinux_2_31` |
| x86_64 | 13.0 | `manylinux_2_35` |
| x86_64 | CPU | `manylinux_2_35` |
| aarch64 | 12.9 | `manylinux_2_35` |
| aarch64 | 13.0 | `manylinux_2_35` |
| aarch64 | CPU | `manylinux_2_35` |

Each wheel is built using Docker with `DOCKER_BUILDKIT=1` and uploaded to the nightly wheels repository.

### Step 4: Performance Validation

Before finalizing a release, end-to-end performance validation is required to ensure no regressions.

#### Getting Access

Request write access to [pytorch/pytorch-integration-testing](https://github.com/pytorch/pytorch-integration-testing) to run the benchmark workflow.

#### Running Benchmarks

Navigate to the [vllm-benchmark workflow](https://github.com/pytorch/pytorch-integration-testing/actions/workflows/vllm-benchmark.yml) and configure:

- **vLLM branch**: Set to the release branch (e.g., `releases/v0.9.2`)
- **vLLM commit**: Set to the RC commit hash

#### Benchmark Coverage

| Models | Hardware |
|---|---|
| Llama3, Llama4, Mixtral | NVIDIA H100 |
| Llama3, Llama4, Mixtral | AMD MI300x |

#### Reviewing Results

Once the workflow completes, results appear on the [vLLM benchmark dashboard](https://hud.pytorch.org/benchmark/llms?repoName=vllm-project%2Fvllm). Compare against the previous release to verify no performance regressions.

### Step 5: Final Release Tag

After all RCs pass validation, push the final release tag:

```bash
git tag v0.9.0
git push upstream v0.9.0
```

The final tag does **not** trigger a build — it's used for:
- GitHub Release notes
- Release assets
- Version tracking

### Step 6: Publish to PyPI

The validated wheels from the RC build are published to PyPI under the final version number.

### Step 7: Announce the Release

- Update the [Google Doc release notes draft](https://docs.google.com/document/d/1YyVqrgX4gHTtrstbq8oWUImOyPCKSGnJ7xtTpmXzlRs/edit)
- Post announcement in vLLM Slack
- Create GitHub Release with release notes

---

## Contributing to a Release

### Targeting a Release

If you want your PR to be included in a specific release:

1. Merge your PR to `main` before the branch cut
2. After branch cut, request a cherry-pick by commenting on your PR or opening an issue

### Requesting a Cherry-Pick

To request a cherry-pick after branch cut:

1. Ensure your PR is already merged to `main`
2. Open an issue or comment on the release tracking issue
3. Explain why the change meets the cherry-pick criteria
4. A maintainer will apply the cherry-pick if approved

### Updating Release Notes

If your change is user-facing, update the release notes:

1. Open the [Google Doc release notes draft](https://docs.google.com/document/d/1YyVqrgX4gHTtrstbq8oWUImOyPCKSGnJ7xtTpmXzlRs/edit)
2. Add your change under the appropriate section:
   - **New Features**
   - **Performance Improvements**
   - **Bug Fixes**
   - **Breaking Changes**
   - **New Models**

---

## Deprecation Policy

vLLM follows a structured deprecation process:

1. **Announce deprecation** — Add a deprecation warning in the code and document in release notes
2. **Maintain for N releases** — Keep the deprecated feature working for a defined number of minor releases
3. **Remove** — Remove the feature after the deprecation period

See the full [deprecation policy](https://docs.vllm.ai/en/latest/contributing/deprecation_policy) for details.

---

## Nightly Builds

In addition to official releases, vLLM publishes nightly wheels built from the `main` branch. These are available at the nightly wheels repository and are useful for testing the latest changes before they appear in an official release.

```bash
# Install the latest nightly build
pip install vllm --pre --extra-index-url https://wheels.vllm.ai/nightly/
```

---

## Version Numbering in Code

vLLM uses `setuptools-scm` to automatically derive the version from Git tags:

```toml
# pyproject.toml
[tool.setuptools_scm]
# no extra settings needed, presence enables setuptools-scm
```

The version is available at runtime:

```python
import vllm
print(vllm.__version__)  # e.g., "0.9.0"
```

For development builds (between tags), the version includes a commit hash:

```
0.9.0.dev123+gabcdef1
```

---

## Release Checklist

For maintainers managing a release:

### Pre-Release

- [ ] All planned features are merged to `main`
- [ ] No known critical bugs
- [ ] Release notes draft is up to date
- [ ] Branch cut performed 1–2 days before release

### RC Phase

- [ ] RC tag pushed (`vX.Y.Z-rc1`)
- [ ] Wheel builds complete for all platforms
- [ ] Performance benchmarks run on H100 and MI300x
- [ ] No performance regressions vs. previous release
- [ ] Critical bugs from RC testing fixed and cherry-picked
- [ ] Additional RC tags pushed if needed (`rc2`, `rc3`, ...)

### Release

- [ ] Final tag pushed (`vX.Y.Z`)
- [ ] Wheels published to PyPI
- [ ] GitHub Release created with release notes
- [ ] Announcement posted in Slack and community channels
- [ ] Documentation updated (if needed)

---

## Related Resources

- [Past Releases](https://vllm.ai/releases)
- [PyPI Package](https://pypi.org/project/vllm)
- [Benchmark Dashboard](https://hud.pytorch.org/benchmark/llms?repoName=vllm-project%2Fvllm)
- [CI/CD Pipeline](ci_cd.md)
- [Deprecation Policy](https://docs.vllm.ai/en/latest/contributing/deprecation_policy)

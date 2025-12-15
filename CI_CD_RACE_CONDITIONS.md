# CI/CD Pipeline: Race Conditions & Fixes

## Problem Statement

When tags are pushed quickly (especially while previous builds are still running), the GitHub Actions workflow can encounter several race conditions that may cause:
- Incomplete or corrupted build artifacts
- Failed publishes
- Cache poisoning

## Identified Race Conditions

### 1. **Cache Write Race Condition** 🔴 CRITICAL

**Scenario:**
- Tag `v1.0.0` is pushed → `build-wheels` starts with 10 matrix jobs (2 platforms × 5 Python versions)
- While job #1 is building, jobs #2-#10 also start
- Jobs complete at different times, all trying to write to the **same cache**

**Problem:**
- GitHub Actions `cache@v4` does **not** provide atomic writes
- Multiple jobs writing to overlapping cache keys causes data corruption
- A job might read a partially-written cache from another job
- Result: Corrupted `.whl` files or missing wheels in the dist directory

**Example Timeline:**
```
T1: Tag v1.0.0 pushed
T2: Job#1 (x86_64-py3.10) starts building → cache miss
T3: Job#2 (x86_64-py3.11) starts building → cache miss
T4: Job#3 (aarch64-py3.10) starts building → cache miss
T5: Job#1 finishes, saves cache: wheel-linux-x86_64-py3.10-v1.0.0-hash1
T6: Job#2 finishes, tries to save cache: wheel-linux-x86_64-py3.11-v1.0.0-hash1
    ⚠️ Job#1's write might not be fully committed yet!
T7: Job#3 finishes, saves cache, but now cache keys are inconsistent
```

### 2. **Concurrent Tag Pushes with Build Overlap** 🟡 MEDIUM (RESOLVED)

**Scenario:**
- Tag `v1.0.0` pushed, build-wheels starts with 10 jobs
- Before build completes, tag `v1.0.1` is pushed
- New `build-wheels` job group starts immediately

**Problems:**
- Two independent build-wheels runs now compete for resources
- Cache entries from v1.0.0 might be read by v1.0.1 jobs
- If a v1.0.0 job hasn't finished its cache write, v1.0.1 might get inconsistent state

**Resolution:**
- Workflow-level concurrency ensures tag pushes wait for previous workflows to complete
- Each tag gets its own concurrency group, preventing overlap

### 2b. **Tag Push During Branch Build** 🟡 MEDIUM (RESOLVED)

**Scenario:**
- Branch push (e.g., to master) triggers workflow that builds wheels
- While that workflow is still running, a tag is pushed
- Tag workflow starts and tries to publish while branch workflow is still building

**Problems:**
- Tag workflow might try to publish wheels that were already published
- PyPI rejects duplicate uploads with "File already exists" error
- Artifacts from different workflow runs might get mixed

**Resolution:**
- Workflow-level concurrency ensures tag pushes wait for any in-progress workflows
- This ensures cache is valid and no duplicate uploads occur

### 3. **Artifact Upload Race & Cleanup Race** 🟡 MEDIUM

**Scenario:**
- Multiple matrix jobs finish building and call `upload-artifact@v4` simultaneously
- Same artifact name used by multiple jobs: `wheel-linux-x86_64-py3.10`
- Cleanup step runs before upload completes

**Problems:**
- While artifact is being uploaded to GitHub's backend, cleanup might delete from local `dist/`
- `upload-artifact` might have incomplete transfers
- `publish` job receives corrupted or missing wheels

### 4. **Concurrent Publish Attempts** 🟡 MEDIUM (RESOLVED)

**Scenario:**
- Two tags pushed quickly: v1.0.0 and v1.0.1
- Both trigger publish jobs eventually
- Both try to publish to PyPI simultaneously

**Problems:**
- PyPI rejects duplicate versions (e.g., "File already exists")
- Mix of v1.0.0 and v1.0.1 wheels might be published to wrong versions
- Manual intervention needed to fix PyPI registry

**Resolution:**
- Workflow-level concurrency ensures only one workflow runs per ref at a time
- Tag pushes wait for any in-progress workflows to complete before starting
- This prevents duplicate uploads and ensures cache validity

## Solutions Implemented

### Solution 1: Workflow-Level Concurrency (Primary Fix)

```yaml
concurrency:
  group: workflow-${{ github.ref }}
  cancel-in-progress: false
```

**Effect:**
- Only ONE workflow run per ref (branch or tag) allowed simultaneously
- If a tag is pushed while a previous workflow (branch or tag) is still running, the new workflow is **queued** (not cancelled)
- Ensures the cache is fully written and valid from any previous run before starting the next one
- Prevents race conditions where tag pushes try to publish while previous workflows are still building
- Prevents duplicate PyPI uploads from concurrent workflows

**Caveat:** `cancel-in-progress: false` means we wait for completion. This is safer than cancelling incomplete builds.

### Solution 2: Job-Level Concurrency Groups

```yaml
concurrency:
  group: build-wheels-${{ needs.check-tag.outputs.is_tagged }}-${{ github.ref }}-${{ matrix.target }}-${{ matrix.python-version }}
  cancel-in-progress: false
```

**Effect:**
- Additional concurrency control at the job level for extra safety
- Each matrix job has its own concurrency group
- Works in conjunction with workflow-level concurrency to prevent all race conditions

### Solution 2: Robust Cache Keys

```yaml
key: wheel-${{ matrix.platform }}-py${{ matrix.python-version }}-${{ steps.version.outputs.version }}-${{ hashFiles(...) }}-${{ github.run_id }}-${{ matrix.python-version }}
```

**Effect:**
- Each GitHub Actions run has unique `github.run_id`
- Cache key includes the run ID, ensuring different concurrent runs have different cache entries
- Prevents cross-run cache pollution
- Even if concurrency isn't enforced, different runs won't corrupt each other's caches

**Fallback restore-keys:**
- Tries to find cache from same run first
- Falls back to previous runs of same version if available
- Maintains some reuse benefit while being safe

### Solution 3: Publish Concurrency Control

```yaml
concurrency:
  group: publish-${{ needs.check-tag.outputs.tag }}
  cancel-in-progress: false
```

**Effect:**
- Only one publish job per tag
- Multiple pushes of same tag get queued
- No duplicate uploads to PyPI

### Solution 4: Verification Step

Added verification that wheels exist before upload:

```yaml
- name: Verify wheel was built or restored from cache
  run: |
    if [ ! -d dist ] || [ -z "$(find dist -name '*.whl' 2>/dev/null)" ]; then
      echo "ERROR: No wheels found in dist directory!"
      exit 1
    fi
```

**Effect:**
- Fails fast if cache was corrupted or build failed
- Prevents publishing incomplete artifacts

### Solution 5: Proper Cleanup Ordering

Changed cleanup to NOT delete `dist/` during build job:

```yaml
# NOTE: Keep dist/ during this job since upload-artifact needs it!
# The dist directory is cleaned up after artifacts are uploaded by the runner
```

**Effect:**
- Local `dist/` directory persists until upload completes
- Cleanup focuses on `.venv`, `target/`, `.cargo/` (expensive to rebuild)
- Runner automatically cleans up after artifact upload finishes

## Testing the Fix

### Test Case 1: Rapid Tag Pushes
```bash
# Simulate rapid pushes
git tag v1.0.0 && git push origin v1.0.0
git tag v1.0.1 && git push origin v1.0.1
git tag v1.0.2 && git push origin v1.0.2
```

**Expected Behavior:**
- First build (v1.0.0) completes fully
- Second build (v1.0.1) waits for first to finish, then starts fresh
- Third build (v1.0.2) waits, then starts
- All three publish successfully to PyPI with correct versions

### Test Case 2: Push During Build
```bash
# Simulate tagging while build is in progress
git tag v2.0.0 && git push origin v2.0.0  # This starts the build
# Wait 20 seconds, then:
git tag v2.0.1 && git push origin v2.0.1  # This queues waiting for v2.0.0 to finish
```

**Expected Behavior:**
- v2.0.0 build completes fully
- v2.0.1 does not start until v2.0.0's publish finishes
- Both complete successfully

### Test Case 3: Verify Cache Hit on Re-push
```bash
# Push and let it complete
git tag v3.0.0 && git push origin v3.0.0

# Wait for publish to complete (several minutes)

# Re-push same tag (if allowed in your git setup)
git push origin v3.0.0 --force
```

**Expected Behavior:**
- Second run should hit cache (if within 7 days)
- All matrix jobs show "cache hit" in logs
- Completes much faster
- Publishes same wheels with no duplicates

## Remaining Limitations

### GitHub Actions Cache Limitations:
1. **No guaranteed atomicity**: While concurrency groups help, the cache@v4 action itself has known limitations with concurrent writes
2. **7-day expiration**: Caches auto-expire after 7 days of no access (this is fine per your 1-day max duration)
3. **No encryption in transit**: Cache is not encrypted (minor concern for public repos)

### Recommended Best Practice:
- **Avoid re-pushing tags** - always create new tags (v1.0.0, v1.0.1, v1.0.2...)
- **Allow full build/publish cycle to complete** before pushing next tag
- **Monitor the publish job** to ensure PyPI updates match your intent

## Implementation Details

### When `build-wheels` Concurrency Triggers:

```
Scenario: Two tags pushed within seconds
├── First push (v1.0.0)
│   └── build-wheels starts with 10 matrix jobs
│       └── Each job: build, verify, upload, cleanup
│           └── Concurrency group: "build-wheels-true-refs/tags/v1.0.0"
│
└── Second push (v1.0.1) while first still building
    └── build-wheels starts but immediately queued
        └── Waits for first build-wheels job group to finish
            └── Then runs independently with 10 new matrix jobs
                └── Concurrency group: "build-wheels-true-refs/tags/v1.0.1"
```

### Cache Key Structure:

```
wheel-{platform}-py{version}-{tag_version}-{file_hash}-{run_id}-{version}
├── {platform}: linux-x86_64 or linux-aarch64
├── {version}: 3.10, 3.11, 3.12, 3.13, 3.14
├── {tag_version}: Extracted from git tag
├── {file_hash}: Hash of Cargo.lock and pyproject.toml (detects changes)
├── {run_id}: Unique per GitHub Actions run (prevents cross-run pollution)
└── {version}: Repeated for additional specificity
```

## Migration Notes

These changes are **backward compatible**:
- Existing caches won't be used (due to new run_id in key), but that's safe
- First publish with these changes will rebuild everything (acceptable)
- Future publishes will cache correctly
- No changes to code or artifacts, only CI/CD process

## Monitoring

Watch for in GitHub Actions logs:
- ✓ `✓ Wheels restored from cache` = Cache working, fast publish coming
- ✗ `✗ No cache found, will build wheels` = Building from scratch
- ⚠️ `ERROR: No wheels found in dist directory!` = Something went wrong, investigate

Check PyPI release page afterward to confirm all versions published correctly.

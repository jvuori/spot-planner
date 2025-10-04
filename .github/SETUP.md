# GitHub Actions Setup Guide

This guide explains how to set up the GitHub Actions workflow for building and publishing `spot-planner` to PyPI.

## Prerequisites

1. **PyPI Account**: You need a PyPI account to publish packages
2. **PyPI Project**: The project must be registered on PyPI
3. **GitHub Repository**: The code must be in a GitHub repository

## Setup Steps

### 1. Configure PyPI OIDC Trusted Publishing

1. Go to [PyPI](https://pypi.org) and log in
2. Navigate to Account Settings → API tokens
3. Click "Add API token"
4. Select "Create a token for trusted publishing"
5. Configure the trusted publisher:
   - **PyPI project name**: `spot-planner`
   - **Owner**: Your GitHub username or organization
   - **Repository name**: `spot-planner`
   - **Workflow filename**: `publish.yml`
   - **Environment name**: `pypi` (optional)
6. Click "Add token"

### 2. Create PyPI Environment

1. Go to your GitHub repository
2. Navigate to Settings → Environments
3. Click "New environment"
4. Name: `pypi`
5. Click "Configure environment"

**Optional Environment Protection:**

- **Required reviewers**: Add team members who must approve PyPI releases
- **Wait timer**: Add a delay before publishing (useful for rollback)
- **Deployment branches**: Restrict which branches can trigger publishing

### 3. Test the Workflow

The workflow will automatically run on:

- **Every push to master branch**: Builds wheels and runs tests
- **Tagged commits**: Builds wheels, runs tests, AND publishes to PyPI

To test publishing:

```bash
# Create and push a tag
git tag v0.1.0
git push origin v0.1.0

# Or push to master to trigger build-only
git push origin master
```

## Workflow Details

### Jobs Overview

1. **check-tag**: Determines if the current commit is tagged
2. **build-wheels**: Builds native wheels for AMD64 and ARM64 Linux
3. **publish**: Publishes to PyPI (only for tagged releases)
4. **test-build**: Runs tests on every push to master

### Supported Platforms

- **AMD64 Linux** (`x86_64-unknown-linux-gnu`): Standard x86_64 systems
- **ARM64 Linux** (`aarch64-unknown-linux-gnu`): Raspberry Pi 4/5, ARM servers

### Version Management

The workflow automatically updates version numbers from git tags:

- Tag `v1.2.3` → Version `1.2.3`
- Tag `1.2.3` → Version `1.2.3`
- Updates both `pyproject.toml` and `Cargo.toml`

### Artifacts

Each build creates:

- **Wheel files**: `spot_planner-{version}-cp3*-{platform}.whl`
- **Source distribution**: `spot-planner-{version}.tar.gz`

## Troubleshooting

### Common Issues

1. **Build fails on ARM64**:

   - Check that cross-compilation dependencies are installed
   - Verify linker settings in the workflow

2. **Publishing fails**:

   - Verify `PYPI_API_TOKEN` secret is set correctly
   - Check that the version doesn't already exist on PyPI
   - Ensure the tag format is correct

3. **Tests fail**:
   - Check that all dependencies are installed
   - Verify the Rust module builds correctly

### Manual Publishing

If you need to publish manually:

```bash
# Build wheels locally
maturin build --release --target x86_64-unknown-linux-gnu
maturin build --release --target aarch64-unknown-linux-gnu

# Build source distribution
uv build --sdist

# Upload to PyPI (requires API token for manual upload)
uv publish dist/*
```

## Security Notes

- **OIDC Authentication**: No long-lived tokens stored in GitHub secrets
- **Trusted Publishing**: PyPI verifies the GitHub Actions workflow and environment
- **Automatic Rotation**: OIDC tokens are short-lived and automatically rotated
- **Least Privilege**: Only the specific repository and workflow can publish

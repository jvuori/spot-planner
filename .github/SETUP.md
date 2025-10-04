# GitHub Actions Setup Guide

This guide explains how to set up the GitHub Actions workflow for building and publishing `spot-planner` to PyPI.

## Prerequisites

1. **PyPI Account**: You need a PyPI account to publish packages
2. **PyPI API Token**: Generate an API token for automated publishing
3. **GitHub Repository**: The code must be in a GitHub repository

## Setup Steps

### 1. Create PyPI API Token

1. Go to [PyPI](https://pypi.org) and log in
2. Navigate to Account Settings → API tokens
3. Click "Add API token"
4. Give it a name like "spot-planner-publish"
5. Set scope to "Entire account" (or create a specific project scope)
6. Copy the token (it starts with `pypi-`)

### 2. Add PyPI Token to GitHub Secrets

1. Go to your GitHub repository
2. Navigate to Settings → Secrets and variables → Actions
3. Click "New repository secret"
4. Name: `PYPI_API_TOKEN`
5. Value: Paste your PyPI API token
6. Click "Add secret"

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

# Upload to PyPI
UV_PUBLISH_TOKEN=your_token uv publish dist/*
```

## Security Notes

- The `PYPI_API_TOKEN` secret is only accessible to the repository
- Tokens should be rotated regularly
- Use project-scoped tokens when possible instead of account-wide tokens

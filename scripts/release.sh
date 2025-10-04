#!/bin/bash

# Release script for spot-planner
# Usage: ./scripts/release.sh <version>
# Example: ./scripts/release.sh 1.0.0

set -e

if [ $# -eq 0 ]; then
    echo "Usage: $0 <version>"
    echo "Example: $0 1.0.0"
    exit 1
fi

VERSION=$1
TAG="v$VERSION"

echo "Releasing version $VERSION..."

# Check if we're on master branch
CURRENT_BRANCH=$(git branch --show-current)
if [ "$CURRENT_BRANCH" != "master" ]; then
    echo "Error: Must be on master branch to release. Current branch: $CURRENT_BRANCH"
    exit 1
fi

# Check if working directory is clean
if [ -n "$(git status --porcelain)" ]; then
    echo "Error: Working directory is not clean. Please commit or stash changes."
    exit 1
fi

# Check if tag already exists
if git tag -l | grep -q "^$TAG$"; then
    echo "Error: Tag $TAG already exists"
    exit 1
fi

# Update version in files
echo "Updating version to $VERSION..."
sed -i "s/^version = \".*\"/version = \"$VERSION\"/" pyproject.toml
sed -i "s/^version = \".*\"/version = \"$VERSION\"/" Cargo.toml

# Commit version update
git add pyproject.toml Cargo.toml
git commit -m "Bump version to $VERSION"

# Create and push tag
echo "Creating tag $TAG..."
git tag -a "$TAG" -m "Release $VERSION"
git push origin master
git push origin "$TAG"

echo "Release $VERSION created and pushed!"
echo "GitHub Actions will now build and publish the package to PyPI using uv."
echo ""
echo "You can monitor the progress at:"
echo "https://github.com/$(git config --get remote.origin.url | sed 's/.*github.com[:/]\([^/]*\/[^/]*\)\.git/\1/')/actions"

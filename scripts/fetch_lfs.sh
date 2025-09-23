#!/usr/bin/env bash
# Helper to fetch Git LFS files if git-lfs is available.
set -e
if command -v git-lfs >/dev/null 2>&1; then
  echo "git-lfs found, pulling LFS files..."
  git lfs pull || { echo "git lfs pull failed"; exit 1; }
else
  echo "git-lfs not installed. Install it to fetch large data files: https://git-lfs.github.com/"
fi

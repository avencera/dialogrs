#!/usr/bin/env bash
# Clones scriptrs and speakrs next to this checkout, where Cargo.toml's path
# dependencies expect them
#
# Each sibling uses the first branch that exists of the pushed or PR head
# branch, the PR base branch, and master, so a change that spans the three
# repositories builds against its matching sibling branches
set -euo pipefail

workspace="$(cd "$(dirname "$0")/../.." && pwd)"

for repo in scriptrs speakrs; do
  url="https://github.com/avencera/${repo}"
  ref=master
  # a full ref, since `ls-remote` matches suffixes and `heads/x` would match `x`
  for candidate in "${HEAD_REF:-}" "${BASE_REF:-}"; do
    if [[ -n "$candidate" ]] && git ls-remote --exit-code "$url" "refs/heads/${candidate}" >/dev/null; then
      ref="$candidate"
      break
    fi
  done

  echo "${repo}: ${ref}"
  git clone --quiet --depth 1 --branch "$ref" "$url" "${workspace}/../${repo}"
done

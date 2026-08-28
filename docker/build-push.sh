#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
experiment_dir="$(cd "$script_dir/.." && pwd)"
nx_worktree="${NX_WORKTREE:-$(cd "$experiment_dir/../.." && pwd)/nx-upstream-main}"
image="${IMAGE:-fa3-tp-nx:local}"

if [[ ! -d "$nx_worktree/exla" || ! -d "$nx_worktree/nx" ]]; then
  printf 'NX_WORKTREE is not an Nx monorepo: %s\n' "$nx_worktree" >&2
  exit 1
fi

context="$(mktemp -d "${TMPDIR:-/tmp}/fa3-image-context.XXXXXX")"
trap 'rm -rf "$context"' EXIT

mkdir -p "$context/experiment" "$context/nx-overlay"

rsync -a \
  --exclude .git \
  --exclude _build \
  --exclude deps \
  "$experiment_dir/" "$context/experiment/"

overlay_files=(
  exla/Makefile
  exla/c_src/exla_test/custom_calls.cc
  exla/lib/exla/custom_call/spec.ex
  exla/lib/exla/defn.ex
  exla/lib/exla/defn/buffers.ex
  exla/lib/exla/mlir/value.ex
  exla/test/exla/custom_call_operation_attributes_test.exs
  exla/test/exla/defn/sharding_test.exs
  exla/test/support/exla_test_operation_attributes_block.ex
)

(
  cd "$nx_worktree"
  rsync -aR "${overlay_files[@]}" "$context/nx-overlay/"
)

cp "$script_dir/Dockerfile" "$context/Dockerfile"

build_args=(
  --platform linux/amd64
  --tag "$image"
  --file "$context/Dockerfile"
  --progress plain
)

if [[ "${PUSH:-0}" == 1 ]]; then
  build_args+=(--push --provenance=true --sbom=true)
else
  build_args+=(--load)
fi

docker buildx build "${build_args[@]}" "$context"

printf 'IMAGE=%s\n' "$image"

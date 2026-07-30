#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

usage() {
  cat <<'EOF'
Usage: bash scripts/env/create_conda_envs.sh [all|notebooks-gpu|figures-cpu|esmc-gpu|chemberta-gpu]

Creates CatRange conda environments from envs/*.yml.

Recommended:
  bash scripts/env/create_conda_envs.sh notebooks-gpu
  bash scripts/env/create_conda_envs.sh esmc-gpu
  bash scripts/env/create_conda_envs.sh chemberta-gpu

The ESM-C and ChemBERTa embedding environments are intentionally separate.
EOF
}

create_env() {
  local file="$1"
  echo "[CatRange] Creating/updating environment from ${file}"
  conda env create -f "${ROOT_DIR}/${file}" || conda env update -f "${ROOT_DIR}/${file}" --prune
}

target="${1:-all}"
case "${target}" in
  all)
    create_env envs/catrange-notebooks-gpu.yml
    create_env envs/catrange-esmc-gpu.yml
    create_env envs/catrange-chemberta-gpu.yml
    ;;
  notebooks-gpu)
    create_env envs/catrange-notebooks-gpu.yml
    ;;
  figures-cpu)
    create_env envs/catrange-cpu-figures.yml
    ;;
  esmc-gpu)
    create_env envs/catrange-esmc-gpu.yml
    ;;
  chemberta-gpu)
    create_env envs/catrange-chemberta-gpu.yml
    ;;
  -h|--help|help)
    usage
    ;;
  *)
    usage
    exit 2
    ;;
esac

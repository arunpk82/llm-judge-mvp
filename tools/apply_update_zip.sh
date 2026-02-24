#!/usr/bin/env bash
set -euo pipefail

print_usage() {
  cat <<'EOF'
Usage:
  tools/apply_update_zip.sh --zip <path> [options]

Required:
  --zip <path>            Path to update ZIP file

Options:
  --branch <name>         Branch to create/switch to (default: update/<zip-basename>-<timestamp>)
  --yes                   Non-interactive: skip prompts and proceed
  --no-install            Skip 'poetry install'
  --no-check              Skip quality gates (ruff/mypy/pytest)
  --commit-msg <msg>      If provided, auto-commit with this message after checks pass
  --keep-tmp              Do not delete temp directory (debug)
  -h, --help              Show this help
EOF
}

die() { echo "ERROR: $*" >&2; exit 1; }
need_cmd() { command -v "$1" >/dev/null 2>&1 || die "Missing required command: $1"; }

confirm() {
  local prompt="$1"
  if [[ "${YES:-0}" == "1" ]]; then return 0; fi
  read -r -p "$prompt [y/N]: " ans
  [[ "$ans" == "y" || "$ans" == "Y" ]]
}

ZIP_PATH=""
BRANCH=""
YES=0
NO_INSTALL=0
NO_CHECK=0
COMMIT_MSG=""
KEEP_TMP=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --zip) ZIP_PATH="${2:-}"; shift 2 ;;
    --branch) BRANCH="${2:-}"; shift 2 ;;
    --yes) YES=1; shift ;;
    --no-install) NO_INSTALL=1; shift ;;
    --no-check) NO_CHECK=1; shift ;;
    --commit-msg) COMMIT_MSG="${2:-}"; shift 2 ;;
    --keep-tmp) KEEP_TMP=1; shift ;;
    -h|--help) print_usage; exit 0 ;;
    *) die "Unknown arg: $1" ;;
  esac
done

[[ -n "$ZIP_PATH" ]] || { print_usage; die "Missing --zip"; }
[[ -f "$ZIP_PATH" ]] || die "ZIP not found: $ZIP_PATH"

git rev-parse --is-inside-work-tree >/dev/null 2>&1 || die "Run this from inside a git repo."
REPO_ROOT="$(git rev-parse --show-toplevel)"

need_cmd unzip
need_cmd rsync
need_cmd git

ZIP_BASENAME="$(basename "$ZIP_PATH")"
STAMP="$(date +%Y%m%d_%H%M%S)"
DEFAULT_BRANCH="update/${ZIP_BASENAME%.zip}-${STAMP}"
BRANCH="${BRANCH:-$DEFAULT_BRANCH}"

echo "Repo:   $REPO_ROOT"
echo "Zip:    $ZIP_PATH"
echo "Branch: $BRANCH"
echo

if ! git diff --quiet || ! git diff --cached --quiet; then
  die "Working tree is not clean. Commit or stash changes first."
fi

if git show-ref --verify --quiet "refs/heads/$BRANCH"; then
  echo "Switching to existing branch: $BRANCH"
  git checkout "$BRANCH"
else
  echo "Creating branch: $BRANCH"
  git checkout -b "$BRANCH"
fi

TMP_DIR="$(mktemp -d -t llm_judge_update_XXXXXX)"
trap '[[ "$KEEP_TMP" == "1" ]] || rm -rf "$TMP_DIR"' EXIT

echo "Unzipping to: $TMP_DIR"
unzip -q "$ZIP_PATH" -d "$TMP_DIR"

echo
echo "=== Package contents (top-level) ==="
find "$TMP_DIR" -maxdepth 3 -type f | sed "s|$TMP_DIR/||" | sort || true
echo

echo "=== RSYNC DRY RUN (preview) ==="
rsync -avun "$TMP_DIR"/ "$REPO_ROOT"/
echo

if ! confirm "Apply these changes to repo?"; then
  die "Aborted by user."
fi

echo "=== APPLY RSYNC ==="
rsync -avu "$TMP_DIR"/ "$REPO_ROOT"/
echo

echo "=== Git status after sync ==="
cd "$REPO_ROOT"
git status --porcelain || true
echo

if [[ "$NO_INSTALL" == "0" ]]; then
  need_cmd poetry
  echo "=== poetry install ==="
  poetry install
  echo
fi

if [[ "$NO_CHECK" == "0" ]]; then
  need_cmd poetry
  echo "=== Quality gates ==="
  poetry run ruff format .
  poetry run ruff check . --fix
  poetry run mypy .
  poetry run pytest
  echo
else
  echo "Skipping checks (--no-check)."
  echo
fi

if [[ -n "$COMMIT_MSG" ]]; then
  if ! git diff --quiet || ! git diff --cached --quiet; then
    echo "=== Auto-commit ==="
    git add -A
    git commit -m "$COMMIT_MSG"
    echo "Committed: $COMMIT_MSG"
  else
    echo "No changes to commit."
  fi
else
  echo "No auto-commit. If everything looks good:"
  echo '  git add -A'
  echo '  git commit -m "<message>"'
fi

echo
echo "Done. Next:"
echo "  git push -u origin $BRANCH"

#!/bin/bash
# Custom gitleaks hook that scans staged files directly

# Find gitleaks binary (pre-commit cache or system)
if [ -f "/Users/alexwoolford/.cache/pre-commit/repojs95mj5x/golangenv-system/bin/gitleaks" ]; then
    GITLEAKS_BIN="/Users/alexwoolford/.cache/pre-commit/repojs95mj5x/golangenv-system/bin/gitleaks"
elif command -v gitleaks >/dev/null 2>&1; then
    GITLEAKS_BIN="gitleaks"
else
    # Try to find it in common pre-commit cache locations
    GITLEAKS_BIN=$(find ~/.cache/pre-commit -name "gitleaks" -type f 2>/dev/null | head -1)
    if [ -z "$GITLEAKS_BIN" ]; then
        echo "❌ Gitleaks not found. Please install gitleaks or run: pre-commit install"
        exit 1
    fi
fi

# Get the repo root directory
REPO_ROOT=$(git rev-parse --show-toplevel)
CONFIG="${REPO_ROOT}/.gitleaks.toml"

# If config doesn't exist, use default
if [ ! -f "$CONFIG" ]; then
    CONFIG=""
fi

# Get list of staged files
STAGED_FILES=$(git diff --cached --name-only --diff-filter=ACM)

if [ -z "$STAGED_FILES" ]; then
    exit 0
fi

# Create a temporary directory to stage files for scanning
TMP_DIR=$(mktemp -d)
trap "rm -rf $TMP_DIR" EXIT

# Copy staged files to temp directory (preserving structure)
for file in $STAGED_FILES; do
    # Get staged content using git show
    STAGED_CONTENT=$(git show ":$file" 2>/dev/null)
    if [ $? -eq 0 ] && [ -n "$STAGED_CONTENT" ]; then
        mkdir -p "$TMP_DIR/$(dirname "$file")"
        echo "$STAGED_CONTENT" > "$TMP_DIR/$file"
    elif [ -f "$file" ]; then
        # Fallback to working directory file
        mkdir -p "$TMP_DIR/$(dirname "$file")"
        cp "$file" "$TMP_DIR/$file"
    fi
done

# Run gitleaks detect on the temp directory
cd "$TMP_DIR"

# Build gitleaks command
# IMPORTANT: We intentionally don't use the config file for pre-commit scanning.
# This ensures strict secret detection at commit time. The .gitleaks.toml config
# is still used in CI for scanning full git history, where allowlisting may be needed
# for historical commits or known false positives.
GITLEAKS_CMD=("$GITLEAKS_BIN" detect --source . --no-git --verbose --exit-code 1)

# Run gitleaks and capture output
GITLEAKS_OUTPUT=$("${GITLEAKS_CMD[@]}" 2>&1)
EXIT_CODE=$?

if [ $EXIT_CODE -ne 0 ]; then
    echo "$GITLEAKS_OUTPUT"
    echo ""
    echo "❌ Gitleaks found secrets in staged files. Commit blocked."
    echo "   Review the findings above and remove secrets before committing."
    exit 1
fi

EXIT_CODE=$?

if [ $EXIT_CODE -ne 0 ]; then
    echo "❌ Gitleaks found secrets in staged files. Commit blocked."
    exit 1
fi

exit 0


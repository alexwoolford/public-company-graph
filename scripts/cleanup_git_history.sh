#!/bin/bash
# Script to clean git history of old database files
# This removes url_checker.db from all git history and keeps only domain_status.db

set -e

echo "⚠️  WARNING: This script will rewrite git history!"
echo "This is a destructive operation. Make sure you:"
echo "  1. Have a backup of your repository"
echo "  2. Have pushed any important work to a remote"
echo "  3. Are ready to force-push after this completes"
echo ""
read -p "Continue? (yes/no): " confirm

if [ "$confirm" != "yes" ]; then
    echo "Aborted."
    exit 1
fi

# Check if git-filter-repo is installed
if ! command -v git-filter-repo &> /dev/null; then
    echo "❌ git-filter-repo is not installed."
    echo "Install it with: pip install git-filter-repo"
    echo "Or use Homebrew: brew install git-filter-repo"
    exit 1
fi

echo ""
echo "Step 1: Removing url_checker.db from git history..."
git filter-repo --path data/url_checker.db --invert-paths --force

echo ""
echo "Step 2: Verifying url_checker.db is removed from history..."
if git log --all --full-history -- "data/url_checker.db" | grep -q .; then
    echo "⚠️  Warning: url_checker.db still found in history"
else
    echo "✓ url_checker.db successfully removed from history"
fi

echo ""
echo "Step 3: Checking current tracked files..."
echo "Database files in git:"
git ls-files | grep -E '\.(db|sqlite)$' || echo "  (none found)"

echo ""
echo "✅ Git history cleanup complete!"
echo ""
echo "Next steps:"
echo "  1. Remove url_checker.db from working directory (if you want):"
echo "     rm data/url_checker.db"
echo ""
echo "  2. Add domain_status.db to git:"
echo "     git add data/domain_status.db"
echo "     git commit -m 'Add domain_status.db database'"
echo ""
echo "  3. Force push to remote (if you have one):"
echo "     git push --force --all"
echo "     git push --force --tags"
echo ""
echo "⚠️  IMPORTANT: All collaborators must re-clone the repository after this!"


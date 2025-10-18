#!/bin/bash
# Auto-commit and push Greene work
# Usage: Run this on Greene after each work session
# ./backup_work.sh "optional commit message"

set -e

cd /scratch/ah7072/GTRS

# Check if git is configured
if ! git config user.email > /dev/null 2>&1; then
    echo "⚙️  Configuring git..."
    git config user.email "ah7072@nyu.edu"
    git config user.name "Ali Hamza"
fi

# Default commit message
COMMIT_MSG="${1:-Auto-backup thesis work $(date '+%Y-%m-%d %H:%M')}"

echo "📦 Backing up work to git..."

# Add thesis-specific files
git add -A navsim/agents/thesis_agents/ 2>/dev/null || true
git add -A navsim/planning/script/config/common/agent/ 2>/dev/null || true
git add -A scripts/ 2>/dev/null || true
git add -A summaries/ 2>/dev/null || true

# Check if there are changes to commit
if git diff --staged --quiet; then
    echo "✅ No changes to backup"
    exit 0
fi

# Show what will be committed
echo "📝 Changes to backup:"
git diff --staged --stat

# Commit
git commit -m "$COMMIT_MSG"

# Push to remote (if configured)
if git remote get-url origin > /dev/null 2>&1; then
    echo "☁️  Pushing to remote..."
    git push origin HEAD || echo "⚠️  Push failed - you may need to configure git remote"
else
    echo "⚠️  No remote configured. Commit is local only."
    echo "💡 To push to GitHub: git remote add origin <your-repo-url>"
fi

echo "✅ Backup complete!"

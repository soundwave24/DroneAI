#!/bin/bash
# This script pulls the latest changes, adds all files, commits with a timestamp, and pushes to the remote repository.
# Ensure you have the correct permissions and remote repository set up.
# Usage: Run this script in the root directory of your git repository.
# Make sure to run this script in the directory where your git repository is initialized.
# cd /path/to/your/repository || { echo "Directory not found!"; exit 1; }
# Use provided argument as commit message, or default to "Auto commit"
COMMIT_MSG="${1:-Auto commit}"

git pull
git add .
git commit -m "$COMMIT_MSG: $(date +'%Y-%m-%d %H:%M:%S')"
git push
if [ $? -ne 0 ]; then
    echo "Git push failed. Please check your repository settings."
else
    echo "Changes pushed successfully."
fi

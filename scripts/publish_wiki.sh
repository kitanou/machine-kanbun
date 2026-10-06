#!/bin/bash
# Publish wiki/*.md to the GitHub Wiki of this repository (Issue #45).
# Prerequisite: the wiki git repository must exist. GitHub creates it only when the first page is created in the web UI
# (Repository -> Wiki -> "Create the first page" -> save). After that:  scripts/publish_wiki.sh
# The wiki/ directory is the source of truth for the wiki pages; this script overwrites the wiki's pages with it.
set -e
cd "$(dirname "$0")/.."
REMOTE=${WIKI_REMOTE:-https://github.com/kitanou/machine-kanbun.wiki.git}
TMP=$(mktemp -d)
git clone --quiet "$REMOTE" "$TMP/wiki"
cp wiki/*.md "$TMP/wiki/"
cd "$TMP/wiki"
git add -A
if git diff --cached --quiet; then echo "wiki is already up to date"; exit 0; fi
git commit --quiet -m "Update wiki from wiki/ in the main repository"
git push --quiet origin HEAD
echo "published $(ls *.md | wc -l) pages to $REMOTE"

#!/bin/sh
# docs/_page.html is the page body (also published as a claude.ai artifact); this wraps it for GitHub Pages.
cd "$(dirname "$0")"
{
  printf '<!doctype html>\n<html lang="en">\n<head>\n<meta charset="utf-8">\n<meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover">\n</head>\n<body>\n'
  cat _page.html
  printf '\n</body>\n</html>\n'
} > index.html
echo "wrote docs/index.html"

#!/bin/bash

SOURCE="${BASH_SOURCE[0]}"
while [ -h "$SOURCE" ]; do # resolve $SOURCE until the file is no longer a symlink
  DIR="$( cd -P "$( dirname "$SOURCE" )" >/dev/null 2>&1 && pwd )"
  SOURCE="$(readlink "$SOURCE")"
  [[ $SOURCE != /* ]] && SOURCE="$DIR/$SOURCE" # if $SOURCE was a relative symlink, we need to resolve it relative to the path where the symlink file was located
done
DIR="$( cd -P "$( dirname "$SOURCE" )" >/dev/null 2>&1 && pwd )"

TEXLIVE_DIR=/usr/local/texlive
TEXLIVE_BIN_DIR="$TEXLIVE_DIR/2020/bin/x86_64-linux/"

$DIR/install-tools.sh

jupyter nbconvert --to markdown $DIR/../thesis.ipynb
export PATH=$PATH:$TEXLIVE_BIN_DIR && pandoc -f markdown -s $DIR/../thesis.md -t pdf -o $DIR/../thesis.pdf --filter pandoc-crossref --filter pandoc-citeproc --bibliography=citations.bib --csl=$DIR/../citation-styles/aip.csl -M date="$(date "+%B %e, %Y")" --template $DIR/../templates/eisvogel-m --listings --number-sections --highlight-style breezedark

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

sudo chmod 777 ~/.jupyter
command -v $TEXLIVE_BIN_DIR/tlmgr >/dev/null 2>&1 && command -v pandoc >/dev/null 2>&1 && echo "Tools already installed. :-)" && exit 0

TMP_DIR=~/tmp
mkdir -p $TMP_DIR

echo "Installing TexLive:"
wget -O $TMP_DIR/tl.tar.gz http://mirror.ctan.org/systems/texlive/tlnet/install-tl-unx.tar.gz
tar -xzf $TMP_DIR/tl.tar.gz -C $TMP_DIR
sudo mkdir -p $TEXLIVE_DIR
sudo chown $USER:$GROUP $TEXLIVE_DIR 
sudo perl $TMP_DIR/install-tl-*/install-tl --profile $DIR/../config/texlive.profile
ln -s $TEXLIVE_BIN_DIR ~/bin
sudo $TEXLIVE_BIN_DIR/tlmgr init-usertree
sudo $TEXLIVE_BIN_DIR/tlmgr install adjustbox pgf tikz-cd float caption xkeyval xcolor setspace koma-script listings babel-german background bidi collectbox csquotes everypage filehook footmisc footnotebackref framed fvextra letltxmacro ly1 mdframed mweights needspace pagecolor sourcecodepro sourcesanspro titling ucharcat ulem unicode-math upquote xecjk xurl zref etoolbox


echo "\n\nInstalling Pandoc and filters:"
wget -O $TMP_DIR/pandoc.deb https://github.com/jgm/pandoc/releases/download/2.9.2.1/pandoc-2.9.2.1-1-amd64.deb
sudo dpkg -i $TMP_DIR/pandoc.deb
wget -O $TMP_DIR/pandoc-crossref.tar.xz https://github.com/lierdakil/pandoc-crossref/releases/download/v0.4.0.0-alpha6d/pandoc-crossref-Linux-2.9.2.1.tar.xz
sudo tar -xf $TMP_DIR/pandoc-crossref.tar.xz -C /usr/bin/ pandoc-crossref

rm -rf $TMP_DIR


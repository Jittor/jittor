#!/bin/bash
bpath=$(dirname "${BASH_SOURCE[0]}")
cd $bpath
cd ..
pwd
git fetch --all
git reset --hard origin/master
python3.7 -c "import jittor"
#! /bin/bash

set -x

folders=(
  "cfg.llvm"
  "cdfg.programl/programl.5000.01"
  "cdfg.programl/programl.5000.02"
  "cdfg.programl/programl.5000.03"
  "cdfg.programl/programl.5000.04"
  "cdfg.programl/programl.5000.05"
  "cdfg.programl/programl.5000.06"
)

echo 'Cloning CCPE-DADOS'
git clone https://github.com/andrefz/ccpe-dados.git ccpe-dados

cd ccpe-dados

for i in "${folders[@]}"; do
  echo "Unpacking ${i}..."
  cd $i
  mkdir -p extracted
  unzip -q -o -d extracted \*.zip
  cd $OLDPWD
  echo "Done"
done

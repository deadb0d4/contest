#!/bin/sh

set -e -x

mkdir -p bin
mkdir -p test/{buf,input,output,perf}
touch test/buf/{in,out,err}.txt
./run.py -r

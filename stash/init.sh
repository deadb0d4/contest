set -e -x

mkdir bin
mkdir -p test/buf
mkdir test/input
mkdir test/output
touch test/buf/{in,out,err}.txt
./run.py -r

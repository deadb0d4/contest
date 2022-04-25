set -e -x

mkdir -p bin
mkdir -p test/{buf,input,output,stress}
touch test/buf/{in,out,err}.txt
./run.py -r

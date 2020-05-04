sh compile.sh

for file in *.txt; do
  echo "-- Test case: $file"
  ./bin < $file
done

# Competetive Programming Helper

## Roadmap

- [x] Support stress testing
- [x] Test generators
- [x] l10n

## Useful materials

These are placed in `{lib, stash}`.

## Runner

### To add test:

```
./run.py -t
```
vim window will be opened and you need to fill input and expected output.

### Check solution:

```
./run.py main.cpp
```
output and expected will be compared line by line.

**Remark:** Supported languages are `c++` and `python3`.

### Run solution without tests:

```
./run.py main.cpp -s
```

### Run solution on tests 0 and 1:

```
./run.py main.cpp -s 0 1
```

### Reset tests:

```
./run.py -r
```

### Reset everything:

```
./run.py -rr
```

## Grunt

`grunt.py` is a small script to use generator scripts for producting perf and test inputs. It isn't tested well so use it on your own risk. Short recipes are included in `grunt.py -h`.

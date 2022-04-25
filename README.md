# Competetive Programming Helper

## Roadmap

- [x] ~~Separate static tests and checker~~ Instead, one tool to use
- [ ] Support stress testing
- [x] Test generators

## Useful materials

These are placed in `./lib/` see also `./stash/`'ed content.

## Tool

1. To add test:
```
./run.py -t
```
vim window will be opened and you need to fill input and expected output.

2. Check solution:
```
./run.py main.cpp
```
output and expected will be compared line by line.

**Remark:** Supported languages are `c++` and `python3`.

3. Run solution without tests:
```
./run.py main.cpp -s
```

4. Run solution on tests 0 and 1:
```
./run.py main.cpp -s 0 1
```

5. Reset tests:
```
./run.py -r
```

6. Reset everything:
```
./run.py -rr
```

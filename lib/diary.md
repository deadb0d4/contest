# Notes to keep in mind

**TODO:** Start new notes in a more helpful format

## General or ad hoc

- Define priorities for greedy algorithms accurately

- The first to use object may have biggest priority and vice versa

- Weaken constraints and try to extend a method or an approach to the problem

- Suboptimal solutions may relax constraints without changing the answer

- Fix a value for any part of the target function and then optimize the rest under this constraint

- Stretching argument: try to start from a basic solution and extend it greedily multiple times (e.g. consider the argmin position)

- Try graph corner cases: trees and complete graphs

- Using dfs tree might be useful for a general graph

- Scale (even logarithmic scale) big and small numbers, compare them roughly

- Shrinking by encoding exploits the sparseness of data

- Pivoting divide-and-conquer may decrease complexity

- Sliding window may become simple yet useful step

- Distinguish building a solution from dynamics on its existence: once you have information on existence it may be simple to construct one (e.g. greedily)

- Dijkstra and bfs are flexible (multi-source and 0-1 (front\back) bfs)

- Skip moves in games maybe crucial

- Standard dynamics can be modified with just a few additional transitions to consider

- Fix a parameter on prefixes (e.g. balance in parenthesis sequence) and use it as a dimension in dynamic programming

- Use distribution like function to sum up (expectation-like approach)

- Sum of subsets dp is based on prefix-like dp approach (`dp[mask][i]` submasks that differ only at first `i` bits)

- Trees may have a persistent analogue (as segment tree, for example)

- Convex hull trick is quite ubiquitous

- Implicit treap can be quite flexible for mass array operations

## Greedy

Greedy algorithm makes locally _optimal_ decisions to achieve the best possible scenario. It is often applicable for a concrete strategy if in any optimal solution the first action which is different from the greedy one we can alter it _without changing_ the remaining decisions (e.g. Kruskal).

## Offline segment queries

One can leverage offline queries by using at the very least:

_1._ Mo's algorithm (sqrt optimization for left or right ends)

_2._ One can process queries iterating by the right end and maintaining answers if:
  - You can answer for a fixed right end
  - You can reuse data to increment right end

---

## Heavy-light style

Dynamics update with "adding" a new child, instead of calculating all and merging all at once. This is effective especially with choosing the centre of a tree. This is the same trick as in the DSU rank heuristic.

---

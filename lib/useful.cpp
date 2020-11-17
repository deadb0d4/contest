#include <bits/stdc++.h>

// lets get funky
using namespace std;


namespace Treap {

// Implicit Treap
struct Node {
  const uint64_t p;
  int count;
  int value;
 
  Node* left;
  Node* right;
 
  Node(int v) : p(Random<uint64_t>()), count(1), value(v), left(nullptr), right(nullptr) {}
};
 
int Size(Node* node) {
  return (node ? node->count : 0);
}
 
void Update(Node* node) {
  node->count = Size(node->left) + Size(node->right) + 1;
}
 
pair<Node*, Node*> Split(Node* node, int k) {
  if (k == 0) {
    return { nullptr, node };
  }
  if (Size(node) <= k) {
    return { node, nullptr };
  }
  pair<Node*, Node*> res;
  if (Size(node->left) >= k) {
    tie(res.first, node->left) = Split(node->left, k);
    res.second = node;
  } else {
    tie(node->right, res.second) = Split(node->right, k - Size(node->left) - 1);
    res.first = node;
  }
  Update(node);
  return res;
}
 
Node* Merge(Node* f, Node* s) {
  if (!f) return s;
  if (!s) return f;
 
  Node* root = nullptr;
  if (f->p < s->p) {
    f->right = Merge(f->right, s);
    root = f;
  } else {
    s->left = Merge(f, s->left);
    root = s;
  }
  Update(root);
  return root;
}

} // namespace Treap
 
//////////////////////////////////////////////////////////////////////////

auto FirstSmallerRight(const vector<int>& vec) {
  stack<int> s;
  vector<int> res(vec.size(), vec.size());
  for (int i = 0; i < vec.size(); ++i) {
    while (s.size() and vec[s.top()] > vec[i]) {
      res[s.top()] = i;
      s.pop();
    }
    s.push(i);
  }
  return res;
}

auto FirstSmallerLeft(const vector<int>& vec) {
  stack<int> s;
  vector<int> res(vec.size(), -1);
  for (int i = vec.size(); i --> 0; ) {
    while (s.size() and vec[s.top()] > vec[i]) {
      res[s.top()] = i;
      s.pop();
    }
    s.push(i);
  }
  return res;
}

//////////////////////////////////////////////////////////////////////////

constexpr int64_t kMod = 1; // Change that

auto Gcd(int64_t a, int64_t b) {
  if (a == 0) {
    int64_t x = 0;
    int64_t y = 1;
    return make_tuple(b, x, y);
  }
  auto [d, x1, y1] = Gcd(b % a, a);
  int64_t x = y1 - (b / a) * x1;
  int64_t y = x1;
  // a * x + b * y = 1
  // gcd(a, b) = d
  return make_tuple(d, x, y);
}

int64_t Mul(int64_t a, int64_t b) {
  return (a * b) % kMod;
}

int64_t Sum(int64_t a, int64_t b) {
  return (a + b) % kMod;
}

int64_t Can(int64_t a) {
  return ((a % kMod) + kMod) % kMod;
}

int64_t Inv(int64_t a) {
  auto [d, x, y] = Gcd(a, kMod);
  if (d != 1) {
    return -1;
  }
  return Can(x);
}

// the list of primes <= n
auto Sieve(int n) {
  vector<bool> pr(n + 1, true);
  pr[0] = pr[1] = false;
  for (int64_t p = 2; p <= n; ++p) {
    if (pr[p] and p * p <= n) {
      for (int64_t d = p * p; d <= n; d += p) {
        pr[d] = false;
      }
    }
  }
  vector<int> res;
  for (int p = 2; p <= n; ++p) if (pr[p]) {
    res.push_back(p);
  }
  return res;
}

// Lazy factorization:
// pv -- a list of primes up to sqrt(n)
// fv -- factors of n
// dv -- degrees of fv
void Factorize(int n,
               vector<int>& fv, vector<int>& dv,
               const vector<int>& pv, int pos = 0) {
  if (n == 1) {
    return;
  }
  if (pos == pv.size()) {
    fv.push_back(n);
    dv.push_back(1);
    return;
  }
  while (pos < pv.size() and (n % pv[pos]) != 0) {
    ++pos;
  }
  if (pos < pv.size()) {
    int d = 0;
    while ((n % pv[pos]) == 0) {
      n /= pv[pos];
      ++d;
    }
    fv.push_back(pv[pos]);
    dv.push_back(d);
  }
  Factorize(n, fv, dv, pv, pos);
}

int64_t SmPow(int64_t p, int64_t a) {
  int64_t res = 1;
  while (a) {
    if (a % 2) {
      res = Mul(res, p);
    }
    p = Mul(p, p);
    a /= 2;
  }
  return res;
}

// next divisor (beware cycling)
void Inc(
    int& d, int& nonzero_count, vector<int>& sv,
    const vector<int>& pr,
    const vector<int>& mx) {
  int pos = 0;
  while (pos < sv.size()) {
    if (sv[pos] == mx[pos]) {
      d /= SmPow(pr[pos], mx[pos]);
      sv[pos++] = 0;
      nonzero_count--;
    } else {
      d *= pr[pos];
      sv[pos]++;
      nonzero_count++;
      return;
    }
  }
}

//////////////////////////////////////////////////////////////////////////

// Simple segment tree: query min + addition on a segment as an update
struct SegTree {
  vector<int64_t> t;
  vector<int64_t> changed;

  SegTree(int len) : t(4 * len, 0), changed(4 * len, 0) {
  }

  void Push(int v, int vl, int vr) {
    t[v] += changed[v];
    if (vl != vr) {
      changed[2 * v] += changed[v];
      changed[2 * v + 1] += changed[v];
    }
    changed[v] = 0;
  }

  int64_t Query(int l, int r, int v = 1, int vl = 0, int vr = -1) {
    if (vr == -1) {
      vr = t.size() / 4 - 1;
    }
    Push(v, vl, vr);
    if (l == vl and r == vr) {
      return t[v];
    }
    int m = (vl + vr) / 2;
    if (l > m) {
      return Query(l, r, 2 * v + 1, m + 1, vr);
    } else if (r <= m) {
      return Query(l, r, 2 * v, vl, m);
    }
    return min(Query(l, m, 2 * v, vl, m),
               Query(m + 1, r, 2 * v + 1, m + 1, vr));
  }

  void Update(int l, int r, int64_t val, int v = 1, int vl = 0, int vr = -1) {
    if (vr == -1) {
      vr = t.size() / 4 - 1;
    }
    Push(v, vl, vr);
    if (l > r) {
      return;
    }
    if (l == vl and r == vr) {
      changed[v] = val;
      Push(v, vl, vr);
      return;
    }
    int m = (vl + vr) / 2;
    Update(l, min(r, m), val, 2 * v, vl, m);
    Update(max(m + 1, l), r, val, 2 * v + 1, m + 1, vr);
    t[v] = min(t[2 * v], t[2 * v + 1]);
  }
};

//////////////////////////////////////////////////////////////////////////

// Cheap dsu with checkpoints
struct DSU {
  vector<tuple<int, int, int>> hist;
  int count;
  vector<int> pv;
  vector<int> sv;

  DSU(int len) : hist(), count(len), pv(len), sv(len, 1) {
    for (int i = 0; i < len; ++i) {
      pv[i] = i;
    }
  }

  int Get(int x) {
    if (pv[x] != x) {
      return Get(pv[x]);
    }
    return x;
  }

  void Union(int a, int b) {
    a = Get(a);
    b = Get(b);
    if (a == b) {
      return;
    }
    if (sv[a] < sv[b]) {
      swap(a, b);
    }
    hist.emplace_back(2, 0, count);
    --count;
    hist.emplace_back(0, b, pv[b]);
    pv[b] = a;
    hist.emplace_back(1, a, sv[a]);
    sv[a] += sv[b];
  }

  void Checkpoint() {
    hist.emplace_back(-1, 0, 0);
  }

  void Rollback() {
    while (get<0>(hist.back()) != -1) {
      auto [t, i, v] = hist.back();
      if (t == 0) {
        pv[i] = v;
      } else if (t == 1) {
        sv[i] = v;
      } else if (t == 2) {
        count = v;
      }
      hist.pop_back();
    }
    hist.pop_back();
  }
};


// Simple dsu
struct DSU {
  int count;
  vector<int> r;
  vector<int> p;

  DSU(int len) : count(len), r(len, 1), p(len) {
    for (int i = 0; i < len; ++i) {
      p[i] = i;
    }
  }

  int Get(int x) {
    return p[x] = (p[x] == x ? x : Get(p[x]));
  }

  void Union(int a, int b) {
    a = Get(a);
    b = Get(b);
    if (a == b) {
      return;
    }
    --count;
    if (r[a] == r[b]) {
      ++r[a];
    }
    if (r[a] > r[b]) {
      p[b] = a;
    } else {
      p[a] = b;
    }
  }
};

//////////////////////////////////////////////////////////////////////////

// (Offline) Graham's convex hull

double OrientedArea(double x1, double y1, double x2, double y2) {
  return x1 * y2 - y1 * x2;
}

auto UpperConvexHull(const vector<pair<double, double>>& pts) {
  // pts is sorted as a pair vector...
  vector<pair<double, double>> res = {pts[0]};
  int pos = 1;
  while (pos < pts.size()) {
    while (res.size() > 1 and
           OrientedArea(
               pts[pos].first - res[res.size() - 1].first,
               pts[pos].second - res[res.size() - 1].second,
               res[res.size() - 1].first - res[res.size() - 2].first,
               res[res.size() - 1].second - res[res.size() - 2].second) <= 0) {
      res.pop_back();
    }
    res.push_back(pts[pos++]);
  }
  return res;
}

auto LowerConvexHull(const vector<pair<double, double>>& pts) {
  // pts is sorted as a pair vector...
  vector<pair<double, double>> res = {pts[0]};
  int pos = 1;
  while (pos < pts.size()) {
    while (res.size() > 1 and
           OrientedArea(
               pts[pos].first - res[res.size() - 1].first,
               pts[pos].second - res[res.size() - 1].second,
               res[res.size() - 1].first - res[res.size() - 2].first,
               res[res.size() - 1].second - res[res.size() - 2].second) >= 0) {
      res.pop_back();
    }
    res.push_back(pts[pos++]);
  }
  return res;
}

//////////////////////////////////////////////////////////////////////////

// Cheap persistent seg tree: a single setter and min + max queries
struct Node {
  int mn = numeric_limits<int>::max();
  int mx = -1;
  int vl;
  int vr;
  shared_ptr<Node> left = nullptr;
  shared_ptr<Node> right = nullptr;

  void Build(int b, int e) {
    vl = b;
    vr = e;

    if (b != e) {
      int m = (b + e) / 2;

      left = make_shared<Node>();
      left->Build(b, m);

      right = make_shared<Node>();
      right->Build(m + 1, e);
    }
  }

  shared_ptr<Node> Set(int pos, bool val) {
    auto new_node = make_shared<Node>();
    new_node->vl = vl;
    new_node->vr = vr;
    new_node->left = left;
    new_node->right = right;

    if (vl != vr) {
      int m = (vl + vr) / 2;
      if (pos <= m) {
        new_node->left = left->Set(pos, val);
      } else {
        new_node->right = right->Set(pos, val);
      }
      new_node->mn = min(new_node->left->mn, new_node->right->mx);
      new_node->mx = max(new_node->left->mx, new_node->right->mx);
    } else if (val) {
      new_node->mn = new_node->mx = vl;
    }
    return new_node;
  }

  int Min(int l, int r) {
    if (vl == l and vr == r) {
      return mn;
    }
    int m = (vl + vr) / 2;
    if (r <= m) {
      return left->Min(l, r);
    }
    if (m + 1 <= l) {
      return right->Min(l, r);
    }
    return min(left->Min(l, m), right->Min(m + 1, r));
  }

  int Max(int l, int r) {
    if (vl == l and vr == r) {
      return mx;
    }
    int m = (vl + vr) / 2;
    if (r <= m) {
      return left->Max(l, r);
    }
    if (m + 1 <= l) {
      return right->Max(l, r);
    }
    return max(left->Max(l, m), right->Max(m + 1, r));
  }
};

//////////////////////////////////////////////////////////////////////////

// Euler path traversal
void EulerDfs(int v, const vector<vector<int>>& g, vector<int>& buf,
              vector<int>& pos) {
  while (pos[v] < g[v].size()) {
    // The thing is to keep deleting edges while
    // you can and then backtrack (via call stack)
    // to last vertex with available edges...
    EulerDfs(g[v][pos[v]++], g, buf, pos);
  }
  buf.push_back(v);
}

//////////////////////////////////////////////////////////////////////////

struct MinSparseTable {
  vector<int> pt;
  vector<vector<int>> t;

  MinSparseTable(const vector<int>& vec) : pt(vec.size(), -1), t(vec.size()) {
    Build(vec);
  }

  void Build(const vector<int>& vec) {
    for (int i = 0; i < vec.size(); ++i) {
      t[i].push_back(vec[i]);
    }
    int k = 1;
    while ((1 << k) <= vec.size()) {
      for (int i = 0; i + (1 << k) <= vec.size(); ++i) {
        t[i].push_back(min(t[i].back(), t[i + (1 << (k - 1))].back()));
      }
      ++k;
    }
    for (int i = 0; i < vec.size(); ++i) {
      int k = 0;
      while ((1 << (k + 1)) <= (i + 1)) {
        ++k;
      }
      pt[i] = k;
    }
  }

  // min on [l, r)
  int Query(int l, int r) {
    int p = pt[r - l - 1];
    return min(t[l][p], t[r - (1 << p)][p]);
  }
};

//////////////////////////////////////////////////////////////////////////

/*
Returns two arrays:
  1. pv -- starting indices of prefixes in sorted order
  2. lcp -- longest common prefix between adjacent prefixes in pv
*/
auto SuffixArray(const string& str) {
  vector<int> pv(str.size() + 1);
  for (int i = 0; i < pv.size(); ++i) {
    pv[i] = i;
  }
  auto char_at = [&str] (int pos) {
    if (pos == str.size()) {
      return '$'; // beware of smaller chars
    }
    return str[pos];
  };
  sort(begin(pv), end(pv), [&char_at] (int l, int r) {
    return char_at(l) < char_at(r);
  });
  vector<int> inv(str.size() + 1);
  inv[str.size()] = 0;
  for (int p = 1; p < pv.size(); ++p) {
    inv[pv[p]] = inv[pv[p - 1]];
    if (char_at(pv[p - 1]) < char_at(pv[p])) {
      ++inv[pv[p]];
    }
  }
  auto count_sort = [] (const auto& vec, auto key_fn) {
    vector<int> cv(vec.size(), 0);
    for (auto&& x : vec) {
      cv[key_fn(x)]++;
    }
    int s = 0;
    for (int i = 0; i < vec.size(); ++i) {
      int cc = cv[i];
      cv[i] = s;
      s += cc;
    }
    auto res = vec;
    for (auto&& x : vec) {
      res[cv[key_fn(x)]++] = x;
    }
    return res;
  };
  for (int shift = 1; shift < str.size(); shift *= 2) {
    auto pre = inv;
    for (auto&& p : pv) {
      p = (str.size() + 1 + p - shift) % (str.size() + 1);
    }
    pv = count_sort(pv, [&pre] (int p) { return pre[p]; });
    auto key_fn = [&pre, shift, mod = str.size() + 1] (int p) {
      return make_pair(pre[p], pre[(p + shift) % mod]);
    };
    for (int p = 1; p < pv.size(); ++p) {
      inv[pv[p]] = inv[pv[p - 1]];
      if (key_fn(pv[p - 1]) < key_fn(pv[p])) {
        inv[pv[p]]++;
      }
    }
  }
  vector<int> lcp(str.size(), 0);
  for (int p = 0; p < str.size(); ++p) {
    int i = inv[p];
    int q = pv[i - 1];
    while (char_at(q + lcp[i - 1]) == char_at(p + lcp[i - 1])) {
      ++lcp[i - 1];
    }
    if (p + 1 < str.size()) {
      lcp[inv[p + 1] - 1] = max(lcp[inv[p + 1] - 1], lcp[i - 1] - 1);
    }
  }
  return make_pair(pv, lcp);
}

//////////////////////////////////////////////////////////////////////////

// Knuth-Morris-Pratt
auto PrefixFunction(const string& str) {
  vector<int> pref(str.size(), 0);

  for (int i = 1; i < str.size(); ++i) {
    pref[i] = (str[i] == str[0]);
    for (int j = pref[i - 1]; j > 0; j = pref[j - 1]) {
      if (str[j] == str[i]) {
        pref[i] = j + 1;
        break;
      }
    }
  }
  return pref;
}

auto Zfunction(const string& str) {
  int l = 0, r = 0;

  vector<int> z(str.size(), 0);
  for (int i = 1; i < str.size(); ++i) {
    if (i <= r) {
      z[i] = min(z[i - l], r - i + 1);
    }
    while (z[i] + i < str.size() and str[z[i]] == str[i + z[i]]) {
      ++z[i];
    }
    if (r < i + z[i] - 1) {
      l = i;
      r = i + z[i] - 1;
    }
  }
  return z;
}

//////////////////////////////////////////////////////////////////////////

// author: adamant

// The idea is that to build vertex hld
// while usual Euler traversal.
// In particular, vertices of the 'v' hld path lie
// in {in[nxt[v]], in[v]} and vertices of the 'v' subtree
// lie in {in[v], out[v] - 1}

int t;
vector<vector<int>> g;
vector<int> sz;
vector<int> in, out;

// the highest point of pathes for vertices
vector<int> nxt;

void dfs_sz(int v = 0) {
  sz[v] = 1;
  for (auto& u : g[v]) {
    dfs_sz(u);
    sz[v] += sz[u];
    // lazy heavy path
    if (sz[u] > sz[g[v][0]]) {
      swap(u, g[v][0]);
    }
  }
}

void dfs_hld(int v = 0) {
  in[v] = t++;
  for (auto u : g[v]) {
    nxt[u] = (u == g[v][0] ? nxt[v] : u);
    dfs_hld(u);
  }
  out[v] = t;  // no increment here
}

//////////////////////////////////////////////////////////////////////////

template <class T>
using MinHeap = priority_queue<T, vector<T>, greater<T>>;

template <class T>
using MaxHeap = priority_queue<T, vector<T>, less<T>>;

//////////////////////////////////////////////////////////////////////////

namespace std {

template <class U, class V>
struct hash<pair<U, V>> {
  size_t operator()(const pair<U, V>& p) const {
    return hash<U>()(p.first) ^ (hash<V>()(p.second) << 1);
  }
};

}  // namespace std

struct SplitMix64Hash {
  static uint64_t splitmix64(uint64_t x) {
    // http://xorshift.di.unimi.it/splitmix64.c
    x += 0x9e3779b97f4a7c15;
    x = (x ^ (x >> 30)) * 0xbf58476d1ce4e5b9;
    x = (x ^ (x >> 27)) * 0x94d049bb133111eb;
    return x ^ (x >> 31);
  }

  size_t operator()(uint64_t x) const {
    static const uint64_t salt =
        chrono::steady_clock::now().time_since_epoch().count();
    return SplitMix64Hash::splitmix64(x + salt);
  }
};

struct PythonHash {
  static uint64_t string_hash(const std::string& str) {
    if (str.empty()) {
      return 0;
    }
    uint64_t x = str[0] << 7;
    for (const char& c : str) {
      x = (1000003 * x) ^ c;
    }
    x ^= str.size();
    if (x == -1) {
      x = -2;
    }
    return x;
  }

  size_t operator()(const string& x) const {
    static const string salt =
        to_string(chrono::steady_clock::now().time_since_epoch().count());
    return string_hash(x + salt);
  }
};

template <class Value>
using integer_hash_map = unordered_map<int, Value, SplitMix64Hash>;

template <class Value>
using string_hash_map = unordered_map<string, Value, PythonHash>;

//////////////////////////////////////////////////////////////////////////

template <typename AriphmeticType>
class Uniform {
 public:
  template <typename T, typename Enable = void>
  struct Distribution;

  template <typename T>
  struct Distribution<
        T, typename std::enable_if_t<std::is_integral<T>::value>> {
    std::uniform_int_distribution<T> handle;

    Distribution()
        : handle(std::numeric_limits<T>::min(), std::numeric_limits<T>::max()) {
    }
  };

  template <typename T>
  struct Distribution<
        T, typename std::enable_if_t<std::is_floating_point<T>::value>> {
    std::uniform_real_distribution<T> handle;
  };

  Uniform(int64_t seed) : generator(seed), distribution() {}

  AriphmeticType Get() {
    return distribution.handle(generator);
  }

private:
  std::mt19937 generator;
  Distribution<AriphmeticType> distribution;
};

template <typename AriphmeticType> AriphmeticType Random() {
  thread_local int64_t seed =
      std::chrono::steady_clock::now().time_since_epoch().count();
  thread_local Uniform<AriphmeticType> instance(seed);

  return instance.Get();
}

//////////////////////////////////////////////////////////////////////////

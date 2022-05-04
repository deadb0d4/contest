#ifndef EBUG
#pragma GCC target("sse,sse2,avx2,bmi,bmi2,lzcnt,popcnt")
#pragma GCC optimize("-O3,-funroll-loops,-march=native,-mtune=native")
#endif

#include <bits/stdc++.h>

using namespace std;

//////////////////////////////////////////////////////////////////////////

inline namespace contest_io {

using namespace std;

struct _cin_t {
  _cin_t() {
    ios_base::sync_with_stdio(false);
    cin.tie(nullptr);
  }

  template <class T> operator T() {
    T x; cin >> x;
    return x;
  }
} cin;

#ifdef EBUG
#define CONTEST_MACRO_DEBUG 1
#else
#define CONTEST_MACRO_DEBUG 0
#endif

#define dout_if(on)      \
if (!on) ;               \
else cerr                \

#define dout dout_if(CONTEST_MACRO_DEBUG)

}  // inline namespace contest_io

//////////////////////////////////////////////////////////////////////////

// Sliding set approach to closest pair of points
// author: bicsi
int64_t ClosestDistance(vector<pair<int64_t, int64_t>>& points) {
  sort(begin(points), end(points));
  int64_t curr_answer = 1'000'000'000'000'000'016;
  int l = 0;
  set<pair<int64_t, int64_t>> bag;
  for (int r = 0; r < len; ++r) {
    int64_t d = ceil(sqrt(curr_answer));
    while (points[r].first - points[l].first > d) {
      bag.erase({points[l].second, points[l].first});
      ++l;
    }
    auto b = bag.lower_bound({points[r].second - d, points[r].first});
    auto e = bag.upper_bound({points[r].second + d, points[r].first});
    // this is on average linear on (e - b)
    // but in a rectangle there is only a constant number of points
    // due to `curr_answer` distance bound
    for (auto it = b; it != e; ++it) {
      int64_t dx = it->second - points[r].first;
      int64_t dy = it->first - points[r].second;
      curr_answer = min(curr_answer, dx * dx + dy * dy);
    }
    bag.insert({points[r].second, points[r].first});
  }
  return curr_answer;
}

//////////////////////////////////////////////////////////////////////////

template <class T, size_t n>
struct VecS {
  typedef vector<typename VecS<T, n - 1>::type> type;
};

template <class T>
struct VecS<T, 1> {
  typedef vector<T> type;
};

template <class T, size_t n>
using Vec = typename VecS<T, n>::type;

template <class T>
auto BuildVec(const T& val) {
  return T(val);
}

template <class T, class... SizeType>
auto BuildVec(const T& val, size_t first, SizeType... rest) {
  return Vec<T, 1 + sizeof...(rest)>(first, BuildVec<T>(val, rest...));
}

//////////////////////////////////////////////////////////////////////////

// Rabin-Karp
struct Hasher {
  int64_t a;
  vector<int64_t> hv;

  Hasher(const string &str, int64_t roll) : a(roll), hv(str.size() + 1, 0) {
    for (int i = 1; i < hv.size(); ++i) {
      hv[i] = Sum(hv[i - 1], Mul(SmPow(a, i - 1), str[i - 1]));
    }
    a = Inv(a);
  }

  int64_t Hash(int b, int e) const {
    return Mul(Can(hv[e + 1] - hv[b]), SmPow(a, b));
  }
};

// Knuth-Morris-Pratt algorithm for prefix function
auto PrefixFunction(const string &str) {
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

// if size is odd, then it is the middle element, else it is the mean value
// between middle elements.
vector<double> MedianSlidingWindow(vector<int> &nums, int k) {
  if (nums.empty()) {
    return {};
  }
  multiset<int64_t> small, big;

  for (int i = 0; i < k; ++i) {
    big.insert(nums[i]);
  }
  while (small.size() != k / 2) {
    auto it = begin(big);
    small.insert(*it);
    big.erase(it);
  }

  vector<double> vec;
  if (k % 2 == 0) {
    vec.push_back((double)*prev(end(small)) + *begin(big));
    vec.back() /= 2;
  } else {
    vec.push_back(*begin(big));
  }

  for (int i = k; i < nums.size(); ++i) {

    if (big.count(nums[i - k])) {
      big.erase(big.find(nums[i - k]));
    } else {
      small.erase(small.find(nums[i - k]));
    }

    if (*begin(big) <= nums[i]) {
      big.insert(nums[i]);
    } else {
      small.insert(nums[i]);
    }

    while (small.size() != k / 2) {
      if (small.size() < k / 2) {
        auto it = begin(big);
        small.insert(*it);
        big.erase(it);
      } else {
        auto it = prev(end(small));
        big.insert(*it);
        small.erase(it);
      }
    }

    if (k % 2 == 0) {
      vec.push_back((double)*prev(end(small)) + *begin(big));
      vec.back() /= 2;
    } else {
      vec.push_back(*begin(big));
    }
  }

  return vec;
}

//////////////////////////////////////////////////////////////////////////

// Dynamic connectivity Offline
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

  void Checkpoint() { hist.emplace_back(-1, 0, 0); }

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

struct Edge {
  int b;
  int e;
  int u;
  int v;
};

void FillAns(DSU &dsu, const vector<Edge> &ev, const vector<int> &qv, int vl,
             int vr) {
  if (vl > vr or qv.empty()) {
    return;
  }
  dsu.Checkpoint();

  int m = (vl + vr) / 2;
  vector<Edge> lev, rev;
  for (auto &&edge : ev)
    if (edge.b <= edge.e) {
      if (edge.b == vl and edge.e == vr) {
        dsu.Union(edge.u, edge.v);
      } else {
        lev.push_back({edge.b, min(edge.e, m), edge.u, edge.v});
        rev.push_back({max(edge.b, m + 1), edge.e, edge.u, edge.v});
      }
    }
  if (vl == vr) {
    cout << dsu.count << '\n';
  } else {
    vector<int> lqv, rqv;
    for (auto &&q : qv) {
      if (q <= m) {
        lqv.push_back(q);
      } else {
        rqv.push_back(q);
      }
    }
    FillAns(dsu, lev, lqv, vl, m);
    FillAns(dsu, rev, rqv, m + 1, vr);
  }
  dsu.Rollback();
}

// Suffix array
void CountSort(vector<int> &pv, const vector<int> &inv) {
  vector<int> buf(pv.size());
  for (int i = 0; i < pv.size(); ++i) {
    int j = pv[i];
    buf[inv[j]]++;
  }
  vector<int> pos(pv.size());
  pos[0] = 0;
  for (int i = 1; i < pv.size(); ++i) {
    pos[i] = pos[i - 1] + buf[i - 1];
  }
  for (auto &&p : pv) {
    buf[pos[inv[p]]] = p;
    pos[inv[p]]++;
  }
  pv = buf;
}

vector<int> SuffArr(const string &str) {
  auto get_sym = [&str](int pi, int j) {
    int pos = (pi + j) % (str.size() + 1);
    if (pos == str.size()) {
      return '$';
    }
    return str[pos];
  };
  auto half_pi = [size = str.size()](int pi, int k) {
    return (pi + (1 << (k - 1))) % (size + 1);
  };

  // k = 0
  vector<int> pv;
  for (int i = 0; i <= str.size(); ++i) {
    pv.push_back(i);
  }
  sort(begin(pv), end(pv),
       [&get_sym](int i, int j) { return get_sym(i, 0) < get_sym(j, 0); });
  vector<int> inv(pv.size());
  inv[pv[0]] = 0;
  for (int i = 1; i < pv.size(); ++i) {
    inv[pv[i]] = inv[pv[i - 1]];
    if (get_sym(pv[i - 1], 0) != get_sym(pv[i], 0)) {
      inv[pv[i]]++;
    }
  }

  for (int k = 0; (1 << k++) <= str.size();) {
    for (int i = 0; i < pv.size(); ++i) {
      pv[i] = (str.size() + 1 + pv[i] - (1 << (k - 1))) % (str.size() + 1);
    }
    CountSort(pv, inv);
    auto pre = inv;
    for (int i = 1; i < pv.size(); ++i) {
      if (pre[pv[i - 1]] == pre[pv[i]] and
          pre[half_pi(pv[i - 1], k)] == pre[half_pi(pv[i], k)]) {
        inv[pv[i]] = inv[pv[i - 1]];
      } else {
        inv[pv[i]] = inv[pv[i - 1]] + 1;
      }
    }
  }
  return pv;
}

// Segtree with minimum and its first index
struct Segtree {
  vector<pair<int, int>> t;

  Segtree(int len) : t(4 * len) { BuildTree(1, 0, len - 1); }

  void BuildTree(int v, int vl, int vr) {
    if (vl == vr) {
      t[v] = {0, vl};
      return;
    }

    int m = (vl + vr) / 2;
    BuildTree(2 * v, vl, m);
    BuildTree(2 * v + 1, m + 1, vr);

    t[v] = min(t[2 * v], t[2 * v + 1]);
  }

  void Set(int pos, int val, int v = 1, int vl = 0, int vr = -1) {
    if (vr == -1)
      vr = t.size() / 4 - 1;

    if (vl == vr) {
      t[v] = {val, vl};
      return;
    }
    int m = (vl + vr) / 2;
    if (pos > m) {
      Set(pos, val, 2 * v + 1, m + 1, vr);
    } else {
      Set(pos, val, 2 * v, vl, m);
    }
    t[v] = min(t[2 * v], t[2 * v + 1]);
  }

  auto Query(int l, int r, int v = 1, int vl = 0, int vr = -1) {
    if (vr == -1)
      vr = t.size() / 4 - 1;

    if (l == vl and r == vr) {
      return t[v];
    }
    int m = (vl + vr) / 2;
    if (r <= m) {
      return Query(l, r, 2 * v, vl, m);
    } else if (m < l) {
      return Query(l, r, 2 * v + 1, m + 1, vr);
    }
    return min(Query(l, m, 2 * v, vl, m),
               Query(m + 1, r, 2 * v + 1, m + 1, vr));
  }
};

// Manacher algorithm
// "abba" -> d2[2] = 2;
// "aba" -> d1[1] = 2;
auto ManacherPalindromeCount(const string &s) {
  vector<int> d1(s.size()), d2(s.size());

  int l = 0, r = -1;
  for (int i = 0; i < s.size(); ++i) {
    d1[i] = 1;

    if (i <= r) {
      d1[i] = min(r - i + 1, d1[l + r - i]);
    }
    while (i - d1[i] >= 0 and i + d1[i] < s.size() and
           s[i - d1[i]] == s[i + d1[i]]) {
      ++d1[i];
    }
    if (r < i + d1[i] - 1) {
      l = i - d1[i] + 1;
      r = i + d1[i] - 1;
    }
  }
  l = 0;
  r = -1;
  for (int i = 1; i < s.size(); ++i) {
    d2[i] = 0;
    if (i <= r) {
      d2[i] = min(r - i + 1, d2[l + r - i + 1]);
    }
    while (i - d2[i] >= 1 and i + d2[i] < s.size() and
           s[i - d2[i] - 1] == s[i + d2[i]]) {
      ++d2[i];
    }
    if (r < i + d2[i] - 1) {
      l = i - d2[i];
      r = i + d2[i] - 1;
    }
  }
  return make_pair(d1, d2);
}

// Implicit segment tree: maximum and a setter
struct Node {
  int64_t mx;
  int64_t vl;
  int64_t vr;
  Node *lnode;
  Node *rnode;

  Node(int64_t l, int64_t r)
      : mx(0), vl(l), vr(r), lnode(nullptr), rnode(nullptr) {}

  ~Node() {
    delete lnode;
    delete rnode;
  }

  void InitNode() {
    if (vl != vr) {
      int64_t m = vl + (vr - vl) / 2;
      if (lnode == nullptr) {
        lnode = new Node(vl, m);
      }
      if (rnode == nullptr) {
        rnode = new Node(m + 1, vr);
      }
    }
  }

  int64_t Query(int64_t l, int64_t r) {
    if (l == vl and r == vr) {
      return mx;
    }
    InitNode();
    int64_t m = vl + (vr - vl) / 2;
    if (m < l) {
      return rnode->Query(l, r);
    } else if (r <= m) {
      return lnode->Query(l, r);
    }
    return max(lnode->Query(l, m), rnode->Query(m + 1, r));
  }

  void Set(int64_t pos, int64_t val) {
    if (vl == vr) {
      mx = val;
      return;
    }
    InitNode();
    int64_t m = vl + (vr - vl) / 2;
    if (m < pos) {
      rnode->Set(pos, val);
    } else {
      lnode->Set(pos, val);
    }
    mx = max(lnode->mx, rnode->mx);
  }
};

// Reading a tuple
template <class Head, class... Tail>
std::tuple<Head, Tail...> ReadTuple(std::istream &is = cin) {
  Head val;
  is >> val;
  if constexpr (sizeof...(Tail) == 0) {
    return std::tuple{val};
  } else {
    return std::tuple_cat(std::tuple{val}, ReadTuple<Tail...>(is));
  }
}
#define _r(...) ReadTuple<__VA_ARGS__>()

// Printing a tuple
template <std::size_t...> struct seq {};

template <std::size_t N, std::size_t... Is>
struct gen_seq : gen_seq<N - 1, N - 1, Is...> {};

template <std::size_t... Is> struct gen_seq<0, Is...> : seq<Is...> {};

template <class Ch, class Tr, class Tuple, std::size_t... Is>
void print_tuple(std::basic_ostream<Ch, Tr> &os, Tuple const &t, seq<Is...>) {
  using swallow = int[];
  (void)swallow{0, (void(os << (Is == 0 ? "" : ", ") << std::get<Is>(t)), 0)...};
}

template <class Ch, class Tr, class... Args>
std::basic_ostream<Ch, Tr>& operator << (
    std::basic_ostream<Ch, Tr> &os, std::tuple<Args...> const &t) {
  os << '(';
  print_tuple(os, t, gen_seq<sizeof...(Args)>());
  return os << ')';
}

template <class Ch, class Tr, class F, class S>
std::basic_ostream<Ch, Tr> &operator << (
    std::basic_ostream<Ch, Tr> &os, const std::pair<F, S> &p) {
  return os << '(' << p.first << ', ' << p.second << ')';
}

template <typename AriphmeticType>
class Uniform {
 public:
  template <typename T, typename Enable = void>
  struct Distribution;

  template <typename T>
  struct Distribution<T, enable_if_t<is_integral_v<T>>> {
    std::uniform_int_distribution<T> handle;
    Distribution()
        : handle(numeric_limits<T>::min(), numeric_limits<T>::max()) {}
  };

  template <typename T>
  struct Distribution<T, enable_if_t<is_floating_point_v<T>>> {
    std::uniform_real_distribution<T> handle;
  };

  Uniform(int64_t seed) : gen(seed), dist() {}

  AriphmeticType Get() {
    return dist.handle(gen);
  }

 private:
  std::mt19937 gen;
  Distribution<AriphmeticType> dist;
};

template <typename AriphmeticType>
AriphmeticType Random() {
  thread_local int64_t seed =
      chrono::steady_clock::now().time_since_epoch().count();
  thread_local Uniform<AriphmeticType> instance(seed);
  return instance.Get();
}

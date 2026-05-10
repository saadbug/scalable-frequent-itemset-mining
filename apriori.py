import time
import tracemalloc
from itertools import combinations

class Apriori:
    """Classical Apriori — Agrawal & Srikant (1994)."""

    def __init__(self, min_sup_ratio, **kwargs):
        self.min_sup_ratio   = min_sup_ratio
        self.min_sup_count   = 0
        self.frequent_itemsets           = {}
        self.total_candidates_generated  = 0
        self.execution_time  = 0.0
        self.peak_memory_mb  = 0.0

    # ── helpers ──────────────────────────────────────────────

    @staticmethod
    def _key(item):
        try:    return (0, int(item))
        except: return (1, str(item))

    def _gen_candidates(self, freq_list, k):
        freq_set = set(freq_list)
        candidates = set()
        
        # 1. Sort the items inside each itemset
        sorted_sets = [sorted(fs, key=self._key) for fs in freq_list]
        
        # 2. CRITICAL BUG FIX: Sort the master list lexicographically!
        # This guarantees L1 will always be mathematically "less than" L2, preventing skipped pairs.
        sorted_sets.sort(key=lambda x: [self._key(item) for item in x])
        
        n = len(sorted_sets)
        for i in range(n):
            for j in range(i+1, n):
                L1, L2 = sorted_sets[i], sorted_sets[j]
                if (L1[:k-2] == L2[:k-2] and self._key(L1[k-2]) < self._key(L2[k-2])):
                    cand = frozenset(L1) | frozenset(L2)
                    if all(frozenset(s) in freq_set for s in combinations(cand, k-1)):
                        candidates.add(cand)
        return candidates

    def _count_support(self, candidates, transactions):
        counts = {c: 0 for c in candidates}
        for t in transactions:
            for c in candidates:
                if c.issubset(t):
                    counts[c] += 1
        return {c: cnt for c, cnt in counts.items()
                if cnt >= self.min_sup_count}

    # ── main ─────────────────────────────────────────────────

    def fit(self, transactions):
        tracemalloc.start()
        t0 = time.perf_counter()

        n = len(transactions)
        self.min_sup_count = self.min_sup_ratio * n

        # k=1
        item_cnt = {}
        for t in transactions:
            for item in t:
                item_cnt[item] = item_cnt.get(item, 0) + 1
        current = {frozenset([it]): cnt
                   for it, cnt in item_cnt.items()
                   if cnt >= self.min_sup_count}

        k = 1
        while current:
            self.frequent_itemsets.update(current)
            k += 1
            cands = self._gen_candidates(list(current.keys()), k)
            self.total_candidates_generated += len(cands)
            if not cands:
                break
            current = self._count_support(cands, transactions)

        self.execution_time = time.perf_counter() - t0
        _, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        self.peak_memory_mb = peak / 1024 / 1024
        return self.frequent_itemsets

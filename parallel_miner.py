import time
import tracemalloc
import multiprocessing as mp
from collections import defaultdict

# ══════════════════════════════════════════════════════════════
#  ALGORITHM 2: PARALLEL VERTICAL MINER
#  Single-node approximation of SHFIM (Al-Bana et al., 2022)
#  Core innovations: vertical TID-lists + multicore partitioning
# ══════════════════════════════════════════════════════════════
def _mine_branch(args):
    root_itemset, root_tids, peers, min_sup_count = args
    result = {root_itemset: len(root_tids)}

    stack = [(root_itemset, root_tids, peers)]
    while stack:
        cur_itemset, cur_tids, cur_peers = stack.pop()
        
        for i in range(len(cur_peers)):
            peer_itemset, peer_tids = cur_peers[i]
            common_tids = cur_tids & peer_tids
            
            if len(common_tids) >= min_sup_count:
                new_itemset = cur_itemset | peer_itemset
                result[new_itemset] = len(common_tids)
                # Pass the remaining peers forward to branch deeper
                stack.append((new_itemset, common_tids, cur_peers[i+1:]))

    return result

class ParallelVerticalMiner:
    """
    Single-node implementation of SHFIM's core algorithmic engine.

    SHFIM (Al-Bana, Farhan, Othman — MDPI Data, 2022) achieves speed via:
      (1) Vertical TID-list format → eliminates repeated database scans
      (2) Diffset / partition-based distribution across compute nodes

    This implementation reproduces (1) exactly and approximates (2)
    using Python multiprocessing across CPU cores — isolating the
    algorithmic gains from distributed infrastructure.

    Reference:
        Al-Bana, M.R., Farhan, M.S., Othman, N.A. (2022).
        "An Efficient Spark-Based Hybrid Frequent Itemset Mining
        Algorithm for Big Data." MDPI Data, 7(1), 11.
        https://doi.org/10.3390/data7010011
    """

    def __init__(self, min_sup_ratio, n_workers=None, **kwargs):
        self.min_sup_ratio   = min_sup_ratio
        self.n_workers       = n_workers or max(1, mp.cpu_count() - 1)
        self.min_sup_count   = 0
        self.frequent_itemsets   = {}
        self.total_intersections = 0
        self.execution_time  = 0.0
        self.peak_memory_mb  = 0.0

    def fit(self, transactions):
        tracemalloc.start()
        t0 = time.perf_counter()

        n = len(transactions)
        self.min_sup_count = self.min_sup_ratio * n

        # ── Phase 1: Build vertical database ─────────────────
        vertical_db = defaultdict(set)
        for tid, transaction in enumerate(transactions):
            for item in transaction:
                vertical_db[item].add(tid)

        # ── Phase 2: Filter frequent 1-itemsets ──────────────
        freq1 = {
            frozenset([item]): tids
            for item, tids in vertical_db.items()
            if len(tids) >= self.min_sup_count
        }
        del vertical_db

        # Sort descending by support (most frequent first — better pruning)
        items = sorted(freq1.keys(), key=lambda x: -len(freq1[x]))

        # ── Phase 3: Build top-level tasks for parallel dispatch
        tasks = []
        for i, itemset_a in enumerate(items):
            tids_a = freq1[itemset_a]
            peers  = []
            for j in range(i+1, len(items)):
                itemset_b = items[j]
                common    = tids_a & freq1[itemset_b]
                self.total_intersections += 1
                if len(common) >= self.min_sup_count:
                    peers.append((itemset_a | itemset_b, common))
            tasks.append((itemset_a, tids_a, peers, self.min_sup_count))

        # ── Phase 4: Parallel DFS across CPU cores ───────────
        if self.n_workers > 1 and len(tasks) > 1:
            with mp.Pool(processes=self.n_workers) as pool:
                results = pool.map(_mine_branch, tasks)
        else:
            results = [_mine_branch(t) for t in tasks]

        for r in results:
            self.frequent_itemsets.update(r)

        self.execution_time = time.perf_counter() - t0
        _, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        self.peak_memory_mb = peak / 1024 / 1024
        return self.frequent_itemsets


# ══════════════════════════════════════════════════════════════
#  BENCHMARK RUNNER
# ══════════════════════════════════════════════════════════════
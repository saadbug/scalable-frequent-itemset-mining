"""
benchmark.py — Final Benchmark Runner
CS-378 Semester Project | Apriori vs Parallel Vertical Miner (SHFIM-inspired)

• Self-contained: no imports from other project files needed
• Runs 3 times per config, averages results
• Saves results.csv for your report tables

Usage:
    python benchmark.py

Place this file in the SAME folder as your .gz datasets.
Rename the dataset filenames in DATASETS config below if yours differ.
"""

import os
import csv
import gc
import gzip
import multiprocessing as mp
from apriori import Apriori
from parallel_miner import ParallelVerticalMiner


# ══════════════════════════════════════════════════════════════
#  CONFIG — edit dataset filenames here if needed
# ══════════════════════════════════════════════════════════════

DATASETS = [
    # (display_name,  filename,          thresholds,          apriori_row_cap)
    ("Chess",    "chess.dat.gz",    [0.90, 0.85, 0.80],  None),  
    ("Connect",  "connect.dat.gz",  [0.98, 0.96, 0.95],  20000), 
    ("Accident", "accidents.dat.gz",[0.90, 0.85, 0.80],  10000), 
]

NUM_RUNS = 3   # average over this many runs (project requires ≥3)


# ══════════════════════════════════════════════════════════════
#  DATASET LOADER
# ══════════════════════════════════════════════════════════════

def load_dataset(filepath, max_rows=None):
    """Load a .dat.gz or .dat file into a list of frozensets."""
    transactions = []
    opener = gzip.open if filepath.endswith('.gz') else open
    with opener(filepath, 'rt', encoding='utf-8', errors='ignore') as f:
        for i, line in enumerate(f):
            if max_rows and i >= max_rows:
                break
            items = line.strip().split()
            if items:
                transactions.append(frozenset(items))
    return transactions


# ══════════════════════════════════════════════════════════════
#  ALGORITHM 1: APRIORI (baseline)
# ══════════════════════════════════════════════════════════════





def run_once(AlgClass, dataset, min_sup, kwargs):
    alg = AlgClass(min_sup, **kwargs)
    alg.fit(dataset)
    return {
        "time_s":        alg.execution_time,
        "memory_mb":     alg.peak_memory_mb,
        "freq_itemsets": len(alg.frequent_itemsets),
        "candidates":    getattr(alg, "total_candidates_generated", "N/A"),
    }

def average(metrics):
    out = {}
    for key in metrics[0]:
        vals = [m[key] for m in metrics if isinstance(m[key], (int, float))]
        out[key] = round(sum(vals)/len(vals), 4) if vals else metrics[0][key]
    return out


def main():
    rows = []
    baseline_times = {}

    n_cpu = max(1, mp.cpu_count() - 1)
    print(f"System: {mp.cpu_count()} CPU cores | using {n_cpu} workers for PVM")

    for ds_name, ds_file, thresholds, apriori_cap in DATASETS:

        if not os.path.exists(ds_file):
            print(f"\n⚠  SKIPPING '{ds_name}': file '{ds_file}' not found.")
            continue

        print(f"\n{'═'*65}")
        print(f"  {ds_name.upper()}  ({ds_file})")
        print(f"{'═'*65}")

        # LOAD DATA
        full_data = load_dataset(ds_file)
        # CRITICAL FIX: Both algorithms MUST use the exact same data subset for a fair comparison
        test_data = load_dataset(ds_file, apriori_cap) if apriori_cap else full_data

        print(f"  Total rows available: {len(full_data):,} transactions")
        if apriori_cap:
            print(f"  *** BENCHMARKING ON CAPPED SUBSET: {len(test_data):,} transactions ***")

        # BOTH algorithms now use `test_data`
        algorithms = [
            ("Apriori", Apriori, test_data, {}),
            ("Parallel-Vertical-Miner", ParallelVerticalMiner, test_data, {"n_workers": n_cpu}),
        ]

        for threshold in thresholds:
            print(f"\n  ── min_sup = {threshold:.2f} ({threshold*100:.0f}%) ──")

            for alg_name, AlgClass, dataset, kwargs in algorithms:
                print(f"    {alg_name:<28} ", end="", flush=True)
                run_metrics = []

                for r in range(NUM_RUNS):
                    try:
                        m = run_once(AlgClass, dataset, threshold, kwargs)
                        run_metrics.append(m)
                        print(f"{m['time_s']:.2f}s  ", end="", flush=True)
                    except MemoryError:
                        print(f"[OOM]  ", end="")
                        break
                    except Exception as e:
                        print(f"[ERR:{e}]  ", end="")
                        break
                    finally:
                        gc.collect()

                print()
                if not run_metrics:
                    continue

                a   = average(run_metrics)
                key = (ds_name, threshold)

                if alg_name == "Apriori":
                    baseline_times[key] = a["time_s"]

                base_t  = baseline_times.get(key, a["time_s"])
                speedup = round(base_t / a["time_s"], 2) if a["time_s"] > 0 else "∞"

                print(f"      → {a['time_s']}s | {a['memory_mb']} MB | "
                      f"{int(a['freq_itemsets'])} freq. itemsets | "
                      f"speedup vs Apriori: {speedup}×")

                rows.append({
                    "Dataset":       ds_name,
                    "Algorithm":     alg_name,
                    "min_sup":       threshold,
                    "Transactions":  len(dataset),
                    "Avg_Time_s":    a["time_s"],
                    "Avg_Memory_MB": a["memory_mb"],
                    "Freq_Itemsets": int(a["freq_itemsets"]),
                    "Candidates":    a["candidates"],
                    "Speedup":       speedup,
                })

        del full_data, test_data
        gc.collect()

    # ── Save CSV ─────────────────────────────────────────────
    if rows:
        with open("results.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
        print(f"\n✅  results.csv saved ({len(rows)} rows)")

    # ── Print summary table ───────────────────────────────────
    print("\n" + "═"*95)
    print(f"{'Dataset':<12} {'Algorithm':<28} {'sup':>5} "
          f"{'Time(s)':>9} {'Mem(MB)':>9} {'#Freq':>8} {'Speedup':>9}")
    print("─"*95)
    for r in rows:
        print(f"{r['Dataset']:<12} {r['Algorithm']:<28} {r['min_sup']:>5.2f} "
              f"{r['Avg_Time_s']:>9.4f} {r['Avg_Memory_MB']:>9.2f} "
              f"{r['Freq_Itemsets']:>8} {str(r['Speedup']):>9}")
    print("═"*95)

if __name__ == "__main__":
    mp.freeze_support()  # required on Windows
    main()
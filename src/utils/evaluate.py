"""
evaluate.py - Standalone mAP Evaluator (khong can chay lai pipeline)
"""

import sys
import pickle
import numpy as np
import argparse
from pathlib import Path

try:
    sys.stdout.reconfigure(encoding="utf-8")
except AttributeError:
    pass

BASE_DIR   = Path(__file__).resolve().parent.parent
DATA_DIR   = BASE_DIR / "data" / "datasets"
OUTPUT_DIR = BASE_DIR / "output" / "stage3"


def compute_ap(ranks, nres):
    nimgranks = len(ranks)
    ap = 0
    recall_step = 1.0 / nres
    for j in range(nimgranks):
        rank = ranks[j]
        precision_0 = 1.0 if rank == 0 else float(j) / rank
        precision_1 = float(j + 1) / (rank + 1)
        ap += (precision_0 + precision_1) * recall_step / 2.0
    return ap


def compute_map(ranks, gnd, kappas=[1, 5, 10]):
    map_val = 0.0
    nq = len(gnd)
    aps = np.zeros(nq)
    pr  = np.zeros(len(kappas))
    prs = np.zeros((nq, len(kappas)))
    nempty = 0

    for i in range(nq):
        qgnd = np.array(gnd[i]["ok"])
        if qgnd.shape[0] == 0:
            aps[i] = float("nan")
            prs[i, :] = float("nan")
            nempty += 1
            continue
        try:
            qgndj = np.array(gnd[i]["junk"])
        except KeyError:
            qgndj = np.empty(0)

        pos  = np.where(np.isin(ranks[:, i], qgnd))[0]
        junk = np.where(np.isin(ranks[:, i], qgndj))[0]

        k, ij = 0, 0
        if len(junk):
            ip = 0
            while ip < len(pos):
                while ij < len(junk) and pos[ip] > junk[ij]:
                    k  += 1
                    ij += 1
                pos[ip] -= k
                ip += 1

        ap = compute_ap(pos, len(qgnd))
        map_val += ap
        aps[i] = ap

        pos_1 = pos + 1
        for j, kappa in enumerate(kappas):
            if len(pos_1) > 0:
                kq = min(max(pos_1), kappa)
                prs[i, j] = (pos_1 <= kq).sum() / kq
            else:
                prs[i, j] = 0.0
        pr += prs[i, :]

    map_val /= (nq - nempty)
    pr      /= (nq - nempty)
    return map_val, aps, pr, prs


def evaluate_protocols(ranks, gnd, kappas=[1, 5, 10]):
    results = {}

    gnd_e = [{"ok": g["easy"],
               "junk": np.concatenate([g["junk"], g["hard"]])} for g in gnd]
    mapE, apsE, mprE, prsE = compute_map(ranks, gnd_e, kappas)
    results["easy"] = {"map": mapE, "aps": apsE, "mpr": mprE, "prs": prsE}

    gnd_m = [{"ok": np.concatenate([g["easy"], g["hard"]]),
               "junk": g["junk"]} for g in gnd]
    mapM, apsM, mprM, prsM = compute_map(ranks, gnd_m, kappas)
    results["medium"] = {"map": mapM, "aps": apsM, "mpr": mprM, "prs": prsM}

    gnd_h = [{"ok": g["hard"],
               "junk": np.concatenate([g["junk"], g["easy"]])} for g in gnd]
    mapH, apsH, mprH, prsH = compute_map(ranks, gnd_h, kappas)
    results["hard"] = {"map": mapH, "aps": apsH, "mpr": mprH, "prs": prsH}

    return results


def print_results(dataset, results, qimlist, kappas, show_per_query=False):
    SEP = "=" * 68
    sep = "-" * 68

    print(f"\n{SEP}")
    print(f"  EVALUATION RESULTS: {dataset.upper()}")
    print(SEP)

    print(f"\n  {'Protocol':<10} {'mAP':>8}  {'P@1':>8}  {'P@5':>8}  {'P@10':>8}")
    print(f"  {sep}")

    for proto, label in [("easy","Easy  "), ("medium","Medium"), ("hard","Hard  ")]:
        r   = results[proto]
        mAP = r["map"] * 100
        p1  = r["mpr"][0] * 100 if len(r["mpr"]) > 0 else 0
        p5  = r["mpr"][1] * 100 if len(r["mpr"]) > 1 else 0
        p10 = r["mpr"][2] * 100 if len(r["mpr"]) > 2 else 0
        print(f"  {label:<10} {mAP:>7.2f}%  {p1:>7.2f}%  {p5:>7.2f}%  {p10:>7.2f}%")

    print(f"  {sep}")

    med_map = results["medium"]["map"] * 100
    target  = 81.69 if "oxf" in dataset else 88.52
    diff    = med_map - target
    sign    = "+" if diff >= 0 else ""
    print(f"\n  >> Medium mAP (metric chinh): {med_map:.2f}%")
    print(f"     Paper target:               {target:.2f}%")
    print(f"     So sanh paper:              {sign}{diff:.2f}%")

    if show_per_query:
        print(f"\n  {SEP}")
        print(f"  PER-QUERY AP (Medium Protocol) - Sort: thap -> cao")
        print(f"  {sep}")
        print(f"  {'#':<5} {'Query Name':<35} {'AP Easy':>9} {'AP Med':>9} {'AP Hard':>9}")
        print(f"  {sep}")

        aps_e = results["easy"]["aps"]
        aps_m = results["medium"]["aps"]
        aps_h = results["hard"]["aps"]
        order = np.argsort(aps_m)

        for rank_q, q_idx in enumerate(order):
            q_name = qimlist[q_idx] if q_idx < len(qimlist) else f"query_{q_idx}"
            ap_e = aps_e[q_idx] * 100 if not np.isnan(aps_e[q_idx]) else float("nan")
            ap_m = aps_m[q_idx] * 100 if not np.isnan(aps_m[q_idx]) else float("nan")
            ap_h = aps_h[q_idx] * 100 if not np.isnan(aps_h[q_idx]) else float("nan")
            marker = " <<< KHO" if ap_m < 50 else ""
            try:
                print(f"  {rank_q+1:<5} {q_name:<35} {ap_e:>8.1f}% {ap_m:>8.1f}% {ap_h:>8.1f}%{marker}")
            except Exception:
                print(f"  {rank_q+1} {q_name}")

        print(f"  {sep}")
        valid_aps = aps_m[~np.isnan(aps_m)]
        print(f"\n  Statistics (Medium):")
        print(f"    Min AP   : {valid_aps.min()*100:.2f}%")
        print(f"    Max AP   : {valid_aps.max()*100:.2f}%")
        print(f"    Median AP: {np.median(valid_aps)*100:.2f}%")
        print(f"    Std AP   : {valid_aps.std()*100:.2f}%")
        print(f"    Queries AP < 50%: {(valid_aps < 0.5).sum()}/{len(valid_aps)}")

    print(f"\n{SEP}\n")


def load_ranks_npy(dataset):
    npy_path = OUTPUT_DIR / f"{dataset}_ranks.npy"
    if npy_path.exists():
        print(f"  [OK] Loading ranks cache: {npy_path.name}")
        return np.load(str(npy_path))
    return None


def evaluate_dataset(dataset, kappas=[1, 5, 10], show_per_query=False):
    print(f"\nLoading: {dataset} ...")

    gnd_path = DATA_DIR / dataset / f"gnd_{dataset}.pkl"
    if not gnd_path.exists():
        print(f"  [ERROR] Ground truth not found: {gnd_path}")
        return None

    with open(gnd_path, "rb") as f:
        gnd_data = pickle.load(f)

    imlist  = gnd_data["imlist"]
    qimlist = gnd_data["qimlist"]
    gnd     = gnd_data["gnd"]
    db_size = len(imlist)
    nq      = len(qimlist)
    print(f"  DB size: {db_size}, Queries: {nq}")

    ranks = load_ranks_npy(dataset)

    if ranks is None:
        print(f"  [WARN] Khong co file ranks cache ({dataset}_ranks.npy).")
        print(f"         Chay stage3_rerank.py de tao file cache va co ket qua chinh xac.")
        print(f"         Hien tai se doc mAP tu file result txt da luu...")
        result_txt = OUTPUT_DIR / f"{dataset}_final_results.txt"
        if result_txt.exists():
            lines = result_txt.read_text(encoding="utf-8").splitlines()
            for line in lines[:10]:
                if "mAP" in line or "Easy" in line or "Medium" in line or "Hard" in line:
                    print(f"    {line.strip()}")
        return None

    if ranks.shape[0] != db_size or ranks.shape[1] != nq:
        print(f"  [WARN] ranks shape {ranks.shape} vs expected ({db_size},{nq})")

    print(f"  Computing mAP...")
    results = evaluate_protocols(ranks, gnd, kappas)
    print_results(dataset, results, qimlist, kappas, show_per_query)
    return results


def main():
    parser = argparse.ArgumentParser(
        description="Standalone mAP evaluator - L2G Image Retrieval"
    )
    parser.add_argument("--dataset", type=str, default=None,
                        choices=["roxford5k", "rparis6k"])
    parser.add_argument("--per-query", action="store_true")
    parser.add_argument("--kappas", type=int, nargs="+", default=[1, 5, 10])
    args = parser.parse_args()

    datasets = [args.dataset] if args.dataset else ["roxford5k", "rparis6k"]

    print("=" * 68)
    print("  L2G STANDALONE EVALUATOR")
    print("=" * 68)

    all_results = {}
    for ds in datasets:
        r = evaluate_dataset(ds, kappas=args.kappas, show_per_query=args.per_query)
        if r:
            all_results[ds] = r

    if len(all_results) == 2:
        print("=" * 68)
        print("  SUMMARY")
        print("=" * 68)
        print(f"\n  {'Dataset':<15} {'Easy':>8}  {'Medium':>8}  {'Hard':>8}")
        print(f"  {'-'*48}")
        for ds, r in all_results.items():
            label = "Oxford" if "oxf" in ds else "Paris"
            print(f"  {label:<15} "
                  f"{r['easy']['map']*100:>7.2f}%  "
                  f"{r['medium']['map']*100:>7.2f}%  "
                  f"{r['hard']['map']*100:>7.2f}%")
        print()


if __name__ == "__main__":
    main()
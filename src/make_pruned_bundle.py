#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
make_pruned_bundle.py
=====================
Copy a GLASS bundle and drop specified generator subspaces (e.g. human-adjacent
ones flagged by build_subspace_ablation.py). No re-encoding: subspaces are
independent, so scoring simply takes min over the remaining S_A_*.npz. S_H and the
whitener are unchanged, so frontier detection is unaffected by construction.

    python3 make_pruned_bundle.py \
        --src .cache/glass_production/v3_diverse \
        --dst .cache/glass_production/v3_diverse_nocohere \
        --exclude cohere,cohere-chat
"""
from __future__ import annotations
import argparse, json, shutil
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--dst", required=True)
    ap.add_argument("--exclude", default="", help="comma-separated generator names to drop")
    ap.add_argument("--prune-cos-above", type=float, default=None,
                    help="also drop any subspace whose mean principal-angle cosine to S_H exceeds this (e.g. 0.15)")
    ap.add_argument("--K", type=int, default=10)
    args = ap.parse_args()
    excl = [g.strip() for g in args.exclude.split(",") if g.strip()]

    src, dst = Path(args.src), Path(args.dst)
    if dst.exists(): shutil.rmtree(dst)
    shutil.copytree(src, dst)

    subs = dst / "subspaces"
    # geometric hygiene: flag subspaces too close to the human subspace
    if args.prune_cos_above is not None:
        import numpy as np
        Bh = np.load(dst / "S_H_universal.npz")["basis"][:, :args.K]
        print(f"  cos(S_A,S_H) per subspace (threshold {args.prune_cos_above}):")
        for p in sorted(subs.glob("S_A_*.npz")):
            Ba = np.load(p)["basis"][:, :args.K]
            c = float(np.mean(np.linalg.svd(Ba.T @ Bh, compute_uv=False)))
            tag = "  <-- prune" if c > args.prune_cos_above else ""
            print(f"    {p.stem[4:]:40s} {c:.3f}{tag}")
            if c > args.prune_cos_above and p.stem[4:] not in excl:
                excl.append(p.stem[4:])

    removed = []
    for p in list(subs.glob("S_A_*.npz")):
        if p.stem[4:] in excl:
            p.unlink(); removed.append(p.stem[4:])
    missing = [g for g in excl if g not in removed]
    kept = sorted(q.stem[4:] for q in subs.glob("S_A_*.npz"))

    # update meta.json if present
    meta = dst / "meta.json"
    if meta.exists():
        try:
            m = json.loads(meta.read_text())
            lib = m.get("S_A_library", {})
            if "generators" in lib:
                lib["generators"] = [g for g in lib["generators"] if g not in removed]
            m.setdefault("pruned", {})["excluded"] = removed
            meta.write_text(json.dumps(m, indent=2))
        except Exception:
            pass

    print(f"[pruned] {dst}")
    print(f"  removed : {removed}")
    if missing: print(f"  NOT FOUND (check names): {missing}")
    print(f"  remaining subspaces: {len(kept)}")


if __name__ == "__main__":
    main()

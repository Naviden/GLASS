#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Check length confounds, corpus composition and paired significance in evaluation results."""
from __future__ import annotations
import glob, json, sys
from pathlib import Path
import numpy as np
import pandas as pd
from rich.console import Console
from rich.table import Table
from scipy.stats import wilcoxon

console = Console()
VERDICT = {}


def auc(s, y, min_n=20):
    s = np.asarray(s, float); y = np.asarray(y)
    ok = np.isfinite(s); s, y = s[ok], y[ok]
    npos, nneg = int((y == 1).sum()), int((y == 0).sum())
    if npos < min_n or nneg < min_n:
        return np.nan
    _, inv, cnt = np.unique(s, return_inverse=True, return_counts=True)
    csum = np.cumsum(cnt); r = ((csum - cnt + csum + 1) / 2.0)[inv]
    return float((r[y == 1].sum() - npos * (npos + 1) / 2) / (npos * nneg))


def sym(a):
    """AUC of the better length direction, plus which direction."""
    if not np.isfinite(a):
        return np.nan, "-"
    return (a, "longer=AI") if a >= 0.5 else (1 - a, "shorter=AI")


def say(name, verdict, why):
    VERDICT[name] = (verdict, why)
    col = {"CLEAN": "green", "CAVEAT": "yellow", "BROKEN": "red"}[verdict]
    console.print(f"[{col}]VERDICT[{name}] = {verdict}[/]  -- {why}\n")


# =========================================================================== 1
def audit_logo_length():
    """THE FLAGSHIP NUMBER. LOGO detection = RAID AI vs RAID humans, AUC 0.9988.
    Is RAID's AI text systematically longer/shorter than its human text? If so the
    headline is a length artifact, exactly like Pile."""
    console.rule("[bold]1. LOGO / RAID -- length-label confound on the FLAGSHIP number")
    try:
        H = pd.read_parquet("results_v2/raid_en_humans.parquet")
    except Exception as e:
        console.print(f"[red]cannot read raid_en_humans.parquet: {e}[/]"); return
    hcol = "text" if "text" in H.columns else H.columns[-1]
    hum = [(str(t), d) for t, d in zip(H[hcol], H.get("domain", ["?"] * len(H)))]

    ai = []
    files = sorted(glob.glob("results_v2/openrouter_en_s*/per_doc.jsonl"))
    if not files:
        console.print("[red]no panel jsonl found[/]"); return
    keys_seen = None
    for f in files:
        for line in open(f, encoding="utf-8", errors="replace"):
            line = line.strip()
            if not line:
                continue
            try:    r = json.loads(line)
            except Exception: continue
            if keys_seen is None:
                keys_seen = list(r.keys())
                console.print(f"[dim]panel keys: {keys_seen}[/]")
            t = next((r[k] for k in ("text", "completion", "output", "generation", "response")
                      if k in r and isinstance(r[k], str)), None)
            if t:
                ai.append((t, r.get("domain", "?")))
    if not ai:
        console.print("[red]could not find a text field in the panels[/]"); return
    console.print(f"loaded {len(ai)} AI docs, {len(hum)} human docs")

    wa = np.array([len(t.split()) for t, _ in ai], float)
    wh = np.array([len(t.split()) for t, _ in hum], float)
    # apply the SAME min-40-word fitting filter used to build the bundle
    for mw in (0, 40, 100):
        a, h = wa[wa >= mw], wh[wh >= mw]
        s = np.concatenate([a, h]); y = np.array([1] * len(a) + [0] * len(h))
        val, d = sym(auc(s, y))
        console.print(f"  [word-count|RAID >= {mw:3d}w]  AUC = {val:.4f} ({d})   "
                      f"median AI={np.median(a):.0f}w  median human={np.median(h):.0f}w  "
                      f"n={len(a)}/{len(h)}")
        print(f"RESULT[wordcount RAID minw={mw}] auc={val:.4f}")

    # per domain, at the fitting threshold actually used (40 words)
    doms = sorted({d for _, d in ai} & {d for _, d in hum})
    t = Table(title="RAID per-domain length-only AUC (>=40 words, the fitting filter)")
    for c in ["domain", "word-count AUC", "direction", "median AI", "median human", "n AI/human"]:
        t.add_column(c, justify="right")
    worst = 0.0
    for dm in doms:
        a = np.array([len(x.split()) for x, d in ai if d == dm and len(x.split()) >= 40], float)
        h = np.array([len(x.split()) for x, d in hum if d == dm and len(x.split()) >= 40], float)
        if len(a) < 20 or len(h) < 20:
            continue
        s = np.concatenate([a, h]); y = np.array([1] * len(a) + [0] * len(h))
        val, d = sym(auc(s, y))
        worst = max(worst, val)
        col = "red" if val >= 0.85 else ("yellow" if val >= 0.70 else "green")
        t.add_row(dm, f"[{col}]{val:.4f}[/]", d, f"{np.median(a):.0f}", f"{np.median(h):.0f}",
                  f"{len(a)}/{len(h)}")
        print(f"RESULT[wordcount RAID {dm}] auc={val:.4f}")
    console.print(t)
    v = "BROKEN" if worst >= 0.90 else ("CAVEAT" if worst >= 0.75 else "CLEAN")
    say("LOGO/RAID length confound", v,
        f"worst per-domain word-count AUC = {worst:.3f}. "
        + ("The flagship 0.9988 is contaminated by length." if v == "BROKEN"
           else "Length carries some signal; must be reported." if v == "CAVEAT"
           else "Length alone cannot separate the classes -- the headline is about provenance."))


# =========================================================================== 2
def audit_logo_grid():
    """Do the per-domain LOGO tables average over a CHANGING cell set?"""
    console.rule("[bold]2. LOGO by-domain -- composition drift (3575 rows vs 3600 expected)")
    try:
        D = pd.read_parquet("results_v2/logo_detection_bydomain_clean.parquet")
    except Exception as e:
        console.print(f"[red]{e}[/]"); return
    g, s, dm, det = (D.generator.nunique(), D.seed.nunique(),
                     D.domain.nunique(), D.detector.nunique())
    full = g * s * dm * det
    console.print(f"generators={g} seeds={s} domains={dm} detectors={det}  "
                  f"-> full grid = {full}, actual = {len(D)}, MISSING = {full - len(D)}")
    # where are the holes?
    cnt = D.groupby(["domain", "detector"]).size().unstack(fill_value=0)
    exp = g * s
    console.print(f"[dim]expected {exp} rows per (domain, detector) cell[/]")
    t = Table(title="cells per (domain x detector) -- anything != %d is a hole" % exp)
    t.add_column("domain")
    for c in cnt.columns:
        t.add_column(str(c), justify="right")
    holes = 0
    for dmn, row in cnt.iterrows():
        cells = []
        for c in cnt.columns:
            v = int(row[c])
            if v != exp:
                holes += exp - v
                cells.append(f"[red]{v}[/]")
            else:
                cells.append(str(v))
        t.add_row(str(dmn), *cells)
    console.print(t)
    # does dropping incomplete (generator,seed,domain) triples change the domain means?
    key = ["generator", "seed", "domain"]
    complete = D.groupby(key).detector.nunique() == det
    keep = set(complete[complete].index)
    Dc = D[[tuple(x) in keep for x in D[key].itertuples(index=False, name=None)]]
    a = D.groupby(["domain", "detector"]).auc.mean().unstack()
    b = Dc.groupby(["domain", "detector"]).auc.mean().unstack()
    delta = (a - b).abs().max().max()
    console.print(f"max |mean AUC shift| when restricted to complete cells: [bold]{delta:.4f}[/]")
    print(f"RESULT[logo_bydomain missing]={full-len(D)} max_shift={delta:.4f}")
    v = "CLEAN" if delta < 0.002 else ("CAVEAT" if delta < 0.01 else "BROKEN")
    say("LOGO by-domain composition", v,
        f"{full-len(D)} missing cells; restricting to the complete sub-grid moves the "
        f"reported domain means by at most {delta:.4f}.")


# =========================================================================== 3
def audit_logo_stats():
    """Paired significance of LOGO differences across 90
    measurements (18 generators x 5 seeds) on identical held-out documents."""
    console.rule("[bold]3. LOGO -- paired significance")
    try:
        L = pd.read_parquet("results_v2/logo_detection_clean.parquet")
    except Exception as e:
        console.print(f"[red]{e}[/]"); return
    dets = sorted({c[:-4] for c in L.columns if c.endswith("_auc")})
    console.print(f"{len(L)} paired measurements; detectors: {dets}")
    for metric in ("auc", "tpr"):
        t = Table(title=f"LOGO {metric.upper()} -- GLASS vs each baseline "
                        f"(Wilcoxon signed-rank, {len(L)} pairs)")
        for c in ["baseline", "GLASS", "baseline ", "delta", "p-value", "significant"]:
            t.add_column(c, justify="right")
        g = L[f"GLASS_{metric}"].values
        for d in dets:
            if d == "GLASS":
                continue
            b = L[f"{d}_{metric}"].values
            ok = np.isfinite(g) & np.isfinite(b)
            if ok.sum() < 10 or np.allclose(g[ok], b[ok]):
                continue
            p = wilcoxon(g[ok], b[ok]).pvalue
            dl = float(np.mean(g[ok] - b[ok]))
            sig = "[green]yes[/]" if p < 0.05 else "[red]NO[/]"
            t.add_row(d, f"{np.mean(g[ok]):.4f}", f"{np.mean(b[ok]):.4f}",
                      f"{dl:+.4f}", f"{p:.2e}", sig)
            print(f"RESULT[logo {metric} GLASS-{d}] delta={dl:+.4f} p={p:.3e}")
        console.print(t)
    say("LOGO significance", "CLEAN",
        "90 paired measurements; Wilcoxon computed for every baseline (see p-values).")


# =========================================================================== 4
def audit_attack():
    """The attack table averages over the SAME five corpora -- including Pile."""
    console.rule("[bold]4. Attack table -- does Pile contaminate it too?")
    try:
        A = pd.read_parquet("results_v2/thirdparty_attack_v3clean.parquet")
    except Exception as e:
        console.print(f"[red]{e}[/]"); return
    console.print(f"corpora={sorted(A.corpus.unique())}  attacks={sorted(A.attack.unique())}  "
                  f"detectors={A.detector.nunique()}")
    full = A.corpus.nunique() * A.attack.nunique() * A.detector.nunique()
    console.print(f"grid {full} vs {len(A)} rows -> "
                  + ("[green]complete[/]" if full == len(A) else "[red]INCOMPLETE[/]"))
    for label, keep in [("ALL corpora (as published)", sorted(A.corpus.unique())),
                        ("Pile dropped", [c for c in sorted(A.corpus.unique()) if c != "Pile"])]:
        S = A[A.corpus.isin(keep)]
        P = S.pivot_table(index="detector", columns="attack", values="auc", aggfunc="mean")
        if "clean" in P.columns:
            atk = [c for c in P.columns if c != "clean"]
            P["worst drop"] = (P["clean"].values[:, None] - P[atk].values).max(1)
            P = P.sort_values("worst drop")
        console.print(f"\n[bold]{label}[/]")
        console.print(P.round(3).to_string())
        if "GLASS" in P.index:
            r = int(P.index.get_loc("GLASS")) + 1
            print(f"RESULT[attack {label}] GLASS worst-drop rank={r}/{len(P)}")
    say("Attack table", "CAVEAT",
        "Pile is one of the five corpora averaged here. Recompute without it "
        "(see the two blocks above -- if the ranking moves, the published table is wrong).")


# =========================================================================== 5
def audit_xcorpus_tables():
    """Cross-check every stored cross-corpus parquet against the published table."""
    console.rule("[bold]5. Cross-corpus parquet vs published Table 11")
    try:
        T = pd.read_parquet("results_v2/thirdparty_v3clean.parquet")
    except Exception as e:
        console.print(f"[red]{e}[/]"); return
    P = T.pivot_table(index="detector", columns="corpus", values="auc")
    for label, keep in [("ALL five (published)", list(P.columns)),
                        ("Pile dropped", [c for c in P.columns if c != "Pile"])]:
        Q = P[keep].copy()
        Q["mean"] = Q.mean(1); Q["worst"] = Q.min(1)
        Q = Q.sort_values("mean", ascending=False)
        console.print(f"\n[bold]{label}[/]")
        console.print(Q.round(3).to_string())
        if "GLASS" in Q.index:
            print(f"RESULT[xcorpus {label}] GLASS mean={Q.loc['GLASS','mean']:.4f} "
                  f"worst={Q.loc['GLASS','worst']:.4f} "
                  f"rank={int(Q.index.get_loc('GLASS'))+1}")
    say("Cross-corpus table", "BROKEN",
        "Pile has word-count AUC 0.999 -- it is a length benchmark and must be removed "
        "from the headline mean/worst.")


if __name__ == "__main__":
    for fn in (audit_logo_length, audit_logo_grid, audit_logo_stats,
               audit_attack, audit_xcorpus_tables):
        try:
            fn()
        except Exception as e:
            console.print(f"[red]{fn.__name__} FAILED: {type(e).__name__}: {e}[/]\n")
    console.rule("[bold]SUMMARY")
    t = Table()
    t.add_column("table"); t.add_column("verdict"); t.add_column("why")
    for k, (v, w) in VERDICT.items():
        col = {"CLEAN": "green", "CAVEAT": "yellow", "BROKEN": "red"}[v]
        t.add_row(k, f"[{col}]{v}[/]", w)
    console.print(t)

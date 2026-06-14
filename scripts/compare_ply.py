
"""
Uso:
    python compare_ply.py reference.ply mine.ply [--save-report report.txt]

Analisi incluse:
  - Opacity: distribuzione, bloom, bimodalità
  - Scale: distribuzione, gaussiane grandi/piccole, aspect ratio
  - Quaternioni: normalizzazione, distribuzione quatW
  - SH coefficients: energia f_dc, f_rest, colore medio
  - Spatial: bounding box, densità locale, distribuzione per distanza
  - Training quality score: stima di convergenza
  - Labels: distribuzione per classe
"""

import sys
import argparse
import numpy as np
from plyfile import PlyData


# ---------------------------------------------------------------------------
# Loader
# ---------------------------------------------------------------------------

def load_ply(path):
    ply  = PlyData.read(path)
    v    = ply["vertex"]
    props = {p.name for p in v.properties}

    def get(name):
        return np.array(v[name], dtype=np.float32) if name in props else None

    xyz = np.stack([v["x"], v["y"], v["z"]], axis=1).astype(np.float32)

    opacity_raw = get("opacity")
    opacity = 1.0 / (1.0 + np.exp(-opacity_raw)) if opacity_raw is not None else None

    scale_cols = [f"scale_{i}" for i in range(3) if f"scale_{i}" in props]
    if scale_cols:
        scales = np.stack([v[c] for c in scale_cols], axis=1).astype(np.float32)
        if scales.size > 0 and np.nanmin(scales) < 0:
            scales = np.exp(np.clip(scales, -20, 20))
    else:
        scales = None

    f_dc = None
    if "f_dc_0" in props:
        dc_cols = sorted([p.name for p in v.properties if p.name.startswith("f_dc_")])
        f_dc = np.stack([v[c] for c in dc_cols], axis=1).astype(np.float32)

    rest_cols = sorted([p.name for p in v.properties if p.name.startswith("f_rest_")])
    f_rest = np.stack([v[c] for c in rest_cols], axis=1).astype(np.float32) if rest_cols else None

    rot_cols = [f"rot_{i}" for i in range(4) if f"rot_{i}" in props]
    rots = np.stack([v[c] for c in rot_cols], axis=1).astype(np.float32) if rot_cols else None

    label = get("label")

    return {
        "path": path, "xyz": xyz, "opacity": opacity, "opacity_raw": opacity_raw,
        "scales": scales, "f_dc": f_dc, "f_rest": f_rest,
        "rots": rots, "label": label, "props": props, "n": len(xyz),
    }


# ---------------------------------------------------------------------------
# Utility
# ---------------------------------------------------------------------------

def pct(arr, p):
    return float(np.percentile(arr.flatten(), p))

def stats_line(arr, name, indent=4):
    if arr is None:
        return " " * indent + f"{name}: N/A"
    flat = arr.flatten()
    return (
        " " * indent + f"{name}: n={len(arr):,}  "
        f"mean={flat.mean():.4f}  std={flat.std():.4f}  "
        f"min={flat.min():.4f}  p25={pct(flat,25):.4f}  "
        f"p50={pct(flat,50):.4f}  p75={pct(flat,75):.4f}  "
        f"max={flat.max():.4f}"
    )

def bimodality_coefficient(arr):
    """BMC > 0.555 suggerisce distribuzione bimodale (gaussiane convergenti)."""
    flat = arr.flatten()
    n    = len(flat)
    if n < 4:
        return 0.0
    m3 = float(np.mean((flat - flat.mean())**3)) / (float(np.std(flat))**3 + 1e-9)
    m4 = float(np.mean((flat - flat.mean())**4)) / (float(np.std(flat))**4 + 1e-9)
    return (m3**2 + 1) / (m4 + 3 * (n-1)**2 / ((n-2)*(n-3) + 1e-9))

def section(title, width=62):
    return f"\n{'='*width}\n  {title}\n{'='*width}"

def subsection(title, width=62):
    return f"\n  {'─'*int(width*0.9)}\n  {title}\n  {'─'*int(width*0.9)}"


# ---------------------------------------------------------------------------
# Per-PLY report
# ---------------------------------------------------------------------------

def report_single(d, tag, lines):
    def p(s=""):
        lines.append(s)

    p(section(f"{tag}:  {d['path']}"))
    p(f"  Gaussiane totali : {d['n']:,}")
    p(f"  Proprietà PLY    : {len(d['props'])} colonne")

    # ── Opacity ─────────────────────────────────────────────────────────────
    op = d["opacity"]
    if op is not None:
        p(subsection("OPACITY"))
        high = (op > 0.9).mean() * 100
        med  = ((op >= 0.5) & (op <= 0.9)).mean() * 100
        low  = (op < 0.5).mean() * 100
        near_zero = (op < 0.05).mean() * 100
        p(f"    > 0.9  (bloom risk)     : {high:.1f}%")
        p(f"    0.5–0.9 (opaque)        : {med:.1f}%")
        p(f"    0.05–0.5 (translucent)  : {low - near_zero:.1f}%")
        p(f"    < 0.05  (invisible/dead): {near_zero:.1f}%")
        bmc = bimodality_coefficient(op)
        convergence = "✓ bimodale (ben convergente)" if bmc > 0.555 else "✗ unimodale (convergenza parziale)"
        p(f"    Bimodality coeff (BMC)  : {bmc:.3f}  →  {convergence}")
        p(stats_line(op, "opacity"))

        # Raw logit range
        if d["opacity_raw"] is not None:
            raw = d["opacity_raw"]
            p(f"    opacity_raw (logit): min={raw.min():.3f}  max={raw.max():.3f}  mean={raw.mean():.3f}")
            if raw.max() < 3.1:
                p(f"    ⚠  logit_max={raw.max():.2f} < 3.1 → opacity cap attivo, possibile cloudiness")

    # ── Scale ────────────────────────────────────────────────────────────────
    sc = d["scales"]
    if sc is not None and sc.size > 0:
        p(subsection("SCALE  (post-exp, unità mondo)"))
        max_sc  = sc.max(axis=1)
        mean_sc = sc.mean(axis=1)
        min_sc  = sc.min(axis=1)

        p(f"    max_scale > 0.5 (blur risk) : {(max_sc > 0.5).mean()*100:.1f}%")
        p(f"    max_scale > 1.0 (large)     : {(max_sc > 1.0).mean()*100:.1f}%")
        p(f"    max_scale < 0.05 (tiny)     : {(max_sc < 0.05).mean()*100:.1f}%")

        # Aspect ratio: scala più grande / scala più piccola
        ar = max_sc / (min_sc + 1e-8)
        p(f"    Aspect ratio (max/min scale):")
        p(f"      mean={ar.mean():.2f}  p50={np.percentile(ar,50):.2f}  "
          f"p90={np.percentile(ar,90):.2f}  p99={np.percentile(ar,99):.2f}")
        p(f"      AR > 5 (needle-like) : {(ar > 5).mean()*100:.1f}%")
        p(f"      AR > 10 (degenerate) : {(ar > 10).mean()*100:.1f}%")

        p(stats_line(max_sc,  "max_scale_per_gaussian"))
        p(stats_line(mean_sc, "mean_scale_per_gaussian"))

    # ── Quaternioni ──────────────────────────────────────────────────────────
    rots = d["rots"]
    if rots is not None and rots.shape[1] == 4:
        p(subsection("QUATERNIONI"))
        norms = np.linalg.norm(rots, axis=1)
        bad_norm = (np.abs(norms - 1.0) > 0.01).mean() * 100
        p(f"    Norma quaternione: mean={norms.mean():.5f}  std={norms.std():.5f}")
        p(f"    |norm - 1| > 0.01 : {bad_norm:.2f}%  {'⚠ necessita normalizzazione' if bad_norm > 1 else '✓ ok'}")

        # quatW = rot_0 (componente scalare)
        qw = rots[:, 0]
        p(stats_line(qw, "quatW (rot_0)"))
        neg_qw = (qw < 0).mean() * 100
        p(f"    quatW < 0 : {neg_qw:.1f}%  {'⚠ molti quaternioni con W negativo — possibile anomalia' if neg_qw > 40 else '✓ ok'}")
        p(f"    quatW ≈ 0 (|qw|<0.1) : {(np.abs(qw) < 0.1).mean()*100:.1f}%")

    # ── SH Coefficients ──────────────────────────────────────────────────────
    if d["f_dc"] is not None:
        p(subsection("SPHERICAL HARMONICS"))
        f_dc = d["f_dc"]
        # Converti SH DC → colore RGB (SH DC = (color - 0.5) / 0.28209)
        rgb_approx = f_dc * 0.28209 + 0.5
        rgb_approx = np.clip(rgb_approx, 0, 1)
        p(f"    f_dc → RGB approx: R={rgb_approx[:,0].mean():.3f}  "
          f"G={rgb_approx[:,1].mean():.3f}  B={rgb_approx[:,2].mean():.3f}")
        p(f"    f_dc range: [{f_dc.min():.3f}, {f_dc.max():.3f}]")
        saturated = (np.abs(f_dc) > 5.0).any(axis=1).mean() * 100
        p(f"    f_dc saturati (|x|>5) : {saturated:.1f}%  {'⚠ colori saturi/clippati' if saturated > 5 else '✓ ok'}")

        if d["f_rest"] is not None:
            rest_e = (d["f_rest"]**2).sum(axis=1)
            p(f"    f_rest energy: mean={rest_e.mean():.6f}  max={rest_e.max():.6f}")
            p(f"    f_rest tutti zero: {(d['f_rest'] == 0).all()}")
            if (d["f_rest"] == 0).all():
                p(f"    ℹ  f_rest=0 → colore piatto (nessuna variazione direzionale)")
        else:
            p(f"    f_rest: assenti nel PLY")

    # ── Spatial ──────────────────────────────────────────────────────────────
    xyz = d["xyz"]
    p(subsection("DISTRIBUZIONE SPAZIALE"))
    bb_min = xyz.min(axis=0)
    bb_max = xyz.max(axis=0)
    bb_size = bb_max - bb_min
    p(f"    Bounding box: X=[{bb_min[0]:.2f}, {bb_max[0]:.2f}]  "
      f"Y=[{bb_min[1]:.2f}, {bb_max[1]:.2f}]  Z=[{bb_min[2]:.2f}, {bb_max[2]:.2f}]")
    p(f"    Dimensioni  : {bb_size[0]:.2f} × {bb_size[1]:.2f} × {bb_size[2]:.2f}")

    # Distanza dall'origine — utile per capire se il cielo occupa gaussiane lontane
    dists = np.linalg.norm(xyz, axis=1)
    p(stats_line(dists, "dist_from_origin"))
    far = (dists > np.percentile(dists, 90)).mean() * 100
    p(f"    Gaussiane in top-10% distanza: {far:.1f}% "
      f"(soglia={np.percentile(dists, 90):.2f})")

    # Distribuzione verticale (Y o Z in base a convenzione)
    y = xyz[:, 1]
    p(f"    Asse Y (altezza): min={y.min():.2f}  p10={np.percentile(y,10):.2f}  "
      f"p50={np.percentile(y,50):.2f}  p90={np.percentile(y,90):.2f}  max={y.max():.2f}")
    high_y = (y > np.percentile(y, 80)).mean() * 100
    p(f"    Gaussiane in top-20% altezza (cielo?): {high_y:.1f}%")

    # ── Labels ───────────────────────────────────────────────────────────────
    if d["label"] is not None:
        lbl = d["label"].astype(np.int64)
        unique, counts = np.unique(lbl, return_counts=True)
        p(subsection("LABELS"))
        p(f"    Valori unici: {len(unique)}  range=[{unique.min()}, {unique.max()}]")
        top_k = min(10, len(unique))
        top_idx = np.argsort(counts)[::-1][:top_k]
        p(f"    Top-{top_k} classi per frequenza:")
        for idx in top_idx:
            pct_lbl = counts[idx] / len(lbl) * 100
            p(f"      label={unique[idx]:>10}  count={counts[idx]:>8,}  ({pct_lbl:.1f}%)")
    else:
        p(subsection("LABELS"))
        p("    Non presenti nel PLY")

    # ── Training quality score ────────────────────────────────────────────────
    p(subsection("TRAINING QUALITY SCORE (stima)"))
    score  = 0
    maxsco = 0
    issues = []

    if op is not None:
        maxsco += 3
        bmc = bimodality_coefficient(op)
        if bmc > 0.555:
            score += 3
        elif bmc > 0.4:
            score += 2
            issues.append("Opacity parzialmente bimodale — training non completamente convergente")
        else:
            score += 0
            issues.append("Opacity unimodale — training probabilmente non convergente o troppo breve")

        if (op < 0.05).mean() > 0.5:
            issues.append("⚠ >50% gaussiane con opacity<0.05 — optimizer bloccato o opacity reset recente")

    if sc is not None and sc.size > 0:
        maxsco += 2
        if sc.max(axis=1).mean() < 0.6:
            score += 2
        elif sc.max(axis=1).mean() < 1.0:
            score += 1
            issues.append("Scale medie elevate — possibile blur residuo")
        else:
            issues.append("Scale medie troppo alte — blur geometrico probabile")

    if rots is not None:
        maxsco += 1
        if bad_norm < 1:
            score += 1
        else:
            issues.append(f"Quaternioni non normalizzati ({bad_norm:.1f}%)")

    if d["f_dc"] is not None:
        maxsco += 1
        if saturated < 5:
            score += 1
        else:
            issues.append("f_dc saturati — colori clippati")

    pct_score = score / maxsco * 100 if maxsco > 0 else 0
    grade = "🟢 Buono" if pct_score >= 75 else ("🟡 Discreto" if pct_score >= 50 else "🔴 Problemi")
    p(f"    Score: {score}/{maxsco}  ({pct_score:.0f}%)  →  {grade}")
    if issues:
        for iss in issues:
            p(f"    • {iss}")


# ---------------------------------------------------------------------------
# Comparazione
# ---------------------------------------------------------------------------

def report_compare(ref, mine, lines):
    def p(s=""):
        lines.append(s)

    p(section("DIFF COMPARATIVA  (mine − reference)"))

    ratio = mine["n"] / max(ref["n"], 1)
    p(f"  Gaussiane: ref={ref['n']:,}  mine={mine['n']:,}  ratio={ratio:.2f}x")
    if ratio < 0.5:
        p(f"  ⚠ Meno della metà delle gaussiane del reference — dettaglio insufficiente")
    elif ratio < 0.8:
        p(f"  ⚠ -20% gaussiane rispetto al reference — più iter/densify consigliati")

    if ref["opacity"] is not None and mine["opacity"] is not None:
        d_op = mine["opacity"].mean() - ref["opacity"].mean()
        p(f"\n  Opacity media   : ref={ref['opacity'].mean():.4f}  mine={mine['opacity'].mean():.4f}  Δ={d_op:+.4f}")
        d_high = (mine["opacity"] > 0.9).mean() - (ref["opacity"] > 0.9).mean()
        p(f"  % opacity>0.9   : ref={(ref['opacity']>0.9).mean()*100:.1f}%  "
          f"mine={(mine['opacity']>0.9).mean()*100:.1f}%  Δ={d_high*100:+.1f}pp")

    if ref["scales"] is not None and mine["scales"] is not None:
        d_sc = mine["scales"].mean() - ref["scales"].mean()
        p(f"  Scale media     : ref={ref['scales'].mean():.4f}  mine={mine['scales'].mean():.4f}  Δ={d_sc:+.4f}")
        d_large = (mine["scales"].max(axis=1) > 0.5).mean() - (ref["scales"].max(axis=1) > 0.5).mean()
        p(f"  % scale>0.5     : ref={(ref['scales'].max(axis=1)>0.5).mean()*100:.1f}%  "
          f"mine={(mine['scales'].max(axis=1)>0.5).mean()*100:.1f}%  Δ={d_large*100:+.1f}pp")

    if ref["rots"] is not None and mine["rots"] is not None:
        qw_ref  = ref["rots"][:, 0]
        qw_mine = mine["rots"][:, 0]
        d_qw = qw_mine.mean() - qw_ref.mean()
        p(f"  quatW medio     : ref={qw_ref.mean():.4f}  mine={qw_mine.mean():.4f}  Δ={d_qw:+.4f}")
        neg_ref  = (qw_ref  < 0).mean() * 100
        neg_mine = (qw_mine < 0).mean() * 100
        p(f"  quatW < 0       : ref={neg_ref:.1f}%  mine={neg_mine:.1f}%  "
          f"{'⚠ anomalia quatW nel tuo PLY' if abs(neg_mine - neg_ref) > 20 else '✓ ok'}")

    if ref["f_rest"] is not None and mine["f_rest"] is not None:
        e_ref  = (ref["f_rest"]**2).sum(axis=1).mean()
        e_mine = (mine["f_rest"]**2).sum(axis=1).mean()
        p(f"  SH f_rest energy: ref={e_ref:.6f}  mine={e_mine:.6f}")

    # Spatial overlap
    p(f"\n  Bounding box overlap:")
    for d, label in [(ref, "ref "), (mine, "mine")]:
        xyz = d["xyz"]
        bb_min = xyz.min(axis=0); bb_max = xyz.max(axis=0)
        p(f"    {label}: X=[{bb_min[0]:.1f},{bb_max[0]:.1f}]  "
          f"Y=[{bb_min[1]:.1f},{bb_max[1]:.1f}]  Z=[{bb_min[2]:.1f},{bb_max[2]:.1f}]")

    # Diagnosi automatica
    p(f"\n  {'─'*55}")
    p(f"  DIAGNOSI AUTOMATICA")
    p(f"  {'─'*55}")
    issues = []

    if mine["opacity"] is not None and ref["opacity"] is not None:
        if mine["opacity"].mean() < 0.06 and (mine["opacity"] < 0.06).mean() > 0.95:
            issues.append("🔴 CRITICO: quasi tutte le opacity a 0.05 → optimizer bloccato o opacity reset a fine training")
        elif mine["opacity"].mean() > ref["opacity"].mean() + 0.1:
            issues.append("🟡 Opacity media troppo alta → rischio bloom")

    if mine["scales"] is not None:
        if mine["scales"].mean() > 0.5:
            issues.append("🟡 Scale medie elevate → blur geometrico probabile")

    if mine["rots"] is not None:
        qw = mine["rots"][:, 0]
        if (qw < 0).mean() > 0.6:
            issues.append("🟡 >60% quatW negativi → distribuzione quaternioni anomala (doppia copertura)")
        norms = np.linalg.norm(mine["rots"], axis=1)
        if (np.abs(norms - 1.0) > 0.01).mean() > 0.01:
            issues.append("🟡 Quaternioni non normalizzati → rotazioni instabili")

    if mine["n"] < ref["n"] * 0.5:
        issues.append(f"🟡 Solo {ratio:.2f}x le gaussiane del reference → servono più iter/densify")

    if mine["f_dc"] is not None:
        saturated = (np.abs(mine["f_dc"]) > 5.0).any(axis=1).mean() * 100
        if saturated > 10:
            issues.append("🟡 f_dc saturati → colori clippati nel rendering")

    if issues:
        for iss in issues:
            p(f"  {iss}")
    else:
        p("  ✓ Nessuna differenza critica rilevata automaticamente.")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Confronta due PLY Gaussian Splatting")
    parser.add_argument("reference", help="PLY di riferimento")
    parser.add_argument("mine",      help="Tuo PLY da analizzare")
    parser.add_argument("--save-report", metavar="FILE", help="Salva report su file di testo")
    args = parser.parse_args()

    print(f"Caricamento {args.reference}...")
    ref  = load_ply(args.reference)
    print(f"Caricamento {args.mine}...")
    mine = load_ply(args.mine)

    lines = []
    report_single(ref,  "REFERENCE", lines)
    report_single(mine, "TUO PLY",   lines)
    report_compare(ref, mine, lines)

    output = "\n".join(lines)
    print(output)

    if args.save_report:
        with open(args.save_report, "w") as f:
            f.write(output)
        print(f"\nReport salvato in: {args.save_report}")

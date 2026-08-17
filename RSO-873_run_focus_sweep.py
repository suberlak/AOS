"""Focus sweep on 20260630 seq 96-110 (BLOCK-T743, HD 121638, empty~SDSSy_65mm).

These are SINGLE exposures stepping focusZ from -0.193 to +0.207 mm, not intra/extra
pairs, so Danish cannot be run on them directly. What they do give, independently of
any wavefront fit, is the classical focus curve: image size vs focusZ.

Two things come out of it:

1. best focus, from the vertex of the fitted curve;
2. a check on the defocus->image-size scaling that underpins the whole zkRef issue.
   For f/18 and the M2->detector magnification of 43.7, a focusZ offset dz (mm of M2)
   should give a geometric blur diameter of dz*43.7/18 mm = dz*2.43 mm, i.e.
   dz*243 pixels at 10um -- so ~24 px radius at dz = 0.2 mm.

If the measured slope of size vs |dz| matches that, the 43.7x magnification and the
f/18 geometry are confirmed on sky, which is the same geometry that makes
getOffAxisCoeff's transverse aberration 43.5x too large for danish.
"""
import csv
import sys

import numpy as np

from lsst.daf.butler import Butler
from lsst.pipe.tasks.quickFrameMeasurement import QuickFrameMeasurementTask
from lsst.summit.utils.bestEffort import BestEffortIsr
from lsst.ts.wep.utils import getTaskInstrument

OUT_CSV = "/sdf/data/rubin/user/scichris/WORK/AOS/RSO-873_focus_sweep.csv"
# All direct-imaging focus sweeps found in the archive (BLOCK-T743). Dispersed
# sequences (holo4_003 / 300lpmm / prism / notch) are excluded -- those are spectra.
SWEEPS = [
    (20260623, 379, 391, "HD 173657"),
    (20260623, 457, 481, "HD 173657"),
    (20260623, 488, 500, "HD 187605"),
    (20260624, 602, 616, "HD 198842"),
    (20260624, 692, 706, "HD 198842"),
    (20260630, 96, 110, "HD 121638"),
    (20260630, 186, 200, "HD 121638"),
]
# Only direct-imaging frames: holo4_003 / 300lpmm / prism / notch are dispersed (spectra).
GOOD_FILTERS = ("empty~SDSSy_65mm", "empty~empty", "SDSSy_65mm~empty")

butler = Butler("/repo/main")
best = BestEffortIsr(repoString="/repo/main", doWrite=False)
qfm = QuickFrameMeasurementTask(config=QuickFrameMeasurementTask.ConfigClass())
inst = getTaskInstrument("LATISS", "RXX_S00", None)

recs = []
for day, s0, s1, tgt in SWEEPS:
    got = sorted(
        butler.registry.queryDimensionRecords(
            "exposure",
            where=(
                "instrument='LATISS' and exposure.observation_type='focus' "
                f"and exposure.day_obs={day} and exposure.seq_num>={s0} "
                f"and exposure.seq_num<={s1}"
            ),
            instrument="LATISS",
        ),
        key=lambda r: r.seq_num,
    )
    got = [r for r in got if not any(
        d in r.physical_filter for d in ("holo4_003", "300lpmm", "prism", "notch")
    )]
    for r in got:
        recs.append((f"{day}_{s0}-{s1}", r))
print(f"{len(recs)} direct-imaging focus frames over {len(SWEEPS)} sweeps", flush=True)

rows = []
for sweep_id, r in recs:
    try:
        exp = best.getExposure(
            {"instrument": "LATISS", "exposure": r.id, "detector": 0}
        )
        focus_z = float(exp.visitInfo.focusZ)
        # Near focus the source is a PSF; at the ends it is a small donut. Ask
        # QuickFrameMeasurement for a generous box either way.
        res = qfm.run(exp.clone(), donutDiameter=120)
        if not res.success:
            print(f"  seq={r.seq_num} focusZ={focus_z:+.4f} QFM FAILED", flush=True)
            continue
        # Second-moment size, robust for both PSFs and small donuts.
        cx, cy = res.brightestObjCentroidCofM
        arr = np.asarray(exp.image.array, dtype=float)
        half = 60
        y0, y1 = int(cy) - half, int(cy) + half + 1
        x0, x1 = int(cx) - half, int(cx) + half + 1
        if y0 < 0 or x0 < 0 or y1 > arr.shape[0] or x1 > arr.shape[1]:
            print(f"  seq={r.seq_num} too close to the edge", flush=True)
            continue
        cut = arr[y0:y1, x0:x1]
        cut = cut - np.median(cut)
        w = np.clip(cut, 0, None)
        tot = w.sum()
        yy, xx = np.indices(cut.shape)
        mx, my = (w * xx).sum() / tot, (w * yy).sum() / tot
        ixx = (w * (xx - mx) ** 2).sum() / tot
        iyy = (w * (yy - my) ** 2).sum() / tot
        rms_px = float(np.sqrt(0.5 * (ixx + iyy)))
        row = dict(
            sweep=sweep_id,
            exposure=r.id,
            seq_num=r.seq_num,
            focus_z=focus_z,
            physical_filter=r.physical_filter,
            rms_px=rms_px,
            fwhm_px=float(2.355 * rms_px),
            # medianXxYy is a (xx, yy) tuple, not a scalar
            qfm_fwhm=float(np.mean(np.atleast_1d(getattr(res, "medianXxYy", np.nan)))),
            peak=float(np.nanmax(cut)),
        )
        rows.append(row)
        print(
            f"  seq={r.seq_num} focusZ={focus_z:+.4f} rms={rms_px:6.2f}px "
            f"fwhm={row['fwhm_px']:6.2f}px",
            flush=True,
        )
    except Exception as exc:
        print(f"  seq={r.seq_num} FAILED {type(exc).__name__}: {str(exc)[:70]}", flush=True)

if not rows:
    sys.exit("no usable frames")

with open(OUT_CSV, "w", newline="") as f:
    w_ = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
    w_.writeheader()
    w_.writerows(rows)
print(f"\nwrote {OUT_CSV} ({len(rows)} rows)")

groups = {}
for r in rows:
    groups.setdefault(r["sweep"], []).append(r)

mag = inst.defocalOffset / inst.batoidOffsetValue
f_ratio = inst.focalLength / (2 * inst.radius)
pred_diam_px_per_mm = mag / f_ratio * 1e-3 / inst.pixelSize
pred_rms_px_per_mm = pred_diam_px_per_mm / 4.0
print()
print(f"geometry: magnification {mag:.1f}x, f/{f_ratio:.1f}")
print(f"  predicted blur DIAMETER {pred_diam_px_per_mm:.0f} px per mm of focusZ")
print(f"  uniform disk of diameter D has rms D/4 -> {pred_rms_px_per_mm:.0f} px rms/mm")
print()

Z4_PER_MM = 4231.0  # nm of wavefront Z4 per mm of M2 motion, from batoid
ks = []
for sweep_id, rs in sorted(groups.items()):
    if len(rs) < 5:
        print(f"{sweep_id}: only {len(rs)} frames, skipping fit")
        continue
    fz = np.array([r["focus_z"] for r in rs])
    rms = np.array([r["rms_px"] for r in rs])
    # Defocus adds in quadrature with seeing: rms^2 = rms0^2 + (k*(fz-fz0))^2,
    # so rms^2 is a parabola in fz.
    a, b, c = np.polyfit(fz, rms**2, 2)
    if a <= 0:
        print(f"{sweep_id}: parabola opens downward (a={a:.3g}) -- not a clean sweep")
        continue
    fz_best = -b / (2 * a)
    rms0 = np.sqrt(max(c - b**2 / (4 * a), 0.0))
    k = np.sqrt(a)
    ks.append(k)
    print(
        f"{sweep_id}: n={len(rs):2d} best focus {fz_best:+.4f} mm  "
        f"in-focus rms {rms0:5.2f} px ({2.355*rms0*0.0955:.2f}\" fwhm)  "
        f"k={k:6.1f} px rms/mm  ratio to geom {k/pred_rms_px_per_mm:5.2f}  "
        f"-> Z4 {fz_best*Z4_PER_MM:+7.0f} nm"
    )

if ks:
    ks = np.array(ks)
    print()
    print(f"defocus slope k over {len(ks)} sweeps: median {np.median(ks):.1f} px rms/mm, "
          f"std {np.std(ks):.1f}")
    print(f"geometric prediction {pred_rms_px_per_mm:.1f} -> median ratio "
          f"{np.median(ks)/pred_rms_px_per_mm:.2f}")

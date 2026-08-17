"""Stage 2: fit pre-cut stamps with danish, under whichever ts_wep is set up.

Answers the question "given the SAME stamp image, does danish give the same Zernikes in
15.1.0 and in 17.x?" -- because the stamps come from a pickle, the cutout differences
between versions are held fixed and only the algorithm varies.

Runs the *version's own* code path as far as possible:

  * 15.1.0    -> DanishAlgorithm.estimateZk via CalcZernikesTask-equivalent call, whose
                 getOffAxisCoeff already returns OPD-scale values (3.4 um Z4), so no fix
                 is needed and this is the summit reference behaviour.
  * 17.x      -> two variants:
                   'stock'  : the unmodified ts_wep path (expected to be degenerate for
                              LATISS: zkRef 43x too large, nfev=1)
                   'fixed'  : latissMonolith.fit_latiss_danish_arrays, i.e. OPD zkRef and
                              peak-normalized stamps

Usage:
    source .../setup_aos_wep_15.1.0.sh
    PYTHONPATH=.../danish_pre1.0_df9f5fc:$PYTHONPATH \
        python RSO-873_fit_stamps.py RSO-873_stamps_cut17.pkl out_15_on_cut17.csv

    source .../setup_aos_wep_17.8.1_donut_viz_4.7.2_danish_1.2.0.sh
    python RSO-873_fit_stamps.py RSO-873_stamps_cut17.pkl out_17_on_cut17.csv
"""
import csv
import pickle
import sys
from pathlib import Path

import numpy as np

import lsst.ts.wep
from lsst.ts.wep.image import Image
from lsst.ts.wep.utils import DefocalType, getTaskInstrument

NOLL = np.arange(4, 23)
REPO_DIR = Path("/sdf/data/rubin/user/scichris/WORK/AOS")
TS_WEP_TASK_DIR = (
    "/sdf/data/rubin/user/scichris/WORK/aos_packages/ts_wep/python/lsst/ts/wep/task"
)


def make_wep_images(rec):
    """Rebuild ts_wep Image objects from the pickled arrays."""
    out = {}
    for side, dt in (("extra", DefocalType.Extra), ("intra", DefocalType.Intra)):
        out[side] = Image(
            image=np.asarray(rec[f"img_{side}"], dtype=float).copy(),
            fieldAngle=np.asarray(rec[f"fa_{side}"], dtype=float),
            defocalType=dt,
            bandLabel="ref",  # AuxTel.yaml only defines the reference band
        )
    return out


def fit_native(rec, inst):
    """The running ts_wep's OWN danish path, unmodified.

    For 15.1.0 this is the summit reference. For 17.x it is the broken path, kept so the
    comparison shows what the regression actually costs.
    """
    from lsst.ts.wep.estimation.danish import DanishAlgorithm

    imgs = make_wep_images(rec)
    algo = DanishAlgorithm()
    zk, meta = algo.estimateZk(
        imgs["extra"],
        imgs["intra"],
        nollIndices=NOLL,
        instrument=inst,
        saveHistory=False,
        startWithIntrinsic=True,
        returnWfDev=False,
        units="nm",
    )
    return np.asarray(zk, dtype=float), meta


def fit_fixed(rec, inst):
    """The monolith fix: OPD zkRef + peak-normalized stamps."""
    sys.path.insert(0, TS_WEP_TASK_DIR)
    from latissMonolith import fit_latiss_danish_arrays

    out = fit_latiss_danish_arrays(
        rec["img_extra"],
        rec["img_intra"],
        rec["fa_extra"],
        rec["fa_intra"],
        inst,
        band="ref",
        noll_indices=NOLL,
    )
    zk_nm = np.array([out["zernikes_nm"][int(j)] for j in NOLL], dtype=float)
    meta = {
        "fit_success": out["success"],
        "lstsq_nfev": out["nfev"],
        "chi_square": out["cost"],
        "fwhm": out["fwhm"],
    }
    return zk_nm, meta


def as_float(v, default=float("nan")):
    try:
        arr = np.atleast_1d(v)
        return float(arr.ravel()[0])
    except Exception:
        return default


if __name__ == "__main__":
    stamps_path = sys.argv[1] if len(sys.argv) > 1 else str(REPO_DIR / "RSO-873_stamps.pkl")
    out_csv = sys.argv[2] if len(sys.argv) > 2 else str(REPO_DIR / "RSO-873_fit_out.csv")
    mode = sys.argv[3] if len(sys.argv) > 3 else "auto"

    wep_tag = lsst.ts.wep.__file__.split("aos_packages/")[1].split("/python")[0]
    is_15 = "ts_wep_15" in wep_tag
    if mode == "auto":
        # 15.1.0's own path is already correct for LATISS; 17.x needs the fix, but run
        # its stock path too so the regression is visible in the same table.
        modes = ["native"] if is_15 else ["native", "fixed"]
    else:
        modes = [mode]
    print(f"ts_wep = {wep_tag}   modes = {modes}", flush=True)

    with open(stamps_path, "rb") as f:
        data = pickle.load(f)
    print(f"loaded {len(data['pairs'])} pairs, cut by {data['cut_by']}", flush=True)

    inst = getTaskInstrument("LATISS", data["pairs"][0]["detector_name"], None)

    rows = []
    for k, rec in enumerate(data["pairs"]):
        for m in modes:
            tag = f"[{k+1}/{len(data['pairs'])}] {rec['intra']} {m:6s}"
            try:
                zk, meta = (fit_native if m == "native" else fit_fixed)(rec, inst)
                row = dict(
                    wep=wep_tag,
                    mode=m,
                    cut_by=data["cut_by"],
                    day_obs=rec["day_obs"],
                    intra=rec["intra"],
                    extra=rec["extra"],
                    target=rec["target"],
                    focusz_intra=rec["focusz_intra"],
                    focusz_extra=rec["focusz_extra"],
                    peak_extra=rec["peak_extra"],
                    fit_success=as_float(meta.get("fit_success")),
                    nfev=as_float(meta.get("lstsq_nfev")),
                    chi_square=as_float(meta.get("chi_square")),
                    fwhm=as_float(meta.get("fwhm")),
                    **{f"Z{int(j)}": float(zk[i]) for i, j in enumerate(NOLL)},
                )
                rows.append(row)
                print(
                    f"{tag} Z4={row['Z4']:9.1f} Z7={row['Z7']:8.1f} Z8={row['Z8']:8.1f} "
                    f"nfev={row['nfev']:.0f}",
                    flush=True,
                )
            except Exception as exc:
                print(f"{tag} FAILED {type(exc).__name__}: {str(exc)[:80]}", flush=True)

    if rows:
        with open(out_csv, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
        print(f"\nwrote {out_csv} ({len(rows)} rows)")

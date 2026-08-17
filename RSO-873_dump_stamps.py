"""Stage 1 of the version comparison: cut stamps once, save them to disk.

The point of the comparison is "given the SAME stamp image, does danish give the same
Zernikes in 15.1.0 and in 17.x?". So the stamps have to be cut once and reused, not
re-cut per version -- the cutout task itself changed between versions (see
RSO-873 notebook section 6b), and that would otherwise contaminate the answer.

Run this under whichever ts_wep should produce the stamps, e.g.

    source .../setup_aos_wep_17.8.1_donut_viz_4.7.2_danish_1.2.0.sh
    python RSO-873_dump_stamps.py 13 /sdf/.../RSO-873_stamps_cut17.pkl

    source .../setup_aos_wep_15.1.0.sh
    PYTHONPATH=.../danish_pre1.0_df9f5fc:$PYTHONPATH \
        python RSO-873_dump_stamps.py 13 /sdf/.../RSO-873_stamps_cut15.pkl

then feed either pickle to RSO-873_fit_stamps.py under either stack.
"""
import pickle
import sys

import numpy as np
import astropy.units as u
from astropy.table import QTable

import lsst.ts.wep
from lsst.daf.butler import Butler
from lsst.obs.lsst import Latiss
from lsst.pipe.tasks.quickFrameMeasurement import QuickFrameMeasurementTask
from lsst.summit.utils.bestEffort import BestEffortIsr
from lsst.ts.wep.task.cutOutDonutsScienceSensorTask import (
    CutOutDonutsScienceSensorTask,
    CutOutDonutsScienceSensorTaskConfig,
)
from lsst.ts.wep.task.generateDonutCatalogUtils import addVisitInfoToCatTable

DZ = 0.8
DONUT_DIAMETER = 2 * int(np.ceil(192 * 1.1 * DZ / 1.5 / 2.0) * 2)

butler = Butler("/repo/main")
best = BestEffortIsr(repoString="/repo/main", doWrite=False)
qfm = QuickFrameMeasurementTask(config=QuickFrameMeasurementTask.ConfigClass())
cfg = CutOutDonutsScienceSensorTaskConfig()
cfg.donutStampSize = DONUT_DIAMETER
cfg.opticalModel = "onAxis"
cfg.initialCutoutPadding = 40
cut_task = CutOutDonutsScienceSensorTask(config=cfg)
camera = Latiss.getCamera()


def donut_catalog(result, exposure):
    wcs = exposure.getWcs()
    ra, dec = wcs.pixelToSkyArray(*result.brightestObjCentroidCofM[:2], degrees=False)
    t = QTable()
    t["coord_ra"] = ra * u.rad
    t["coord_dec"] = dec * u.rad
    t["centroid_x"] = [result.brightestObjCentroidCofM[0]] * u.pixel
    t["centroid_y"] = [result.brightestObjCentroidCofM[1]] * u.pixel
    t["source_flux"] = [result.brightestObjApFlux70] * u.nJy
    t.meta["blend_centroid_x"] = ""
    t.meta["blend_centroid_y"] = ""
    t.sort("source_flux", reverse=True)
    return addVisitInfoToCatTable(exposure, t)


def find_pairs_first_of_night(day_min=20260514, day_max=20260626, n_per_night=3):
    """The first few CWFS pairs of each night.

    The initial focus sequence of a night is taken before the telescope is aligned, so it
    is where large aberrations (and large intra/extra asymmetry) actually occur -- exactly
    the regime where 15.1.0 and the fixed 17.x path were seen to disagree.
    """
    recs = sorted(
        butler.registry.queryDimensionRecords(
            "exposure",
            where=(
                "instrument='LATISS' and exposure.observation_type='cwfs' "
                f"and exposure.day_obs>={day_min} and exposure.day_obs<={day_max}"
            ),
            instrument="LATISS",
        ),
        key=lambda r: r.id,
    )
    by_day = {}
    for r in recs:
        by_day.setdefault(r.day_obs, []).append(r)
    pairs = []
    for _day, rs in sorted(by_day.items()):
        rs = sorted(rs, key=lambda r: r.seq_num)
        found = 0
        i = 0
        while i < len(rs) - 1 and found < n_per_night:
            a, b = rs[i], rs[i + 1]
            ra_, rb = (a.observation_reason or ""), (b.observation_reason or "")
            if b.seq_num == a.seq_num + 1 and {ra_, rb} == {"intra", "extra"}:
                intra, extra = (a, b) if ra_ == "intra" else (b, a)
                pairs.append((intra, extra))
                found += 1
                i += 2
            else:
                i += 1
    return pairs


def find_pairs(day_min=20260601):
    recs = sorted(
        butler.registry.queryDimensionRecords(
            "exposure",
            where=(
                "instrument='LATISS' and exposure.observation_type='cwfs' "
                f"and exposure.day_obs>{day_min}"
            ),
            instrument="LATISS",
        ),
        key=lambda r: r.id,
    )
    by_day = {}
    for r in recs:
        by_day.setdefault(r.day_obs, []).append(r)
    pairs = []
    for _day, rs in by_day.items():
        rs = sorted(rs, key=lambda r: r.seq_num)
        i = 0
        while i < len(rs) - 1:
            a, b = rs[i], rs[i + 1]
            ra_, rb = (a.observation_reason or ""), (b.observation_reason or "")
            if b.seq_num == a.seq_num + 1 and {ra_, rb} == {"intra", "extra"}:
                intra, extra = (a, b) if ra_ == "intra" else (b, a)
                pairs.append((intra, extra))
                i += 2  # non-overlapping: one exposure must not join two pairs
            else:
                i += 1
    return pairs


def cut_one(intra_rec, extra_rec):
    ei = best.getExposure({"instrument": "LATISS", "exposure": intra_rec.id, "detector": 0})
    ee = best.getExposure({"instrument": "LATISS", "exposure": extra_rec.id, "detector": 0})
    ri = qfm.run(ei.clone(), donutDiameter=DONUT_DIAMETER)
    re_ = qfm.run(ee.clone(), donutDiameter=DONUT_DIAMETER)
    if not (ri.success and re_.success):
        raise RuntimeError(f"QFM failed intra={ri.success} extra={re_.success}")
    cut = cut_task.run([ee, ei], [donut_catalog(re_, ee), donut_catalog(ri, ei)], camera)
    if not len(cut.donutStampsExtra) or not len(cut.donutStampsIntra):
        raise RuntimeError("no stamps")
    se, si = cut.donutStampsExtra[0], cut.donutStampsIntra[0]
    return dict(
        day_obs=intra_rec.day_obs,
        intra=intra_rec.id,
        extra=extra_rec.id,
        target=intra_rec.target_name,
        program=intra_rec.science_program,
        band=str(se.wep_im.bandLabel.value),
        detector_name=se.detector_name,
        focusz_intra=float(ei.visitInfo.focusZ),
        focusz_extra=float(ee.visitInfo.focusZ),
        # the arrays are the whole point -- everything downstream reuses these
        img_extra=np.asarray(se.wep_im.image, dtype=float).copy(),
        img_intra=np.asarray(si.wep_im.image, dtype=float).copy(),
        fa_extra=tuple(float(v) for v in se.wep_im.fieldAngle),
        fa_intra=tuple(float(v) for v in si.wep_im.fieldAngle),
        peak_extra=float(np.nanmax(se.wep_im.image)),
        peak_intra=float(np.nanmax(si.wep_im.image)),
    )


if __name__ == "__main__":
    limit = int(sys.argv[1]) if len(sys.argv) > 1 else 13
    out_path = (
        sys.argv[2]
        if len(sys.argv) > 2
        else "/sdf/data/rubin/user/scichris/WORK/AOS/RSO-873_stamps.pkl"
    )
    wep_tag = lsst.ts.wep.__file__.split("aos_packages/")[1].split("/python")[0]
    print(f"cutting stamps with {wep_tag}", flush=True)

    # "first" selects the opening pairs of each night (large-aberration regime);
    # anything else spreads the sample over the whole date range.
    mode = sys.argv[3] if len(sys.argv) > 3 else "spread"
    if mode == "first":
        pairs = find_pairs_first_of_night()
        print(f"first-of-night mode: {len(pairs)} pairs", flush=True)
    else:
        pairs = find_pairs()
    if limit < len(pairs):
        idx = np.linspace(0, len(pairs) - 1, limit).round().astype(int)
        pairs = [pairs[k] for k in dict.fromkeys(idx)]
    print(f"{len(pairs)} pairs selected", flush=True)

    out = {"cut_by": wep_tag, "donut_diameter": DONUT_DIAMETER, "pairs": []}
    for k, (intra, extra) in enumerate(pairs):
        tag = f"[{k+1}/{len(pairs)}] {intra.day_obs} {intra.id}/{extra.id}"
        try:
            rec = cut_one(intra, extra)
            out["pairs"].append(rec)
            print(
                f"{tag} {rec['target']:12s} shape={rec['img_extra'].shape} "
                f"peak={rec['peak_extra']:.4g}",
                flush=True,
            )
        except Exception as exc:
            print(f"{tag} FAILED {type(exc).__name__}: {str(exc)[:70]}", flush=True)

    with open(out_path, "wb") as f:
        pickle.dump(out, f)
    print(f"\nwrote {out_path} ({len(out['pairs'])} pairs, cut by {wep_tag})")

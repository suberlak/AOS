"""Run the LATISS monolith Danish fit over many CWFS pairs.

Consistency across pairs is the real validation: a single pair cannot tell a good fit
from a lucky one, and 2026071300013/14 is known to be unusually defocused.

Writes a CSV so the notebook can plot without re-running (each pair costs ~1 min).
"""
import csv
import sys
import traceback

import numpy as np
import astropy.units as u
from astropy.table import QTable

sys.path.insert(
    0, "/sdf/data/rubin/user/scichris/WORK/aos_packages/ts_wep/python/lsst/ts/wep/task"
)
from latissMonolith import fit_latiss_danish  # noqa: E402

from lsst.daf.butler import Butler  # noqa: E402
from lsst.obs.lsst import Latiss  # noqa: E402
from lsst.pipe.tasks.quickFrameMeasurement import QuickFrameMeasurementTask  # noqa: E402
from lsst.summit.utils.bestEffort import BestEffortIsr  # noqa: E402
from lsst.ts.wep.task.cutOutDonutsScienceSensorTask import (  # noqa: E402
    CutOutDonutsScienceSensorTask,
    CutOutDonutsScienceSensorTaskConfig,
)
from lsst.ts.wep.task.generateDonutCatalogUtils import addVisitInfoToCatTable  # noqa: E402
from lsst.ts.wep.utils import getTaskInstrument  # noqa: E402

OUT_CSV = "/sdf/data/rubin/user/scichris/WORK/AOS/RSO-873_many_pairs.csv"
NOLL = np.arange(4, 23)
DZ = 0.8
DONUT_DIAMETER = 2 * int(np.ceil(192 * 1.1 * DZ / 1.5 / 2.0) * 2)

butler = Butler("/repo/main")
best = BestEffortIsr(repoString="/repo/main", doWrite=False)
qfm = QuickFrameMeasurementTask(config=QuickFrameMeasurementTask.ConfigClass())
cut_cfg = CutOutDonutsScienceSensorTaskConfig()
cut_cfg.donutStampSize = DONUT_DIAMETER
cut_cfg.opticalModel = "onAxis"
cut_cfg.initialCutoutPadding = 40
cut_task = CutOutDonutsScienceSensorTask(config=cut_cfg)
camera = Latiss.getCamera()
inst = getTaskInstrument("LATISS", "RXX_S00", None)


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


def find_pairs(day_min=20260601):
    """Group cwfs exposures into (intra, extra) pairs by consecutive seq_num."""
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
    by_reason = {}
    for r in recs:
        by_reason.setdefault(r.day_obs, []).append(r)
    pairs = []
    for day, rs in by_reason.items():
        rs = sorted(rs, key=lambda r: r.seq_num)
        i = 0
        while i < len(rs) - 1:
            a, b = rs[i], rs[i + 1]
            ra_, rb = (a.observation_reason or ""), (b.observation_reason or "")
            if b.seq_num == a.seq_num + 1 and {ra_, rb} == {"intra", "extra"}:
                intra, extra = (a, b) if ra_ == "intra" else (b, a)
                pairs.append((intra, extra))
                i += 2  # consume both, so no exposure lands in two pairs
            else:
                i += 1
    return pairs


def run_pair(intra_rec, extra_rec):
    ei = best.getExposure(
        {"instrument": "LATISS", "exposure": intra_rec.id, "detector": 0}
    )
    ee = best.getExposure(
        {"instrument": "LATISS", "exposure": extra_rec.id, "detector": 0}
    )
    ri = qfm.run(ei.clone(), donutDiameter=DONUT_DIAMETER)
    re_ = qfm.run(ee.clone(), donutDiameter=DONUT_DIAMETER)
    if not (ri.success and re_.success):
        raise RuntimeError(f"QFM failed: intra={ri.success} extra={re_.success}")
    cut = cut_task.run(
        [ee, ei], [donut_catalog(re_, ee), donut_catalog(ri, ei)], camera
    )
    if len(cut.donutStampsExtra) == 0 or len(cut.donutStampsIntra) == 0:
        raise RuntimeError("no stamps")
    fit = fit_latiss_danish(
        cut.donutStampsExtra[0], cut.donutStampsIntra[0], inst, noll_indices=NOLL
    )
    nm = fit["zernikes_nm"]
    models = fit["model_images"]
    if models is None:
        res = [float("nan")] * 2
    else:
        res = [
            float(
                np.nansum(np.abs(np.asarray(img) - np.asarray(mod)))
                / np.nansum(np.abs(np.asarray(img)))
            )
            for img, mod in zip(fit["images"], models)
        ]
    return dict(
        day_obs=intra_rec.day_obs,
        intra=intra_rec.id,
        extra=extra_rec.id,
        target=intra_rec.target_name,
        program=intra_rec.science_program,
        band=ee.filter.bandLabel,
        focusz_intra=float(ei.visitInfo.focusZ),
        focusz_extra=float(ee.visitInfo.focusZ),
        nfev=fit["nfev"],
        cost=fit["cost"],
        fwhm=float(fit["fwhm"]),
        success=fit["success"],
        res_extra=res[0],
        res_intra=res[1],
        **{f"Z{j}": nm[j] for j in (4, 5, 6, 7, 8, 9, 10, 11)},
    )


if __name__ == "__main__":
    limit = int(sys.argv[1]) if len(sys.argv) > 1 else 12
    pairs = find_pairs()
    # Spread the sample over the whole date range rather than taking the first N from
    # one or two nights -- night-to-night scatter is exactly what we are measuring.
    if limit < len(pairs):
        idx = np.linspace(0, len(pairs) - 1, limit).round().astype(int)
        pairs = [pairs[k] for k in dict.fromkeys(idx)]
    print(f"found {len(find_pairs())} cwfs pairs; running {len(pairs)} spread over the range",
          flush=True)
    rows = []
    for k, (intra, extra) in enumerate(pairs[:limit]):
        tag = f"[{k+1}/{min(limit,len(pairs))}] {intra.day_obs} {intra.id}/{extra.id}"
        try:
            row = run_pair(intra, extra)
            rows.append(row)
            print(
                f"{tag} {row['target']:12s} Z4={row['Z4']:8.1f} Z7={row['Z7']:7.1f} "
                f"Z8={row['Z8']:7.1f} fwhm={row['fwhm']:.2f} cost={row['cost']:.4g} "
                f"res={row['res_extra']:.3f}/{row['res_intra']:.3f}",
                flush=True,
            )
        except Exception as exc:
            print(f"{tag} FAILED {type(exc).__name__}: {str(exc)[:80]}", flush=True)
            traceback.print_exc()
    if rows:
        with open(OUT_CSV, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
        print(f"\nwrote {OUT_CSV} ({len(rows)} rows)")
        z4 = np.array([r["Z4"] for r in rows])
        print(f"Z4: median={np.median(z4):8.1f} nm  std={np.std(z4):6.1f} nm  "
              f"range=[{z4.min():.0f}, {z4.max():.0f}]")

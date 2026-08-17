#!/usr/bin/env python
"""Assemble AOS_LATISS_RSO-873_wep_align_monolith.ipynb from cell sources.

Kept as a build script so the notebook can be regenerated after edits.
"""
import json
from pathlib import Path

OUT = Path(__file__).resolve().parent / "AOS_LATISS_RSO-873_wep_align_monolith.ipynb"

cells = []


def _lines(src):
    """Split into nbformat `source`: every entry keeps its trailing newline but the last."""
    text = src.strip("\n")
    return [line + "\n" for line in text.split("\n")[:-1]] + [text.split("\n")[-1]]


def md(src):
    cells.append({"cell_type": "markdown", "metadata": {}, "source": _lines(src)})


def code(src):
    cells.append(
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": _lines(src),
        }
    )


# ----------------------------------------------------------------- 1. header
md(r"""
# AOS LATISS RSO-873

`latiss_wep_align` in a notebook, and towards a LATISS monolith

[RSO-873](https://rubinobs.atlassian.net/browse/RSO-873)

Notebook version of `run_wep` from
[`ts_externalscripts/.../auxtel/latiss_wep_align.py`](https://github.com/lsst-ts/ts_externalscripts/blob/develop/python/lsst/ts/externalscripts/auxtel/latiss_wep_align.py):
load two exposures with the butler → `QuickFrameMeasurementTask` → donut catalog →
`CutOutDonutsScienceSensorTask` → `CalcZernikesTask` → hexapod offsets.

The point is to see every intermediate product, because **`run_wep` does not work with a
modern ts_wep**. The summit still runs **ts_wep 15.1.0**. Three independent problems, all in
`CalcZernikesTask`, all specific to the single-donut / single-pair case that AuxTel produces:

1. **`blurClip` crashes.** `calcZernikesTask.py:545` — `sigma_clip` on a length-1 array returns
   a 0-d mask, so `np.where(blurMask)` raises
   `ValueError: Calling nonzero on 0d arrays is not allowed`.
   `doBlurClip` defaults to `True`, so `run_wep` as written **cannot** succeed.
2. **All-NaN averages.** `combineZernikes.zkClipType` defaults to `"deviation"`, but
   deviation = zk − intrinsics and LATISS has **no** intrinsic Zernike calibration
   (`calcZernikesTask.py:650-663` warns instead of raising, values stay NaN). So the only pair
   gets clipped, `used=False`, and the `average` row is all NaN. `zkClipType="opd"` fixes it.
3. **Danish silently returns garbage.** This is the real one, and it is a **LATISS-only
   regression**.

## The Danish / `zkRef` regression

`_prepDanish` (`estimation/danish.py`) builds `zkRef` from `instrument.getOffAxisCoeff()` and
passes it straight to `danish.DonutFactory`, which expects **wavefront OPD**. For AuxTel that
value is **43.5× too large**, so the model donut overfills the stamp, the mask rejects every
pixel, and `least_squares` bails at `nfev=1` with an all-zero model image and zero Jacobian.
The result is not an error — it is a plausible-looking Z4 of a few nm.

On-axis, extra-focal, `getTaskInstrument("LATISS", ...)`:

| quantity | 15.1.0 | 17.8.1 / 17.9.0 |
|---|---|---|
| `getOffAxisCoeff()[4]` | **3.409 µm** | **149.07 µm** |
| `batoid.zernike` (true OPD) | 3.42 µm | 3.42 µm |

`getOffAxisCoeff` and `batoid.zernikeTA` are **byte-identical** between the two versions. The
change is one line in `_getIntrinsicZernikesTACached`:

* **15.1.0** — `batoidModel.withLocallyShiftedOptic("Detector", [0, 0, sign*self.defocalOffset])`
  → shifts the **Detector** by 34.94 mm.
* **17.x** — `self._applyBatoidOffsets(batoidModel, defocalSign)` → shifts every optic named in
  `batoidOffsetOptic`. `policy/instruments/AuxTel.yaml` sets `batoidOffsetOptic: M2`,
  `batoidOffsetValue: 0.8e-3`, so it shifts **M2** by 0.8 mm.

Those two are equivalent in *wavefront* Z4 — that equivalence is exactly what
`Instrument.defocalOffset` back-solves with `minimize_scalar` — but **not** in *transverse
aberration*, which is what `zernikeTA` returns. The ratio is the M2→detector defocus
magnification, 34.94/0.8 = 43.7 ≈ the 43.5× observed.

**LSSTCam is unaffected**: it has no `batoidOffsetOptic`, so it takes the `_applyBatoidOffsets`
fallback branch that still shifts the Detector. That is why this went unnoticed.

## Environment

Two setup scripts, because `setup -kr` binds one directory per version:

```bash
# modern
source /sdf/data/rubin/user/scichris/WORK/aos_packages/setup_aos_wep_17.8.1_donut_viz_4.7.2_danish_1.2.0.sh
# summit reference (ts_wep only; 15.1.0 defaults to TIE and predates donut_viz/danish configs)
source /sdf/data/rubin/user/scichris/WORK/aos_packages/setup_aos_wep_15.1.0.sh
```

`ts_wep_15.1.0` is a `git worktree` at tag v15.1.0. Everything below is written to run under
**either**, so the same notebook can be executed twice and the numbers compared.
""")

code(r"""
import lsst.ts.wep

# Every number below depends on which ts_wep is set up -- print it first.
WEP_PATH = lsst.ts.wep.__file__
print("ts_wep from:", WEP_PATH)

# The 16.0.0 boundary is where CombineZernikes/intrinsics/zkClipType arrived.
IS_MODERN_WEP = "ts_wep_15" not in WEP_PATH
print("treating as modern (>=16):", IS_MODERN_WEP)
""")

code(r"""
import warnings

import astropy.units as u
import matplotlib.pyplot as plt
import numpy as np
from astropy.table import QTable
from astropy.visualization import ZScaleInterval
from lsst.daf.butler import Butler
from lsst.obs.lsst import Latiss
from lsst.pipe.tasks.quickFrameMeasurement import QuickFrameMeasurementTask

from lsst.summit.utils.bestEffort import BestEffortIsr
from lsst.ts.wep.task.calcZernikesTask import CalcZernikesTask, CalcZernikesTaskConfig
from lsst.ts.wep.task.cutOutDonutsScienceSensorTask import (
    CutOutDonutsScienceSensorTask,
    CutOutDonutsScienceSensorTaskConfig,
)
from lsst.ts.wep.task.estimateZernikesTieTask import EstimateZernikesTieTask
from lsst.ts.wep.task.generateDonutCatalogUtils import addVisitInfoToCatTable

REPO = "/repo/main"
zscale = ZScaleInterval()
""")

# ------------------------------------------------------- 2. pair selection
md(r"""
## 1. Find a CWFS pair

AuxTel CWFS sequences are `observation_type='cwfs'` with `observation_reason` alternating
`intra`/`extra`. Below lists a night so a different pair can be picked.
""")

code(r"""
butler = Butler(REPO)

DAY_OBS = 20260713
recs = sorted(
    butler.registry.queryDimensionRecords(
        "exposure",
        where=(
            "instrument='LATISS' and exposure.observation_type='cwfs' "
            f"and exposure.day_obs={DAY_OBS}"
        ),
        instrument="LATISS",
    ),
    key=lambda r: r.seq_num,
)
print(f"{len(recs)} cwfs exposures on {DAY_OBS}")
for r in recs:
    print(f"  {r.id}  seq={r.seq_num:4d}  {r.physical_filter:22s} "
          f"{r.target_name:12s} {r.exposure_time:5.1f}s  {r.observation_reason}")
""")

code(r"""
# Defaults: BLOCK-305 pair on 20260713, r band, 30s, focusZ +0.80 / -0.80 mm.
EXP_INTRA = 2026071300013
EXP_EXTRA = 2026071300014

dataId_intra = {"instrument": "LATISS", "exposure": EXP_INTRA, "detector": 0}
dataId_extra = {"instrument": "LATISS", "exposure": EXP_EXTRA, "detector": 0}
""")

# --------------------------------------------------------------- 3. ISR
md(r"""
## 2. ISR

`run_wep` uses `BestEffortIsr` (via `get_image` from `ts_observing_utilities`). Here we call it
directly.

Two arguments matter:

* `repoString="/repo/main"` — the default resolves the `LATISS` repo alias, or `/repo/embargo`
  with `embargo=True`.
* **`doWrite=False`** — the default is `True`, which tries to write `quickLookExp` back into
  the repo. Not what you want from a notebook.
""")

code(r"""
best_effort_isr = BestEffortIsr(repoString=REPO, doWrite=False)

exposure_intra = best_effort_isr.getExposure(dataId_intra)
exposure_extra = best_effort_isr.getExposure(dataId_extra)

# LATISS inverts the usual convention: cutOutDonutsScienceSensorTask.assignExtraIntraIdx
# (:268-282) treats the SMALLEST focusZ as EXTRA-focal, opposite to LSSTCam/ComCam.
# pairTask.py:130-131 encodes the same thing as `separation = -0.8`.
for label, exp in (("intra", exposure_intra), ("extra", exposure_extra)):
    vi = exp.visitInfo
    print(f"{label}: visit={vi.id}  focusZ={vi.focusZ:+.4f} mm  band={exp.filter.bandLabel}"
          f"  physical={exp.filter.physicalLabel}")
""")

code(r"""
fig, axes = plt.subplots(1, 2, figsize=(13, 6))
for ax, (label, exp) in zip(axes, (("extra", exposure_extra), ("intra", exposure_intra))):
    arr = exp.image.array
    vmin, vmax = zscale.get_limits(arr)
    ax.imshow(arr, origin="lower", vmin=vmin, vmax=vmax, cmap="gray")
    ax.set_title(f"{label}-focal  {exp.visitInfo.id}  focusZ={exp.visitInfo.focusZ:+.2f} mm")
fig.tight_layout()
""")

# ------------------------------------------------- 4. QuickFrameMeasurement
md(r"""
## 3. `QuickFrameMeasurementTask`

This is the centroid finder `run_wep` uses. `donutDiameter` comes from the `side` property of
`LatissBaseAlign` (`latiss_base_align.py:202-205`), which scales a `dz=1.5` reference size of
192 px by the actual `dz`:

```python
self._side = 192 * 1.1                                    # :161
side = int(np.ceil(self._side * self.dz / 1.5 / 2.0) * 2)  # must be even
donut_diameter = 2 * side                                  # what run_align passes
```

Note `.clone()` — the task modifies the exposure it is given (`run_wep` clones for the same
reason).
""")

code(r'''
DZ = 0.8  # mm of M2 defocus; LatissBaseAlign schema default


def get_side(_side=192 * 1.1, dz=DZ):
    """Reproduce LatissBaseAlign.side -- must be an even number."""
    return int(np.ceil(_side * dz / 1.5 / 2.0) * 2)


donut_diameter = 2 * get_side()
print("side =", get_side(), " donut_diameter =", donut_diameter)

quick_frame_measurement_task = QuickFrameMeasurementTask(
    config=QuickFrameMeasurementTask.ConfigClass()
)
result_intra = quick_frame_measurement_task.run(
    exposure_intra.clone(), donutDiameter=donut_diameter
)
result_extra = quick_frame_measurement_task.run(
    exposure_extra.clone(), donutDiameter=donut_diameter
)

# run_wep raises RuntimeError if either fails.
print("success intra/extra:", result_intra.success, result_extra.success)
for label, res in (("intra", result_intra), ("extra", result_extra)):
    print(f"{label}: brightestObjCentroid={np.round(res.brightestObjCentroid, 1)} "
          f"CofM={np.round(res.brightestObjCentroidCofM, 1)} "
          f"apFlux70={res.brightestObjApFlux70:.3e}")
''')

code(r"""
# run_wep also rejects sources too far from the boresight
# (max_distance_from_boresight=500 px). The real check uses
# lsst.ts.observing.utilities...calculate_xy_offsets against latiss_constants.boresight;
# here we just measure from the detector centre, which is close enough to sanity-check.
MAX_DISTANCE_FROM_BORESIGHT = 500.0

bbox = exposure_intra.getBBox()
centre = np.array([bbox.getWidth() / 2, bbox.getHeight() / 2])
print(f"detector centre (approx boresight): {centre}")
for label, res in (("intra", result_intra), ("extra", result_extra)):
    dr = np.hypot(*(np.array(res.brightestObjCentroid) - centre))
    flag = "OK" if dr < MAX_DISTANCE_FROM_BORESIGHT else "OUT OF BOUNDS"
    print(f"{label}: dr = {dr:6.1f} px  ({flag})")
""")

code(r"""
fig, axes = plt.subplots(1, 2, figsize=(13, 6))
for ax, (label, exp, res) in zip(
    axes,
    (("extra", exposure_extra, result_extra), ("intra", exposure_intra, result_intra)),
):
    arr = exp.image.array
    vmin, vmax = zscale.get_limits(arr)
    ax.imshow(arr, origin="lower", vmin=vmin, vmax=vmax, cmap="gray")
    cx, cy = res.brightestObjCentroidCofM
    ax.plot(cx, cy, "r+", ms=20, mew=2)
    half = donut_diameter / 2
    ax.add_patch(
        plt.Rectangle((cx - half, cy - half), donut_diameter, donut_diameter,
                      ec="red", fc="none", lw=1.5)
    )
    ax.set_title(f"{label}-focal, centroid + {donut_diameter} px stamp")
fig.tight_layout()
""")

# ------------------------------------------------------- 5. donut catalog
md(r"""
## 4. Donut catalog

`get_donut_catalog` verbatim from `latiss_wep_align.py:252-285`. Three things are load-bearing
and identical in 15.1.0 and 17.x:

* it must be an **astropy `QTable`** (not a DataFrame) with `coord_ra`/`coord_dec` in rad,
  `centroid_x`/`centroid_y` in pixels, `source_flux` in nJy;
* `meta["blend_centroid_x"] = ""` — `len("") == 0`, which is what makes `cutOutDonutsBase` skip
  the blend branch;
* `addVisitInfoToCatTable` is **mandatory** — `addVisitLevelMetadata` reads
  `meta["visit_info"]["instrument_label"]`, `visit_id`, the six `boresight_*` and
  `donut_radius`. In 17.x it also appends the `donut_id` column that `cutOutStamps` now
  requires, so always call the running version's helper rather than hand-building the table.
""")

code(r'''
def get_donut_catalog(result, exposure):
    """Donut catalog from a QuickFrameMeasurement result.

    Verbatim from latiss_wep_align.get_donut_catalog (:252-285).
    """
    wcs = exposure.getWcs()
    ra, dec = wcs.pixelToSkyArray(
        result.brightestObjCentroidCofM[0],
        result.brightestObjCentroidCofM[1],
        degrees=False,
    )
    donut_catalog = QTable()
    donut_catalog["coord_ra"] = ra * u.rad
    donut_catalog["coord_dec"] = dec * u.rad
    donut_catalog["centroid_x"] = [result.brightestObjCentroidCofM[0]] * u.pixel
    donut_catalog["centroid_y"] = [result.brightestObjCentroidCofM[1]] * u.pixel
    donut_catalog["source_flux"] = [result.brightestObjApFlux70] * u.nJy
    donut_catalog.meta["blend_centroid_x"] = ""
    donut_catalog.meta["blend_centroid_y"] = ""
    donut_catalog.sort("source_flux", reverse=True)
    donut_catalog = addVisitInfoToCatTable(exposure, donut_catalog)

    return donut_catalog


donut_catalog_intra = get_donut_catalog(result_intra, exposure_intra.clone())
donut_catalog_extra = get_donut_catalog(result_extra, exposure_extra.clone())

donut_catalog_intra
''')

code(r"""
print("columns:", donut_catalog_intra.colnames)
# donut_id is added by addVisitInfoToCatTable in 17.x, absent in 15.1.0.
print("has donut_id:", "donut_id" in donut_catalog_intra.colnames)
vi_meta = donut_catalog_intra.meta["visit_info"]
for key in ("visit_id", "instrument_label", "focus_z", "donut_radius", "boresight_rot_angle"):
    print(f"  visit_info[{key!r}] = {vi_meta[key]}")
""")

# ----------------------------------------------------------- 6. cutouts
md(r"""
## 5. Cut out the donut stamps

`opticalModel = "onAxis"` is **required** for AuxTel — the default is `"offAxis"`, and there is
no off-axis batoid fit for AuxTel. `run` takes `[extra, intra]` in that order and figures out
which is which from `focusZ`; the signature is identical in both ts_wep versions.
""")

code(r"""
cut_out_config = CutOutDonutsScienceSensorTaskConfig()
cut_out_config.donutStampSize = donut_diameter
cut_out_config.opticalModel = "onAxis"
cut_out_config.initialCutoutPadding = 40
cut_out_task = CutOutDonutsScienceSensorTask(config=cut_out_config)

camera = Latiss.getCamera()

cut_out_output = cut_out_task.run(
    [exposure_extra, exposure_intra],
    [donut_catalog_extra, donut_catalog_intra],
    camera,
)
print("n stamps extra/intra:", len(cut_out_output.donutStampsExtra),
      len(cut_out_output.donutStampsIntra))

# Confirm the extra/intra assignment came out right despite the inverted LATISS convention.
focusZ = (exposure_extra.visitInfo.focusZ, exposure_intra.visitInfo.focusZ)
extra_idx, intra_idx = cut_out_task.assignExtraIntraIdx(*focusZ, camera.getName())
print(f"focusZ passed as (extra, intra) = {np.round(focusZ, 3)}"
      f" -> extraExpIdx={extra_idx}, intraExpIdx={intra_idx} (expect 0, 1)")
""")

code(r"""
stamp_extra = cut_out_output.donutStampsExtra[0]
stamp_intra = cut_out_output.donutStampsIntra[0]

fig, axes = plt.subplots(1, 2, figsize=(10, 5))
for ax, (label, stamp) in zip(axes, (("extra", stamp_extra), ("intra", stamp_intra))):
    ax.imshow(stamp.stamp_im.image.array, origin="lower")
    ax.set_title(f"{label}  {stamp.stamp_im.image.array.shape}")
fig.tight_layout()

print("defocalType:", stamp_extra.wep_im.defocalType, stamp_intra.wep_im.defocalType)
print("fieldAngle (deg):", np.round(stamp_extra.wep_im.fieldAngle, 5))
print("bandLabel:", stamp_extra.wep_im.bandLabel)
""")

# --------------------------------------------- 6. the two regressions, measured
md(r"""
## 6. What breaks Danish for AuxTel in modern ts_wep

Two independent problems, both found by running the identical chain under 15.1.0 and 17.x.

### 6a. `zkRef` is transverse aberration, not wavefront OPD

`_prepDanish` builds `zkRef` from `instrument.getOffAxisCoeff()` and hands it to
`danish.DonutFactory`, which expects **wavefront OPD**.

Between v15.1.0 and v16.7.0 `_getIntrinsicZernikesTACached` changed from

```python
batoidModel = batoidModel.withLocallyShiftedOptic("Detector", offset)          # 15.1.0
batoidModel = self._applyBatoidOffsets(batoidModel, defocalSign)               # 16.7+
```

where the latter shifts every optic in `batoidOffsetOptic`. `AuxTel.yaml` sets
`batoidOffsetOptic: M2`, `batoidOffsetValue: 0.8e-3`, so AuxTel now shifts **M2 by 0.8 mm**
instead of the **detector by 34.94 mm**.

That substitution is physically *more* correct, and the two are **wavefront-equivalent** —
which is exactly the equivalence `Instrument.defocalOffset` back-solves with
`minimize_scalar`:

| shift | OPD Z4 |
|---|---|
| Detector, 34.94 mm | 3432.09 nm |
| M2, 0.8 mm | 3431.98 nm |

*0.003% apart.* So the moved optic is **not** the bug — `DonutFactory` genuinely does not
care what moved. The bug is the **quantity**: `zernikeTA` returns *transverse aberration*,
which unlike OPD is **not** invariant under that substitution — it scales by the
M2 → detector magnification, 34.94/0.8 = 43.7:

| | 15.1.0 | 17.x |
|---|---|---|
| `getOffAxisCoeff()[4]` | **3.409 µm** | **149.07 µm** |

So the model donut overfills the stamp, the mask rejects every pixel, and `least_squares`
returns at `nfev=1` with an all-zero model image (it prints as a flat frame).

**LSSTCam is unaffected** — not because it lacks `batoidOffsetOptic` (`LsstCam.yaml` sets it
to `Detector`), but because its `batoidOffsetValue` (1.5 mm) *equals* its `defocalOffset` and
the shifted optic is the camera itself, so TA/OPD = 1.01 and the change was a no-op:

| instrument | offset optic | `defocalOffset` | TA Z4 | OPD Z4 | ratio |
|---|---|---|---|---|---|
| LSSTCam | `LSSTCamera` | 1.500 mm | 23.75 µm | 23.61 µm | 1.01 |
| LSSTComCam | `ComCam` | 1.500 mm | 23.74 µm | 23.56 µm | 1.01 |
| LATISS | `M2` | 34.950 mm | 149.07 µm | 3.43 µm | **43.43** |

### 6b. The cutout task no longer peak-normalizes, and gives different pixels

Same exposure, same `CutOutDonutsScienceSensorTaskConfig`, same QFM centroid:

| ts_wep | stamp peak (extra / intra) |
|---|---|
| 15.1.0 | 1.00 / 1.00 |
| 17.8.1 | 87073 / 55578 |

Danish's `sky_levels`/`fluxes` are scale-dependent, so the raw-ADU stamps give a background
variance ~10¹⁰ off. Worse, the stamps are not merely rescaled — normalizing each to its own
peak, they correlate only **0.69 (extra) / 0.54 (intra)**, and 17.x's donuts cover ~1.7×
more pixels. `cutOutDonutsBase.py` is functionally identical between the versions, and
`createTemplateForDetector` returns a byte-identical template, so the likely culprit is in
`imageMapper.py`: 17.x inserted `defocalType` as **positional arg 3** of
`getIntrinsicZernikes`, and the mask call sites moved from positional `image.bandLabel` to
keyword `band=image.bandLabel`.

This is why the 15.1.0 Zernikes are not a bit-exact oracle: the fit and the *pixels* both
changed. It is **not**, however, a reason to hand-roll the cutout — see §7a, where a plain
box around the centroid turns out measurably worse because it does not centre the donut.
""")

code(r"""
# --- 6a: measure the TA vs OPD discrepancy for this instrument
import batoid

from lsst.ts.wep.utils import DefocalType, getTaskInstrument
from lsst.ts.wep.utils.enumUtils import BandLabel

inst = getTaskInstrument("LATISS", stamp_extra.detector_name, None)
noll = np.arange(4, 23)

ta = inst.getOffAxisCoeff(
    0.0, 0.0, DefocalType.Extra, BandLabel.REF,
    nollIndicesModel=np.arange(0, 79), nollIndicesIntr=noll,
)
tel = inst.getBatoidModel(BandLabel.REF)
wl = inst.wavelength[BandLabel.REF]


def opd_z4(optic, value):
    shifted = tel.withLocallyShiftedOptic(optic, [0, 0, value])
    return batoid.zernike(
        shifted, 0.0, 0.0, wavelength=wl, nx=255, eps=inst.obscuration, jmax=12
    )[4] * wl


z4_det = opd_z4("Detector", inst.defocalOffset)
z4_m2 = opd_z4(inst.batoidOffsetOptic, inst.batoidOffsetValue)

print(f"batoidOffsetOptic = {inst.batoidOffsetOptic}, value = {inst.batoidOffsetValue}")
print(f"defocalOffset     = {inst.defocalOffset*1e3:.3f} mm  "
      f"(= {inst.defocalOffset/inst.batoidOffsetValue:.1f} x the M2 shift)")
print()
print(f"getOffAxisCoeff (TA)   Z4 = {ta[4]*1e6:9.2f} um   <- what _prepDanish passes")
print(f"batoid.zernike  (OPD)  Z4 = {z4_det*1e6:9.3f} um   <- what danish wants")
print(f"                    ratio = {ta[4]/z4_det:9.2f} x")
print()
print("the two shifts are wavefront-equivalent:")
print(f"  Detector {inst.defocalOffset*1e3:6.2f} mm -> OPD Z4 = {z4_det*1e9:8.2f} nm")
print(f"  {inst.batoidOffsetOptic:8s} {inst.batoidOffsetValue*1e3:5.2f} mm -> "
      f"OPD Z4 = {z4_m2*1e9:8.2f} nm")
print(f"  differ by {100*abs(z4_m2-z4_det)/abs(z4_det):.4f} %")
""")

code(r"""
# --- 6b: what scale do the stamps come out at?
print(f"stamp peak: extra={np.nanmax(stamp_extra.wep_im.image):.6g}  "
      f"intra={np.nanmax(stamp_intra.wep_im.image):.6g}")
print("15.1.0 gave ~1.0 (peak-normalized); 17.x gives raw ADU (~1e5).")
print(f"exposure peak for reference: {np.nanmax(exposure_extra.image.array):.6g}")
""")

# ------------------------------------- 7. the monolith fit + visual diagnostics
md(r"""
## 7. The monolith fit, and looking at it

`fit_latiss_danish` does the two fixes (OPD `zkRef`, peak-normalized stamps) plus two
practical details: `pack_params` rather than a hand-built `x0` (the danish v1.0 rename added
per-donut `fluxes`, so the vector changed shape), and a band → `refBand` fallback because
`AuxTel.yaml` declares a scalar `wavelength` and so `instrument.wavelength[LSST_R]` raises.

The plots follow `donut_viz`'s `PlotDonutFitsTask` (`plot_aos_task.py`): image / model /
residual per side of focus, greyscale on `vmin=-vmax/10`, residual in `bwr` on `±vmax/3`,
annotated with the fitted blur and the total residual.
""")

code(r"""
import sys

# Prototype lives on the ts_wep tickets/RSO-873 branch; import from there so the
# notebook and the eventual task never diverge.
sys.path.insert(
    0, "/sdf/data/rubin/user/scichris/WORK/aos_packages/ts_wep/python/lsst/ts/wep/task"
)
from latissMonolith import (  # noqa: E402
    cut_and_evaluate_stamp,
    donut_mask,
    fit_latiss_danish,
    fit_latiss_danish_arrays,
)

fit = fit_latiss_danish(stamp_extra, stamp_intra, inst, noll_indices=noll)

print(f"success={fit['success']} status={fit['status']} nfev={fit['nfev']} "
      f"cost={fit['cost']:.5g}")
print(f"fwhm={fit['fwhm']:.3f} arcsec   dxs={np.round(fit['dxs'],3)} "
      f"dys={np.round(fit['dys'],3)}")
print()
print("Zernikes (nm), zkSum = zkFit + mean(zkStart):")
for j in (4, 7, 8, 11):
    print(f"  Z{j:<3d} {fit['zernikes_nm'][j]:9.1f}")
print()
print("reference, ts_wep 15.1.0 + pre-1.0 danish, same pair:")
print("  Z4  -467.9   Z7  -255.5   Z8   167.2     cost 3749  fwhm 2.51")
""")

code(r'''
def plot_donut_fits(images, models, labels=("extra", "intra"), suptitle=None):
    """image / model / residual per side of focus, a la donut_viz PlotDonutFitsTask."""
    fig, axs = plt.subplots(len(images), 3, figsize=(11, 4 * len(images)))
    axs = np.atleast_2d(axs)
    for row, (img, mod, label) in enumerate(zip(images, models, labels)):
        vmax = np.nanmax(img)
        res = img - mod
        tot_res = float(np.nansum(np.abs(res)) / np.nansum(np.abs(img)))
        axs[row, 0].imshow(img, origin="lower", cmap="gray", vmin=-vmax / 10, vmax=vmax)
        axs[row, 0].set_title(f"{label}: data")
        axs[row, 1].imshow(mod, origin="lower", cmap="gray", vmin=-vmax / 10, vmax=vmax)
        axs[row, 1].set_title(f"{label}: danish model")
        axs[row, 2].imshow(res, origin="lower", cmap="bwr", vmin=-vmax / 3, vmax=vmax / 3)
        axs[row, 2].set_title(f"{label}: residual")
        axs[row, 2].text(0.05, 0.05, f"res: {tot_res:5.3f}",
                         transform=axs[row, 2].transAxes, fontsize="small", va="bottom")
        for ax in axs[row]:
            ax.set_xticks([])
            ax.set_yticks([])
    if suptitle:
        fig.suptitle(suptitle)
    fig.tight_layout()
    return fig


if fit["model_images"] is not None:
    plot_donut_fits(fit["images"], fit["model_images"],
                    suptitle=f"Danish fit, cost={fit['cost']:.4g}, fwhm={fit['fwhm']:.2f}\"")
else:
    print("no model images returned -- danish version did not expose them via unpack_params")
''')

code(r"""
# The masks danish/ts_wep use, for inspection: if the donut overfills these, the fit is dead.
fig, axs = plt.subplots(2, 3, figsize=(11, 8))
for row, (label, stamp) in enumerate((("extra", stamp_extra), ("intra", stamp_intra))):
    wim = stamp.wep_im
    npix = fit["images"][row].shape[0]
    m_src, m_bkg = donut_mask(
        inst, wim.fieldAngle, wim.defocalType.value, npix,
        noll_indices=noll, band="ref",
    )
    axs[row, 0].imshow(fit["images"][row], origin="lower", cmap="gray")
    axs[row, 0].set_title(f"{label}: stamp")
    if m_src is None:
        axs[row, 1].text(0.5, 0.5, "mask unavailable", ha="center")
        axs[row, 2].axis("off")
    else:
        axs[row, 1].imshow(m_src, origin="lower")
        axs[row, 1].set_title(f"{label}: source mask")
        axs[row, 2].imshow(m_bkg, origin="lower")
        axs[row, 2].set_title(f"{label}: background mask")
    for ax in axs[row]:
        ax.set_xticks([])
        ax.set_yticks([])
fig.tight_layout()
""")

md(r"""
### 7a. Box cutout — tried, and NOT adopted

`CutOutDonutsScienceSensorTask` (§7) is what we keep. Verified on this pair: its donuts are
well centred, the §7 residuals are small and the model matches the stamp in shape.

The plain box below is kept only as a documented dead end, and as a fallback if the cutout
task regresses again. It cuts a fixed box around the QuickFrameMeasurement centroid — what
`latiss_wep_align` did before the cutout task existed. **It is measurably worse**: the donut
is not centred in the box, so `inner_frac` comes out ~1.1 (essentially all the flux inside
what the geometry calls the pupil hole, which should be near zero) and Z8 collapses from
+86 nm to +2 nm.

Note the box fit reports a *lower* `cost` (2002 vs 2591) while being the worse cutout — cost
is not comparable across different pixels. Judge by the residual images and `inner_frac`.

So the version-instability of §6b is real but is **not** a reason to hand-roll the cutout;
it only means the *reference* Zernikes from 15.1.0 are not a bit-exact oracle.

Note the stamp size: `donutStampSize` defaults to **160** px, which is LSSTCam-sized and
would clip an AuxTel donut. AuxTel's physical donut is **194 px** across
(`instrument.donutRadius` = 97.1 px) versus LSSTCam's 133 px. ts_wep's LATISS test pipelines
use 200; `latiss_wep_align` derives `2 * side` = **228** from `dz = 0.8`. Set it explicitly.
""")

code(r"""
print(f"AuxTel donutRadius = {inst.donutRadius:.1f} px -> donut diameter "
      f"{2*inst.donutRadius:.1f} px")
print(f"donut_diameter used here (2 * side, from dz={DZ}): {donut_diameter}")

cut_e = cut_and_evaluate_stamp(
    exposure_extra, result_extra.brightestObjCentroidCofM, donut_diameter, instrument=inst
)
cut_i = cut_and_evaluate_stamp(
    exposure_intra, result_intra.brightestObjCentroidCofM, donut_diameter, instrument=inst
)
box_extra, box_intra = cut_e["stamp"], cut_i["stamp"]
for lbl, c in (("extra", cut_e), ("intra", cut_i)):
    print(f"{lbl}: {c['stamp'].shape} peak={c['peak']:.4g} snr={c['snr']:.1f} "
          f"inner_frac={c['inner_frac']:.3f} outer_frac={c['outer_frac']:.3f} "
          f"sat={c['saturated']}")
print()
print("NB inner_frac should be SMALL (flux inside the pupil hole). Values ~1 mean the")
print("annulus geometry does not match the actual donut -- inspect the images below")
print("before trusting any Zernikes.")

fig, axs = plt.subplots(1, 2, figsize=(9, 4.5))
for ax, (label, arr) in zip(axs, (("extra", box_extra), ("intra", box_intra))):
    ax.imshow(arr, origin="lower", cmap="gray")
    ax.set_title(f"{label}: plain box cutout")
    ax.set_xticks([])
    ax.set_yticks([])
fig.tight_layout()
""")

code(r"""
fit_box = fit_latiss_danish_arrays(
    box_extra, box_intra,
    stamp_extra.wep_im.fieldAngle, stamp_intra.wep_im.fieldAngle,
    inst, band="ref", noll_indices=noll,
)
print(f"box cutout: nfev={fit_box['nfev']} cost={fit_box['cost']:.5g} "
      f"fwhm={fit_box['fwhm']:.3f}")
for j in (4, 7, 8):
    print(f"  Z{j:<3d} {fit_box['zernikes_nm'][j]:9.1f} nm")

if fit_box["model_images"] is not None:
    plot_donut_fits(fit_box["images"], fit_box["model_images"],
                    suptitle="Danish fit on plain box cutouts")
""")

md(r"""
### 7b. Comparison across the three paths

Filled in from the runs above. The 15.1.0 column needs a separate kernel
(`setup_aos_wep_15.1.0.sh` **plus** `PYTHONPATH=.../danish_pre1.0_df9f5fc`, because 15.1.0
calls `danish.MultiDonutModel`, removed in danish ≥ v1.0.0).

| path | Z4 | Z7 | Z8 | cost | fwhm | verdict |
|---|---|---|---|---|---|---|
| 15.1.0 + pre-1.0 danish (summit) | −467.9 | −255.5 | +167.2 | 3749 | 2.51″ | reference |
| 17.8.1 stock Danish | −2.8 | +8.5 | 0.0 | 3.3e7 | 0.7 (unfit) | **dead**: `nfev=1` |
| 17.8.1 + fix, ts_wep cutout | −303 | +30 | +80 | 2572 | 1.67″ | alive |
| 17.8.1 + fix, **15.1.0 stamps** | −416 | −159 | +162 | 4598 | 2.32″ | closest |
| 17.8.1 + fix, box cutout (§7a) | −360 | +38 | +2 | 2002 | 1.46″ | rejected, donut off-centre |

The last row is the informative one: feeding the *same* fix the *same pixels* the reference
used moves Z4 from −303 to −416 (reference −468) and Z8 to within 6 nm. So the fit logic is
substantially right and the residual disagreement is dominated by the stamps, not the fit.

A caution on reading the `cost` column: it is **not** comparable across rows with different
stamps or different normalizations. The 17.8.1-cutout row has the lowest cost while being
furthest from the reference — it is fitting different pixels. Judge these by the residual
images above, not by cost.
""")

md(r"""
## 8. Hexapod offsets

Reproduces `LatissBaseAlign.calculate_results` (`latiss_base_align.py:299`). Importing
`LatissWEPAlign` is not practical in a notebook — it is a `salobj.BaseScript` needing a SAL
index, and `configure` builds ATCS/LATISS remotes — but the arithmetic is four lines.

Three things to get right:

* `zern = [-Z8, Z7, Z4]` in **nm**. The minus on Z8 lives in `run_align`
  (`latiss_wep_align.py:137-141`), *not* in `calculate_results`.
* `np.matmul(zern, M)` is **row-vector × matrix**. `M @ zern` gives different, wrong numbers —
  the matrix is not symmetric.
* There is **no** nm → mm conversion anywhere; it is baked into `matrix_sensitivity`
  (documented as "mm of hexapod motion for nm of wfs", see
  [SITCOMTN-072](https://sitcomtn-072.lsst.io/)).

`angle = 90 - boresight_angle` normally comes from the ATCS; set it by hand here.
""")

code(r'''
# Constants from latiss_base_align.py (:114-118, :132, :144) and
# ts_observatory_control atcs_constants.py (:26-33).
MATRIX_SENSITIVITY = np.array([
    [0.00688945, -0.00008867, 0.00004848],
    [0.00031787, -0.00650340, -0.00007319],
    [-0.00000782, -0.00003023, 0.00025634],
])
HEXAPOD_OFFSET_SCALE = np.array([  # arcsec/mm; x-axis is elevation
    [52.459, 0.0, 0.0],
    [0.0, 50.468, 0.0],
    [0.0, 0.0, 0.0],
])
GAIN = np.array([0.5, 0.5, 0.9])
CAMERA_ROTATION_ANGLE = 0.0


def matrix_rotation(angle_deg):
    a = np.radians(angle_deg)
    return np.array([
        [np.cos(a), -np.sin(a), 0.0],
        [np.sin(a), np.cos(a), 0.0],
        [0.0, 0.0, 1.0],
    ])


def calculate_results(zern, angle_deg=0.0, gain=GAIN):
    # zern = [coma-X, coma-Y, focus] in nm -> hexapod (mm) and telescope (arcsec)
    rot_zern = np.matmul(zern, matrix_rotation(angle_deg + CAMERA_ROTATION_ANGLE))
    hexapod_offset = np.matmul(rot_zern, MATRIX_SENSITIVITY) * gain
    tel_offset = np.matmul(hexapod_offset, HEXAPOD_OFFSET_SCALE)
    return rot_zern, hexapod_offset, tel_offset


BORESIGHT_ANGLE = 0.0  # from atcs.get_bore_sight_angle(); angle = 90 - that
angle = 90.0 - BORESIGHT_ANGLE

zk_nm = fit["zernikes_nm"]
zern = [-zk_nm[8], zk_nm[7], zk_nm[4]]  # note the sign flip on Z8
rot_zern, hexapod_offset, tel_offset = calculate_results(zern, angle_deg=angle)

zern_str = (len(zern) * "{:0.1f}, ").format(*zern)
rot_str = (len(rot_zern) * "{:0.1f}, ").format(*rot_zern)
hex_str = (len(hexapod_offset) * "{:0.3f}, ").format(*hexapod_offset)
tel_str = (len(tel_offset) * "{:0.1f}, ").format(*tel_offset)
print(f"""==============================
Measured [coma-X, coma-Y, focus] zernike coefficients [nm]: [{zern_str}]
De-rotated [coma-X, coma-Y, focus]  zernike coefficients [nm]: [{rot_str}]
Gain: {GAIN}
Hexapod [x, y, z] offsets [mm] : [{hex_str}]
Telescope offsets [arcsec]: [{tel_str}]
==============================""")
''')

code(r"""
# Cross-check the arithmetic against the ts_externalscripts unit test
# (test_latiss_wep_align.py:690-697): meas [-60.1, 30.8, 50.7] nm should de-rotate
# to [-52.6, 42.4, 50.7] nm. That fixes the angle convention independent of our data.
rz, hx, tl = calculate_results([-60.1, 30.8, 50.7], angle_deg=12.0)
print("de-rotated:", np.round(rz, 1), " expected ~[-52.6, 42.4, 50.7]")
print("hexapod   :", np.round(hx, 3))
print("NB the test's commented hexapod values predate the gain vector, so they do not match.")
""")

# ------------------------------------------- 10. validation across pairs + sweeps
md(r"""
## 9. Validation: many pairs, and the focus sweeps

A single pair cannot tell a good fit from a lucky one, and the pair used above
(2026071300013/14) turns out to be **unusually defocused** — so it is the worst possible
yardstick. Two independent checks, both precomputed into CSVs by scripts in this repo
(each pair costs ~1 min, so the notebook plots rather than re-runs):

* `RSO-873_run_many_pairs.py` → `RSO-873_many_pairs.csv` — the monolith fit over 13 CWFS
  pairs spread over 20260602-20260710.
* `RSO-873_run_focus_sweep.py` → `RSO-873_focus_sweep.csv` — the classical focus curve from
  the real focusZ sweeps.

**Finding the sweeps is itself a trap.** They are `observation_type='focus'`,
`science_program='BLOCK-T743'`, but you must (a) exclude dispersed frames — any
`physical_filter` containing `holo4_003`, `300lpmm`, `prism` or `notch` is a spectrum, direct
imaging is `empty~empty` or `empty~SDSSy_65mm` — and (b) check `focusZ` actually varies, because
most `focus` sequences are *repeats at fixed focus*. Of seven candidate runs, only four are
real sweeps: 20260624 seq 602-616 and 692-706 (span 0.2 mm), and 20260630 seq 96-110 and
186-200 (span 0.4 mm).

Note the sweep frames are **single** exposures, not intra/extra pairs, so Danish cannot be run
on them, and at +/-0.2 mm (a quarter of the nominal 0.8 mm) they are near-focus PSFs rather
than the ~97 px CWFS donuts. They constrain the *geometry* and best focus, not the wavefront.
""")

code(r"""
import csv
from pathlib import Path

REPO_DIR = Path("/sdf/data/rubin/user/scichris/WORK/AOS")


def read_csv(name):
    path = REPO_DIR / name
    if not path.exists():
        print(f"{name} missing -- run the matching RSO-873_run_*.py first")
        return []
    with open(path) as f:
        return list(csv.DictReader(f))


pairs = read_csv("RSO-873_many_pairs.csv")
sweeps = read_csv("RSO-873_focus_sweep.csv")
print(f"{len(pairs)} pair fits, {len(sweeps)} sweep frames")
""")

code(r"""
# --- Zernikes across pairs. Cut on fractional residual, not on cost: cost is not
# comparable between pairs with different stamps or normalizations.
RES_CUT = 0.40

z4 = np.array([float(r["Z4"]) for r in pairs])
z7 = np.array([float(r["Z7"]) for r in pairs])
z8 = np.array([float(r["Z8"]) for r in pairs])
res = np.array([float(r["res_extra"]) for r in pairs])
fwhm_fit = np.array([float(r["fwhm"]) for r in pairs])
good = res < RES_CUT

print(f"{good.sum()}/{len(pairs)} pairs with res_extra < {RES_CUT}")
for label, arr in (("Z4", z4), ("Z7", z7), ("Z8", z8), ("fwhm", fwhm_fit)):
    print(f"  {label:5s} median={np.median(arr[good]):8.1f}  std={np.std(arr[good]):7.1f}")

fig, axs = plt.subplots(1, 3, figsize=(14, 4))
x = np.arange(len(pairs))
for ax, (label, arr) in zip(axs, (("Z4", z4), ("Z7", z7), ("Z8", z8))):
    ax.scatter(x[good], arr[good], c="C0", label=f"res < {RES_CUT}")
    ax.scatter(x[~good], arr[~good], c="C3", marker="x", s=70, label="rejected")
    ax.axhline(np.median(arr[good]), ls="--", c="k",
               label=f"median {np.median(arr[good]):.0f} nm")
    ax.axhline(0, ls=":", c="grey", lw=0.8)
    ax.set_xlabel("pair index")
    ax.set_ylabel(f"{label} (nm)")
    ax.set_title(label)
    ax.legend(fontsize="small")
fig.suptitle("Monolith Danish fit across 13 LATISS CWFS pairs")
fig.tight_layout()
""")

code(r"""
# --- Does Z4 respond to applied focus? The intra/extra MIDPOINT is nonzero when a net
# focus error was applied during the alignment iteration loop, so it acts as a proxy for
# real defocus. (A proxy only -- the focus sweeps below are the cleaner handle.)
mid = np.array([
    (float(r["focusz_intra"]) + float(r["focusz_extra"])) / 2.0 for r in pairs
])

fig, ax = plt.subplots(figsize=(6.5, 5))
ax.scatter(mid[good], z4[good], c="C0")
ax.scatter(mid[~good], z4[~good], c="C3", marker="x", s=70, label="rejected")
if good.sum() > 2:
    slope, icept = np.polyfit(mid[good], z4[good], 1)
    corr = np.corrcoef(mid[good], z4[good])[0, 1]
    xs = np.linspace(mid[good].min(), mid[good].max(), 10)
    ax.plot(xs, slope * xs + icept, "k--",
            label=f"slope {slope:.0f} nm/mm, r={corr:.2f}")
    print(f"corr(midpoint focusZ, Z4) = {corr:.3f}, slope {slope:.0f} nm/mm")
ax.axhline(0, ls=":", c="grey", lw=0.8)
ax.axvline(0, ls=":", c="grey", lw=0.8)
ax.set_xlabel("midpoint of (focusZ_intra, focusZ_extra)  [mm] = net applied focus error")
ax.set_ylabel("fitted Z4 (nm)")
ax.set_title("Z4 tracks applied focus offset")
ax.legend()
fig.tight_layout()
""")

code(r"""
# --- Focus curves. Defocus adds in quadrature with the seeing disk, so rms^2 is a
# parabola in focusZ; the vertex is best focus and sqrt(quadratic coeff) is the slope.
by_sweep = {}
for r in sweeps:
    by_sweep.setdefault(r["sweep"], []).append(r)

# Geometric expectation, same numbers that drive the zkRef bug in section 6a:
# blur diameter = focusZ * magnification / f-number; a uniform disk has rms = D/4.
mag = inst.defocalOffset / inst.batoidOffsetValue
f_ratio = inst.focalLength / (2 * inst.radius)
pred_rms_per_mm = (mag / f_ratio * 1e-3 / inst.pixelSize) / 4.0
print(f"geometry: {mag:.1f}x magnification, f/{f_ratio:.1f} "
      f"-> {pred_rms_per_mm:.1f} px rms per mm of focusZ")

real = {}
for name, rs in by_sweep.items():
    fz = np.array([float(x["focus_z"]) for x in rs])
    if fz.max() - fz.min() > 0.05:  # else it is a repeat sequence, not a sweep
        real[name] = rs
    else:
        print(f"  skipping {name}: focusZ span {fz.max()-fz.min():.4f} mm (repeats)")

fig, axs = plt.subplots(1, len(real), figsize=(4.2 * len(real), 4), squeeze=False)
Z4_PER_MM = 4231.0  # nm of wavefront Z4 per mm of M2 motion, from batoid
for ax, (name, rs) in zip(axs[0], sorted(real.items())):
    fz = np.array([float(x["focus_z"]) for x in rs])
    rms = np.array([float(x["rms_px"]) for x in rs])
    order = np.argsort(fz)
    ax.plot(fz[order], rms[order], "o-", ms=4)
    a, b, c = np.polyfit(fz, rms**2, 2)
    fz_best = -b / (2 * a)
    k = np.sqrt(a)
    xs = np.linspace(fz.min(), fz.max(), 100)
    ax.plot(xs, np.sqrt(np.clip(a * xs**2 + b * xs + c, 0, None)), "k--", lw=1)
    ax.axvline(fz_best, ls=":", c="C3")
    title = " | ".join([
        name,
        f"best {fz_best:+.4f} mm -> Z4 {fz_best*Z4_PER_MM:+.0f} nm",
        f"k={k:.0f} px/mm ({k/pred_rms_per_mm:.2f} x geom)",
    ])
    ax.set_title(title, fontsize="small")
    ax.set_xlabel("focusZ (mm)")
    ax.set_ylabel("image rms (px)")
fig.suptitle("LATISS focus curves — the vertex gives best focus, the width gives the geometry")
fig.tight_layout()
""")

md(r"""
### 9a. What the validation shows

**Across 13 CWFS pairs** (12 passing the residual cut): median Z4 = **−42 nm**,
Z7 = **+1 nm**, Z8 = **+22 nm** — a well-aligned telescope, which is the expected answer.
Fitted fwhm 1.5-2.8″, median 2.0″, i.e. plausible AuxTel seeing: an independent sign that the
forward model is at the right physical scale, since fwhm is fitted, not imposed.

**The notebook's own pair is an outlier.** Its Z4 = −295 nm sits far from the −42 nm median.
So the failure to reproduce 15.1.0's −467.9 nm *on that pair* is not evidence the fit is wrong.

**Z4 responds to applied focus**: correlating against the intra/extra midpoint gives r ≈ −0.84.
The slope from that proxy came out ~1.4× the batoid expectation, which the sweeps below
suggest is an artifact of the proxy rather than of the fit.

**The focus sweeps confirm the geometry on sky.** The two well-constrained sweeps (20260630,
0.4 mm span, 1.9″ seeing) give measured/geometric slope ratios of **0.94** and **0.89** —
essentially the predicted 60.7 px rms/mm from the 43.7× magnification and f/18. The two
20260624 sweeps (half the span, 2.6-3.1″ seeing) give 0.55/0.70, expected when the defocus
term barely exceeds the seeing disk and the parabola is poorly constrained.

**And they cross-check the absolute Z4 scale.** Best focus lands at −0.013 to +0.015 mm,
i.e. Z4 of **−55 to +64 nm** at 4231 nm per mm of M2 motion — bracketing the −42 nm median
from the CWFS pairs, derived from image sizes alone with no wavefront fit involved.

Taken together: the fit is consistent pair-to-pair, physically scaled, responds correctly to
focus, and agrees with an independent geometric measurement. That is the basis for packaging
it as a pipetask.
""")

# ------------------------------------- 9b. version comparison on identical stamps
md(r"""
### 9b. 15.1.0 vs the fix, on identical stamps

The cleanest version test: cut the stamps **once**, then fit the same arrays under both
stacks. That holds the cutout fixed (§6b showed it changed between versions) so only the
algorithm varies. Two scripts do it:

* `RSO-873_dump_stamps.py` — cuts and pickles stamps. Takes a mode argument: `spread`
  samples the whole date range, `first` takes the opening pairs of each night, which is
  where large aberrations live (telescope not yet aligned).
* `RSO-873_fit_stamps.py` — fits a pickle under whichever ts_wep is set up. Auto-detects
  the version: under 15.1.0 it runs that version's own `DanishAlgorithm` (the summit
  reference); under 17.x it runs both the stock path and the monolith fix.

Results on 12 spread pairs, identical stamps:

| path | outcome |
|---|---|
| 15.1.0 native (summit) | works on all 12 |
| 17.8.1 **stock** | **fails on 11/12** — NaN, "Non-positive image flux" |
| 17.8.1 + monolith fix | works on all 12 |

Worth noting the stock failure is harder than "a degenerate fit": on raw-ADU stamps
`_prepDanish` subtracts a background that overshoots, total flux goes negative, and danish
bails to NaN. The Z4 ≈ −2.8 nm seen in §7 was the *lucky* case.

15.1.0 vs the fix, same pixels: **Z4 corr 0.953, Z8 corr 0.953**, median |diff| 20-37 nm.
So given the same stamp the two algorithms agree — which is the result that matters here.
""")

code(r"""
fit15 = read_csv("RSO-873_fit_15_first.csv")
fit17 = read_csv("RSO-873_fit_17_first.csv")

if fit15 and fit17:
    a = {r["intra"]: r for r in fit15}
    b = {r["intra"]: r for r in fit17}
    keys = [k for k in a if k in b]
    z4a = np.array([float(a[k]["Z4"]) for k in keys])
    z4b = np.array([float(b[k]["Z4"]) for k in keys])
    cost17 = np.array([float(b[k]["chi_square"]) for k in keys])
    ok = np.isfinite(z4a) & np.isfinite(z4b) & np.isfinite(cost17)
    dz4 = np.abs(z4a - z4b)

    print(f"{len(keys)} first-of-night pairs, {ok.sum()} finite both sides")
    print(f"corr(log cost, |dZ4|) = {np.corrcoef(np.log(cost17[ok]), dz4[ok])[0,1]:.3f}")
    split = np.median(cost17[ok])
    lo = cost17[ok] < split
    print(f"  low-cost  half: Z4 corr {np.corrcoef(z4a[ok][lo], z4b[ok][lo])[0,1]:.3f}, "
          f"median |dZ4| {np.median(dz4[ok][lo]):.1f} nm")
    print(f"  high-cost half: Z4 corr {np.corrcoef(z4a[ok][~lo], z4b[ok][~lo])[0,1]:.3f}, "
          f"median |dZ4| {np.median(dz4[ok][~lo]):.1f} nm")

    fig, axs = plt.subplots(1, 2, figsize=(12, 5))
    lim = max(np.abs(z4a[ok]).max(), np.abs(z4b[ok]).max()) * 1.1
    sc = axs[0].scatter(z4a[ok], z4b[ok], c=np.log10(cost17[ok]), cmap="viridis")
    axs[0].plot([-lim, lim], [-lim, lim], "k--", lw=1)
    axs[0].set_xlabel("Z4, ts_wep 15.1.0 (nm)")
    axs[0].set_ylabel("Z4, 17.8.1 + fix (nm)")
    axs[0].set_title("same stamps, both algorithms")
    plt.colorbar(sc, ax=axs[0], label="log10(fit cost)")

    axs[1].scatter(cost17[ok], dz4[ok])
    axs[1].set_xscale("log")
    axs[1].set_xlabel("fit cost (17.8.1 + fix)")
    axs[1].set_ylabel("|Z4 difference| (nm)")
    axs[1].set_title("disagreement grows with fit cost")
    fig.tight_layout()
""")

# ------------------------------------------------- 9c. simulated ground truth
md(r"""
### 9c. Simulated ground truth — and an honest note on circularity

`RSO-873_simulate_truth.py` renders AuxTel donut pairs from **known** Zernikes, adds a
Kolmogorov blur and Poisson/read noise, then refits them. Unlike everything above, the answer
is known, so this measures *recovery* rather than goodness-of-fit.

**The circularity caveat, stated plainly:** the renderer is `danish.DonutFactory` — the same
forward model the fit inverts. So the "does the model look like the data" panel below is
close to the §7 residual plot by construction; it is not independent evidence that the model
describes the real telescope. What it *does* test is whether the fitter inverts its own model
with the right scale and sign conventions, which is exactly the bug class §6a was.

Why not the 2023 route? `analysis_tools.simulate_defocal_atmosphere` /
`simulate_zernike_screen` (used in `AOS_DM-37629_auxTel_batoid_pipeline*.ipynb`) need
`wfsim.SimpleSimulator`, and **wfsim no longer exists on this system** — it lived at the
retired NCSA path that `analysis_tools.py:45` still `sys.path.append`s, and that module's
`import batoid` / `import wfsim` lines are commented out, so those functions raise
`NameError` if called. batoid and galsim are both available, hence the DonutFactory route,
and the batoid *raytrace* route in §9d which is genuinely independent.

**A sign trap worth knowing** (it cost me a wrong first answer): a batoid `OPDScreen` of
amplitude +X produces a wavefront of **−X**. Verified ratio −1.00 term by term — and Z6 comes
out at −1.84, so low-order cross-talk means it is not even a clean sign flip. The truth must
be read off `batoid.zernike(perturbed_telescope, ...)` *before* the defocus shift, which is
what `simulate_zernike_screen` does. Comparing against the injected `z_in` instead makes the
fit look far worse than it is.

Results over 8 trials with aberrations drawn uniform ±300 nm across Z4-Z11:

* the **zero-aberration control recovers to 0.3 nm rms** — conventions are right;
* **excluding Z4, median rms(fit − truth) = 24.5 nm**; Z6 and Z11 recover at corr 0.96;
* **Z4 is the fragile term** (rms 282 nm), and its error is predicted by the fit cost:
  corr(log cost, |Z4 error|) = **0.90**. Cost < 1000 → median 12 nm error; cost > 1000 → 68 nm.

That is the same effect as §9b: the versions diverge in Z4 precisely where cost is high,
because both are near the edge of reliability. **So cost — and the fitted fwhm — is a usable
internal quality flag**, and worth applying as a QA cut in the pipetask.
""")

code(r"""
sim = read_csv("RSO-873_sim_truth.csv")
if sim:
    noll_sim = list(range(4, 23))
    truth = np.array([[float(r[f"truth_Z{j}"]) for j in noll_sim] for r in sim])
    fitted = np.array([[float(r[f"fit_Z{j}"]) for j in noll_sim] for r in sim])
    sim_cost = np.array([float(r["cost"]) for r in sim])
    z4_err = np.abs(fitted[:, 0] - truth[:, 0])

    per_term = np.sqrt(np.mean((fitted - truth) ** 2, axis=0))
    print("per-term rms(fit - truth), nm:")
    for j, v in zip(noll_sim[:9], per_term[:9]):
        print(f"  Z{j:<3d} {v:7.1f}")
    no_z4 = np.sqrt(np.mean((fitted[:, 1:] - truth[:, 1:]) ** 2, axis=1))
    print("")
    print(f"excluding Z4: median rms = {np.median(no_z4):.1f} nm")
    print(f"corr(log cost, |Z4 err|) = {np.corrcoef(np.log(sim_cost), z4_err)[0,1]:.3f}")

    fig, axs = plt.subplots(1, 3, figsize=(15, 4.5))
    axs[0].scatter(truth.ravel(), fitted.ravel(), s=12, alpha=0.5)
    lim = np.abs(truth).max() * 1.2
    axs[0].plot([-lim, lim], [-lim, lim], "k--", lw=1)
    axs[0].set_xlabel("truth (nm)")
    axs[0].set_ylabel("fitted (nm)")
    axs[0].set_title("all terms Z4-Z22, 8 trials")

    axs[1].bar([f"Z{j}" for j in noll_sim[:9]], per_term[:9])
    axs[1].set_ylabel("rms(fit - truth) (nm)")
    axs[1].set_title("Z4 is the fragile term")
    axs[1].tick_params(axis="x", rotation=45)

    axs[2].scatter(sim_cost, z4_err)
    axs[2].set_xscale("log")
    axs[2].set_xlabel("fit cost")
    axs[2].set_ylabel("|Z4 error| (nm)")
    axs[2].set_title("cost predicts Z4 error (r=0.90)")
    fig.tight_layout()
""")

md(r"""
### 9d. An independent renderer: batoid ray tracing

To escape the circularity of §9c, render the donut by **ray tracing** the perturbed AuxTel
model — this is the forward model `wfsim` used underneath, and it shares no code with
danish's Fourier-optics `DonutFactory`. Recovering the input Zernikes from a raytraced donut
is therefore a real test of the fit rather than a self-inversion.

Verified working: injecting Z7 = 200 nm and Z11 = 150 nm via an `OPDScreen`, tracing 161712
unvignetted rays through a detector shifted by ±34.95 mm gives donuts spanning 163 px (extra)
and 228 px (intra) — the right size for AuxTel, and the intra/extra asymmetry is itself
physical.

The cell below regenerates a raytraced pair so it can be compared by eye with a real stamp
and with the DonutFactory model. Fitting raytraced donuts at scale is the natural next step,
and is the version of this test I would trust most.
""")

code(r'''
import batoid


def raytrace_donut(z_in_m, inst, sign, npix=227, nrad=200, naz=1200, wavelength=None):
    # Raytrace one defocused AuxTel donut -- independent of danish's forward model.
    wavelength = wavelength or inst.wavelength[BandLabel.REF]
    auxtel = batoid.Optic.fromYaml("AuxTel.yaml")
    phase = batoid.Zernike(
        np.asarray(z_in_m), R_outer=inst.radius, R_inner=inst.radius * inst.obscuration
    )
    perturbed = batoid.CompoundOptic(
        (
            batoid.optic.OPDScreen(
                batoid.Plane(), phase, name="PhaseScreen",
                obscuration=batoid.ObscNegation(batoid.ObscCircle(5.0)),
                coordSys=auxtel.stopSurface.coordSys,
            ),
            *auxtel.items,
        ),
        name="PerturbedAuxTel", backDist=auxtel.backDist, pupilSize=auxtel.pupilSize,
        inMedium=auxtel.inMedium, stopSurface=auxtel.stopSurface,
        sphereRadius=auxtel.sphereRadius, pupilObscuration=auxtel.pupilObscuration,
    )
    # truth read BEFORE the defocus shift (see the sign trap in 9c)
    zk_truth = batoid.zernike(
        perturbed, 0.0, 0.0, wavelength, eps=inst.obscuration, jmax=22
    ) * wavelength

    tel = perturbed.withLocallyShiftedOptic("Detector", [0, 0, sign * inst.defocalOffset])
    rays = batoid.RayVector.asPolar(
        optic=tel, wavelength=wavelength, theta_x=0.0, theta_y=0.0, nrad=nrad, naz=naz
    )
    tel.trace(rays)
    keep = ~rays.vignetted
    x, y = rays.x[keep], rays.y[keep]
    half = npix / 2 * inst.pixelSize
    img, _, _ = np.histogram2d(
        y - y.mean(), x - x.mean(), bins=npix, range=[[-half, half], [-half, half]]
    )
    return img, zk_truth


z_inject = np.zeros(12)
z_inject[7] = 200e-9
z_inject[11] = 150e-9
rt_extra, zk_rt = raytrace_donut(z_inject, inst, +1)
rt_intra, _ = raytrace_donut(z_inject, inst, -1)
print("raytrace truth Z4,Z7,Z8,Z11 (nm):", np.round(zk_rt[[4, 7, 8, 11]] * 1e9, 1))

fig, axs = plt.subplots(1, 4, figsize=(16, 4))
axs[0].imshow(rt_extra, origin="lower", cmap="gray")
axs[0].set_title("raytrace, extra")
axs[1].imshow(rt_intra, origin="lower", cmap="gray")
axs[1].set_title("raytrace, intra")
axs[2].imshow(fit["images"][0], origin="lower", cmap="gray")
axs[2].set_title("real stamp, extra")
if fit["model_images"] is not None:
    axs[3].imshow(np.asarray(fit["model_images"][0]), origin="lower", cmap="gray")
    axs[3].set_title("danish model, extra")
else:
    axs[3].axis("off")
for ax in axs:
    ax.set_xticks([])
    ax.set_yticks([])
fig.suptitle("independent raytrace vs real data vs danish model")
fig.tight_layout()
''')

md(r"""
## 10. Next: package as a pipetask

The fit lives in `lsst.ts.wep.task.latissMonolith` on the ts_wep `tickets/RSO-873` branch.
Packaging it follows `donutBlitzMonolith` exactly — that task is the precedent for a
single-file, ISR-to-Zernikes PipelineTask that owns its own cutout, and it is driven by
`donut_viz/pipelines/production/lsstcam_usdf/blitzBase.yaml`:

```yaml
description: rapid analysis pipeline for LSSTCam using the MonolithBlitzTask
instrument: lsst.obs.lsst.LsstCam
tasks:
  donutBlitzMonolithTask:
    class: lsst.ts.wep.task.donutBlitzMonolith.DonutBlitzMonolithTask
    config:
      donutSelector.useCustomMagLimit: True
      ...
  formatBlitzTask:
    class: lsst.donut.viz.FormatBlitzTask
```

The LATISS analogue needs `instrument: lsst.obs.lsst.Latiss`, a
`LatissMonolithTask` wrapping `fit_latiss_danish`, and none of the refcat/astrometry config
(one bright donut per exposure, so `donutSelector`/`astromTask` do not apply). Specifics
established in this notebook that the task config must pin:

* `opticalModel: "onAxis"` — the default `offAxis` is wrong for AuxTel.
* `donutStampSize` — the default **160** is LSSTCam-sized and clips an AuxTel donut, which is
  194 px across. ts_wep's LATISS test pipelines use 200; `latiss_wep_align` derives 228 from
  `dz=0.8`.
* `nollIndices` 4-22.
* No `intrinsicZernikes` connection — LATISS has no such calibration.
* Keep `CutOutDonutsScienceSensorTask` for the cutout (§7 vs §7a).
""")

md(r"""
## 11. Where this leaves `latiss_wep_align`

* Modern ts_wep **cannot** run `run_wep` as written: `doBlurClip=True` crashes on a single
  donut pair (`calcZernikesTask.py:545`, `sigma_clip` of a length-1 array gives a 0-d mask),
  `zkClipType="deviation"` NaNs the average because LATISS has no intrinsic Zernike
  calibration, and Danish itself is dead from §6a.
* Fixing §6a/§6b **in the monolith** rather than in `_prepDanish` — once the monolith exists
  AuxTel never calls `_prepDanish`, and `_prepDanish` is correct for LSSTCam, so
  special-casing shared code would complicate it for no live use case.
* The monolith **keeps** `CutOutDonutsScienceSensorTask` (§7): its donuts are well centred,
  residuals are small, and the model matches the stamp. A hand-rolled box cutout was tried
  (§7a) and is worse — it does not centre the donut (`inner_frac` ~1.1, Z8 collapses).
* The hexapod arithmetic is confirmed against the `ts_externalscripts` unit test:
  de-rotating `[-60.1, 30.8, 50.7]` at 12° gives `[-52.4, 42.6, 50.7]` vs the expected
  `[-52.6, 42.4, 50.7]`.
* Not settled here: whether −416 nm or −468 nm is the better answer for this pair. That
  needs the residual images above plus a focus sweep, not a single-pair comparison.
""")

nb = {
    "cells": cells,
    "metadata": {
        "kernelspec": {"display_name": "LSST", "language": "python", "name": "lsst"},
        "language_info": {
            "codemirror_mode": {"name": "ipython", "version": 3},
            "file_extension": ".py",
            "mimetype": "text/x-python",
            "name": "python",
            "nbconvert_exporter": "python",
            "pygments_lexer": "ipython3",
            "version": "3.13.0",
        },
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}

OUT.write_text(json.dumps(nb, indent=1) + "\n")
print(f"wrote {OUT} with {len(cells)} cells")

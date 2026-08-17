"""Ground-truth validation: render AuxTel donuts from KNOWN Zernikes, then refit them.

Follows the logic of AOS_DM-37629_auxTel_batoid_pipeline*.ipynb
(`analysis_tools.simulate_zernike_screen`): put a Zernike phase screen at the stop of the
AuxTel batoid model, defocus by shifting M2 +/- 0.8 mm, render intra/extra images, and read
the truth wavefront off the *unshifted* perturbed telescope.

Difference from the 2023 notebook: it used `wfsim.SimpleSimulator` for atmosphere + stellar
SED, and **wfsim no longer exists on this system** (it lived at a retired NCSA path;
`analysis_tools.py` still has a dead `sys.path.append` to it, and its `import batoid` /
`import wfsim` lines are commented out, so the `simulate_*` functions raise NameError).
batoid and galsim are both available, so we render with `danish.DonutFactory` instead --
the same forward model the fit uses.

That last point is the honest caveat: rendering with DonutFactory and fitting with
DonutFactory tests self-consistency (does the fitter invert its own model, with the right
scale/sign conventions?), NOT whether the model matches the real telescope. It is exactly
the test that catches the zkRef bug class, since that was a units/scale error. Adding a
Kolmogorov atmospheric blur makes it a slightly harder problem than pure inversion.
"""
import csv
import sys
from pathlib import Path

import numpy as np

import batoid
import danish
import galsim

from lsst.ts.wep.utils import getTaskInstrument
from lsst.ts.wep.utils.enumUtils import BandLabel

REPO_DIR = Path("/sdf/data/rubin/user/scichris/WORK/AOS")
TS_WEP_TASK_DIR = (
    "/sdf/data/rubin/user/scichris/WORK/aos_packages/ts_wep/python/lsst/ts/wep/task"
)
NOLL = np.arange(4, 23)
NPIX = 227  # odd, comparable to the 228 px CWFS stamps


def truth_wavefront(z_in, inst, wavelength, thx=0.0, thy=0.0):
    """Zernikes of the AuxTel model perturbed by a phase screen, read BEFORE defocus.

    Mirrors analysis_tools.simulate_zernike_screen: the screen sits at the stop surface,
    and the truth is read off the un-shifted telescope.
    """
    auxtel = batoid.Optic.fromYaml("AuxTel.yaml")
    phase = batoid.Zernike(
        np.asarray(z_in), R_outer=inst.radius, R_inner=inst.radius * inst.obscuration
    )
    perturbed = batoid.CompoundOptic(
        (
            batoid.optic.OPDScreen(
                batoid.Plane(),
                phase,
                name="PhaseScreen",
                obscuration=batoid.ObscNegation(batoid.ObscCircle(5.0)),
                coordSys=auxtel.stopSurface.coordSys,
            ),
            *auxtel.items,
        ),
        name="PerturbedAuxTel",
        backDist=auxtel.backDist,
        pupilSize=auxtel.pupilSize,
        inMedium=auxtel.inMedium,
        stopSurface=auxtel.stopSurface,
        sphereRadius=auxtel.sphereRadius,
        pupilObscuration=auxtel.pupilObscuration,
    )
    zk = batoid.zernike(
        perturbed, thx, thy, wavelength, eps=inst.obscuration, jmax=max(NOLL) + 1
    )
    return zk * wavelength, perturbed  # metres


def render_pair(z_truth_m, inst, factory, wavelength, fwhm_arcsec=0.8, flux=2e6,
                seed=1234, thx=0.0, thy=0.0):
    """Render intra/extra donuts for a known wavefront, with atmospheric blur and noise."""
    rng = np.random.default_rng(seed)
    imgs = {}
    for side, sign in (("extra", +1), ("intra", -1)):
        # zkRef in the same convention the fit uses: OPD of the defocused telescope,
        # plus the injected aberration.
        tel = batoid.Optic.fromYaml("AuxTel.yaml").withLocallyShiftedOptic(
            "Detector", [0, 0, sign * inst.defocalOffset]
        )
        zk_defocus = batoid.zernike(
            tel, thx, thy, wavelength=wavelength, nx=255,
            eps=inst.obscuration, jmax=78,
        ) * wavelength
        # `z_truth_m` is the WAVEFRONT to inject (already in batoid's wavefront sign
        # convention), so it adds directly to the defocus OPD. Do not pass a raw
        # OPDScreen amplitude here -- that differs by a sign.
        aber = np.zeros(79)
        aber[4:] = zk_defocus[4:]
        aber[: len(z_truth_m)] += np.asarray(z_truth_m)
        aber[:4] = 0.0

        img = factory.image(aberrations=tuple(aber), thx=thx, thy=thy, npix=NPIX)
        img = np.asarray(img, dtype=float)
        img = img / img.max() * flux

        # Atmospheric blur: convolve with a Kolmogorov kernel of the given seeing.
        if fwhm_arcsec > 0:
            pixel_scale = np.rad2deg(inst.pixelSize / inst.focalLength) * 3600
            gs = galsim.Image(img, scale=pixel_scale)
            psf = galsim.Kolmogorov(fwhm=fwhm_arcsec)
            gs = galsim.Convolve(
                galsim.InterpolatedImage(gs), psf
            ).drawImage(nx=NPIX, ny=NPIX, scale=pixel_scale)
            img = np.asarray(gs.array, dtype=float)

        # Poisson noise plus a small read noise, so the fit sees a realistic SNR.
        img = rng.poisson(np.clip(img, 0, None)).astype(float)
        img += rng.normal(0.0, 10.0, img.shape)
        imgs[side] = img
    return imgs


if __name__ == "__main__":
    n_trials = int(sys.argv[1]) if len(sys.argv) > 1 else 8
    out_csv = sys.argv[2] if len(sys.argv) > 2 else str(REPO_DIR / "RSO-873_sim_truth.csv")

    sys.path.insert(0, TS_WEP_TASK_DIR)
    from latissMonolith import fit_latiss_danish_arrays

    inst = getTaskInstrument("LATISS", "RXX_S00", None)
    wavelength = inst.wavelength[BandLabel.REF]
    factory = danish.DonutFactory(
        R_outer=inst.radius,
        R_inner=inst.radius * inst.obscuration,
        mask_params=inst.maskParams,
        focal_length=inst.focalLength,
        pixel_scale=inst.pixelSize,
    )
    print(f"AuxTel: defocalOffset={inst.defocalOffset*1e3:.2f} mm, "
          f"wavelength={wavelength*1e9:.0f} nm, npix={NPIX}", flush=True)

    rng = np.random.default_rng(2026)
    rows = []
    for trial in range(n_trials):
        # Inject a plausible AuxTel aberration: a few hundred nm spread over Z4-Z11.
        z_in = np.zeros(12)
        z_in[4:12] = rng.uniform(-300e-9, 300e-9, size=8)
        if trial == 0:
            z_in[:] = 0.0  # control: no aberration beyond the defocus itself
        # The truth is the wavefront batoid actually produces, NOT z_in: an OPDScreen of
        # +X yields a wavefront of -X (verified: ratio -1.00 term by term), and low-order
        # cross-talk means e.g. Z6 is not a clean sign flip either. So always read the
        # truth off the perturbed telescope rather than assuming it equals the input.
        z_truth_full, _ = truth_wavefront(z_in, inst, wavelength)

        # Render from the wavefront truth, so render and compare use one convention.
        z_render = np.zeros(23)
        upto = min(len(z_truth_full), len(z_render))
        z_render[:upto] = z_truth_full[:upto]
        z_render[:4] = 0.0
        imgs = render_pair(z_render, inst, factory, wavelength, seed=1000 + trial)
        try:
            out = fit_latiss_danish_arrays(
                imgs["extra"], imgs["intra"], (0.0, 0.0), (0.0, 0.0), inst,
                band="ref", noll_indices=NOLL,
            )
        except Exception as exc:
            print(f"[{trial}] FIT FAILED {type(exc).__name__}: {str(exc)[:70]}", flush=True)
            continue

        truth_nm = {
            int(j): float(z_truth_full[j] * 1e9) if j < len(z_truth_full) else 0.0
            for j in NOLL
        }
        fit_nm = out["zernikes_nm"]
        diffs = np.array([fit_nm[int(j)] - truth_nm[int(j)] for j in NOLL])

        row = dict(
            trial=trial,
            nfev=out["nfev"],
            cost=out["cost"],
            fwhm=out["fwhm"],
            rms_resid_nm=float(np.sqrt(np.mean(diffs**2))),
            max_resid_nm=float(np.max(np.abs(diffs))),
            **{f"truth_Z{int(j)}": truth_nm[int(j)] for j in NOLL},
            **{f"fit_Z{int(j)}": fit_nm[int(j)] for j in NOLL},
        )
        rows.append(row)
        print(
            f"[{trial}] rms(fit-truth) over Z4-Z22 = {row['rms_resid_nm']:7.1f} nm  "
            f"max {row['max_resid_nm']:7.1f} nm   "
            f"Z4 {truth_nm[4]:+7.1f}->{fit_nm[4]:+7.1f}  "
            f"Z7 {truth_nm[7]:+7.1f}->{fit_nm[7]:+7.1f}  "
            f"Z8 {truth_nm[8]:+7.1f}->{fit_nm[8]:+7.1f}",
            flush=True,
        )

    if rows:
        with open(out_csv, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
        print(f"\nwrote {out_csv} ({len(rows)} trials)")
        rms = np.array([r["rms_resid_nm"] for r in rows])
        print(f"rms(fit - truth): median {np.median(rms):.1f} nm, "
              f"worst {rms.max():.1f} nm over {len(rows)} trials")

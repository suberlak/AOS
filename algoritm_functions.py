
from scipy.ndimage import generate_binary_structure, iterate_structure
from scipy.ndimage import binary_dilation, binary_erosion
from scipy.interpolate import RectBivariateSpline
from scipy.signal import correlate

from lsst.ts.wep.ParamReader import ParamReader
from lsst.ts.wep.cwfs.Tool import (
    padArray,
    extractArray,
    ZernikeAnnularGrad,
    ZernikeAnnularJacobian,
)
from lsst.ts.wep.cwfs.Image import Image
from lsst.ts.wep.Utility import DefocalType, CentroidFindType
from galsim.utilities import horner2d

import yaml
import os
import numpy as np
from scipy.ndimage import rotate

import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import matplotlib as mpl
from matplotlib import rcParams

from lsst.ts.wep.cwfs.Instrument import Instrument
from lsst.ts.wep.cwfs.Algorithm import Algorithm
from lsst.ts.wep.cwfs.CompensableImage import CompensableImage
from lsst.ts.wep.Utility import (
    getConfigDir,
    DonutTemplateType,
    DefocalType,
    CamType,
    getCamType,
    getDefocalDisInMm,
    CentroidFindType
)

from lsst.ts.wep.task.DonutStamps import DonutStamp, DonutStamps
from lsst.ts.wep.task.EstimateZernikesCwfsTask import (
    EstimateZernikesCwfsTask,
    EstimateZernikesCwfsTaskConfig,
)

from lsst.daf import butler as dafButler
import lsst.afw.cameraGeom as cameraGeom
from lsst.afw.cameraGeom import FOCAL_PLANE, PIXELS, FIELD_ANGLE
from lsst.geom import Point2D

from astropy.visualization import ZScaleInterval

from lsst.ts.wep.cwfs.Tool import (
    padArray,
    extractArray,
    ZernikeAnnularEval,
    ZernikeMaskedFit,
)

rcParams['ytick.labelsize'] = 15
rcParams['xtick.labelsize'] = 15
rcParams['axes.labelsize'
        ] = 20
rcParams['axes.linewidth'] = 2
rcParams['font.size'] = 15
rcParams['axes.titlesize'] = 18



def imageCoCenter_store(I, inst, fov=3.5, debugLevel=0,
                       store=None,increaseFactor=1.0):
    """Shift the weighting center of donut to the center of reference
    image with the correction of projection of fieldX and fieldY.
    Parameters
    ----------
    inst : Instrument
        Instrument to use.
    fov : float, optional
        Field of view (FOV) of telescope. (the default is 3.5.)
    debugLevel : int, optional
        Show the information under the running. If the value is higher, the
        information shows more. It can be 0, 1, 2, or 3. (the default is
        0.)
    store: dic, optional
        A dictionary where additional information is to be stored
    increaseFactor: float, optional 
        A factor by which we scale the radial shift imparted 
        to the weighted donut image in this step (default is 1.0)
    """

    # Calculate the weighting center (x, y) and radius
    x1, y1 = I._image.getCenterAndR()[0:2]
    
    #  make a storage array if none was supplied
    if store is None:
        store = {}
    store['cocenter_centroid_x1'] = x1
    store['cocenter_centroid_y1'] = y1

    # Show the co-center information
    if debugLevel >= 3:
        print("imageCoCenter: (x, y) = (%8.2f,%8.2f)\n" % (x1, y1))

    # Calculate the center position on image
    # 0.5 is the half of 1 pixel
    dimOfDonut = inst.dimOfDonutImg
    stampCenterx1 = dimOfDonut / 2 + 0.5
    stampCentery1 = dimOfDonut / 2 + 0.5
    store['cocenter_centerx1_original'] = stampCenterx1
    store['cocenter_centerx1_original'] = stampCentery1

    # Shift in the radial direction
    # The field of view (FOV) of LSST camera is 3.5 degree
    offset = inst.defocalDisOffset
    pixelSize = inst.pixelSize
    radialShift = increaseFactor*fov * (offset / 1e-3) * (10e-6 / pixelSize)
    store['cocenter_fov'] = fov
    store['cocenter_offset'] = offset
    store['cocenter_pixelSize'] = pixelSize

    # Calculate the projection of distance of donut to center
    fieldDist = I._getFieldDistFromOrigin()
    radialShift = radialShift * (fieldDist / (fov / 2))
    store['cocenter_fieldDist'] = fieldDist
    store['cocenter_radialShift'] = radialShift

    # Do not consider the condition out of FOV of lsst
    if fieldDist > (fov / 2):
        radialShift = 0

    # Calculate the cos(theta) for projection
    I1c = I.fieldX / fieldDist

    # Calculate the sin(theta) for projection
    I1s = I.fieldY / fieldDist
    store['cocenter_xShift'] = radialShift * I1c
    store['cocenter_yShift'] = radialShift * I1s

    # Get the projected x, y-coordinate        
    stampCenterx1 = stampCenterx1 + radialShift * I1c
    stampCentery1 = stampCentery1 + radialShift * I1s
    store['cocenter_centerx1_shifted'] = stampCenterx1
    store['cocenter_centery1_shifted'] = stampCentery1

    # Shift the image to the projected position
    I.updateImage(
        np.roll(I.getImg(), int(np.round(stampCentery1 - y1)), axis=0)
    )
    I.updateImage(
        np.roll(I.getImg(), int(np.round(stampCenterx1 - x1)), axis=1)
    )
    store['cocenter_x_shift_amount'] = int(np.round(stampCentery1 - y1))
    store['cocenter_y_shift_amount'] = int(np.round(stampCenterx1 - x1))
    return store

def centerOnProjection_store(img, template, window=20,
                            store=None):
    """Center the image to the template's center.
    Parameters
    ----------
    img : numpy.array
        Image to be centered with the template. The input image needs to
        be a n-by-n matrix.
    template : numpy.array
        Template image to have the same dimension as the input image
        ('img'). The center of template is the position of input image
        tries to align with.
    window : int, optional
        Size of window in pixel. Assume the difference of centers of input
        image and template is in this range (e.g. [-window/2, window/2] if
        1D). (the default is 20.)
    Returns
    -------
    numpy.array
        Recentered image.
    """

    # Calculate the cross-correlate
    corr = correlate(img, template, mode="same")

    # Calculate the shifts of center

    # Only consider the shifts in a certain window (range)
    # Align the input image to the center of template
    length = template.shape[0]
    center = length // 2

    r = window // 2

    mask = np.zeros(corr.shape)
    mask[center - r : center + r, center - r : center + r] = 1
    idx = np.argmax(corr * mask)

    # The above 'idx' is an interger. Need to rematch it to the
    # two-dimension position (x and y)
    xmatch = idx % length
    ymatch = idx // length

    dx = center - xmatch
    dy = center - ymatch
    store['recenter_x_shift'] = dx
    store['recenter_y_shift'] = dy
    
    # Shift/ recenter the input image
    return np.roll(np.roll(img, dx, axis=1), dy, axis=0), store



def compensate_store(I, inst, algo, zcCol, model, store):
    ''' Do the image compensation, but keep 
    calculated quantities in a dictionary for
    plotting later
    '''
    # initialize storage dictionary if needed
    if store is None:
        store = {}
    # else, assume storage is an already existing dictionary 
    store['imgBeforeCompensate'] = I.getImg()
    # Dimension of image
    sm, sn = I.getImg().shape

    # Dimension of projected image on focal plane
    projSamples = sm

    # Let us create a look-up table for x -> xp first.
    luty, lutx = np.mgrid[
        -(projSamples / 2 - 0.5) : (projSamples / 2 + 0.5),
        -(projSamples / 2 - 0.5) : (projSamples / 2 + 0.5),

    ]

    sensorFactor = algo._inst.getSensorFactor()
    lutx = lutx / (projSamples / 2 / sensorFactor)
    luty = luty / (projSamples / 2 / sensorFactor)


    # Set up the mapping
    lutxp, lutyp, J = I._aperture2image(
    inst, algo, zcCol, lutx, luty, projSamples, model
    )

    show_lutxyp = I._showProjection(
                lutxp, lutyp, sensorFactor, projSamples, raytrace=False
            )
    store['show_lutxyp'] = show_lutxyp

    # Extend the dimension of image by 20 pixel in x and y direction
    show_lutxyp_padded = padArray(show_lutxyp, projSamples + 20)


    # Get the binary matrix of image on pupil plane if raytrace=False
    struct0 = generate_binary_structure(2, 1)

    struct1 = iterate_structure(struct0, 4)

    struct = binary_dilation(struct1, structure=struct0, iterations=2).astype(int)

    show_lutxyp_dilated = binary_dilation(show_lutxyp_padded, structure=struct)

    show_lutxyp_eroded = binary_erosion(show_lutxyp_dilated, structure=struct)

    # Extract the region from the center of image and get the original one
    show_lutxyp_extracted = extractArray(show_lutxyp_eroded, projSamples)
    store['show_lutxyp_extracted'] = show_lutxyp_extracted

    imgRecenter, store = centerOnProjection_store(
                I.getImg(), show_lutxyp.astype(float), window=20, store=store
                                                    ) 

    I.updateImage(imgRecenter)
    store['imgAfterRecenter'] = imgRecenter

    # Construct the interpolant to get the intensity on (x', p') plane
    # that corresponds to the grid points on (x,y)
    yp, xp = np.mgrid[
        -(sm / 2 - 0.5) : (sm / 2 + 0.5), -(sm / 2 - 0.5) : (sm / 2 + 0.5)
    ]

    xp = xp / (sm / 2 / sensorFactor)
    yp = yp / (sm / 2 / sensorFactor)

    # Put the NaN to be 0 for the interpolate to use
    lutxp[np.isnan(lutxp)] = 0
    lutyp[np.isnan(lutyp)] = 0

    # Construct the function for interpolation
    ip = RectBivariateSpline(yp[:, 0], xp[0, :], I.getImg(), kx=1, ky=1)

    # Construct the projected image by the interpolation
    lutIp = ip(lutyp, lutxp, grid=False)
    store['lutIp'] = lutIp
    store['Jacobian'] = J 

    # Calculate the image on focal plane with compensation based on flux
    # conservation
    # I(x, y)/I'(x', y') = J = (dx'/dx)*(dy'/dy) - (dx'/dy)*(dy'/dx)
    I.updateImage(lutIp * J)

    if I.defocalType == DefocalType.Extra:
        I.updateImage(np.rot90(I.getImg(), k=2))
        #print('Rotated the extra-focal image')

    # Put NaN to be 0
    imgCompensate = I.getImg()
    imgCompensate[np.isnan(imgCompensate)] = 0

    # Check the compensated image has the problem or not.
    # The negative value means the over-compensation from wavefront error
    if np.any(imgCompensate < 0) and np.all(I.image0 >= 0):
        print(
            "WARNING: negative scale parameter, image is within caustic, zcCol (in um)=\n"
        )
        I.caustic = True

    # Put the overcompensated part to be 0
    imgCompensate[imgCompensate < 0] = 0
    I.updateImage(imgCompensate)
    store['imgAfterCompensate'] = I.getImg()
    return I, store


def _solvePoissonEq_store(algo, I1, I2, iOutItr=0, store=None):
    """Solve the Poisson's equation by Fourier transform (differential) or
    serial expansion (integration).
    There is no convergence for fft actually. Need to add the difference
    comparison and X-alpha method. Need to discuss further for this.
    Parameters
    ----------
    I1 : CompensableImage
        Intra- or extra-focal image.
    I2 : CompensableImage
        Intra- or extra-focal image.
    iOutItr : int, optional
        ith number of outer loop iteration which is important in "fft"
        algorithm. (the default is 0.)
    Returns
    -------
    numpy.ndarray
        Coefficients of normal/ annular Zernike polynomials.
    numpy.ndarray
        Estimated wavefront.
    """

    # Calculate the aperture pixel size
    apertureDiameter = algo._inst.apertureDiameter
    sensorFactor = algo._inst.getSensorFactor()
    dimOfDonut = algo._inst.dimOfDonutImg
    aperturePixelSize = apertureDiameter * sensorFactor / dimOfDonut

    # Calculate the differential Omega
    dOmega = aperturePixelSize**2

    # Solve the Poisson's equation based on the type of algorithm
    numTerms = algo.getNumOfZernikes()
    zobsR = algo.getObsOfZernikes()
    PoissonSolver = algo.getPoissonSolverName()
    if PoissonSolver == "fft":

        # Use the differential method by fft to solve the Poisson's
        # equation

        # Parameter to determine the threshold of calculating I0.
        sumclipSequence = algo.getSignalClipSequence()
        cliplevel = sumclipSequence[iOutItr]

        # Generate the v, u-coordinates on pupil plane
        padDim = algo.getFftDimension()
        v, u = np.mgrid[
            -0.5
            / aperturePixelSize : 0.5
            / aperturePixelSize : 1.0
            / padDim
            / aperturePixelSize,
            -0.5
            / aperturePixelSize : 0.5
            / aperturePixelSize : 1.0
            / padDim
            / aperturePixelSize,
        ]

        # Show the threshold and pupil coordinate information
        if algo.debugLevel >= 3:
            print("iOuter=%d, cliplevel=%4.2f" % (iOutItr, cliplevel))
            print(v.shape)

        # Calculate the const of fft:
        # FT{Delta W} = -4*pi^2*(u^2+v^2) * FT{W}
        u2v2 = -4 * (np.pi**2) * (u * u + v * v)

        # Set origin to Inf to result in 0 at origin after filtering
        ctrIdx = int(np.floor(padDim / 2.0))
        u2v2[ctrIdx, ctrIdx] = np.inf

        # Calculate the wavefront signal
        Sini = algo._createSignal(I1, I2, cliplevel)

        # Find the just-outside and just-inside indices of a ring in pixels
        # This is for the use in setting dWdn = 0
        boundaryT = algo.getBoundaryThickness()

        struct = generate_binary_structure(2, 1)
        struct = iterate_structure(struct, boundaryT)

        ApringOut = np.logical_xor(
            binary_dilation(algo.mask_pupil, structure=struct), algo.mask_pupil
        ).astype(int)
        ApringIn = np.logical_xor(
            binary_erosion(algo.mask_pupil, structure=struct), algo.mask_pupil
        ).astype(int)

        bordery, borderx = np.nonzero(ApringOut)

        # Put the signal in boundary (since there's no existing Sestimate,
        # S just equals self.S as the initial condition of SCF
        S = Sini.copy()
        for jj in range(algo.getNumOfInnerItr()):

            # Calculate FT{S}
            SFFT = np.fft.fftshift(np.fft.fft2(np.fft.fftshift(S)))

            # Calculate W by W=IFT{ FT{S}/(-4*pi^2*(u^2+v^2)) }
            W = np.fft.fftshift(
                np.fft.irfft2(np.fft.fftshift(SFFT / u2v2), s=S.shape)
            )

            # Estimate the wavefront (includes zeroing offset & masking to
            # the aperture size)

            # Take the estimated wavefront
            West = extractArray(W, dimOfDonut)

            # Calculate the offset
            offset = West[self.mask_pupil == 1].mean()
            West = West - offset
            West[self.mask_pupil == 0] = 0

            # Set dWestimate/dn = 0 around boundary
            WestdWdn0 = West.copy()

            # Do a 3x3 average around each border pixel, including only
            # those pixels inside the aperture. This averaging can be
            # efficiently computed using 1 numpy/scipy vectorized
            # convolve2d instruction to first sum the values in the 3x3
            # region, and dividing by a second convolve2d which counts
            # the non-zero pixels in each 3x3 region.

            kernel = np.ones((1 + 2 * boundaryT, 1 + 2 * boundaryT))
            tmp = convolve2d(West * ApringIn, kernel, mode="same")
            tmp /= convolve2d(ApringIn, kernel, mode="same")
            WestdWdn0[borderx, bordery] = tmp[borderx, bordery]

            # Take Laplacian to find sensor signal estimate (Delta W = S)
            del2W = laplace(WestdWdn0) / dOmega

            # Extend the dimension of signal to the order of 2 for "fft" to
            # use
            Sest = padArray(del2W, padDim)

            # Put signal back inside boundary, leaving the rest of
            # Sestimate
            Sest[algo.mask_pupil_pad == 1] = Sini[algo.mask_pupil_pad == 1]

            # Need to recheck this condition
            S = Sest

        # Calculate the coefficient of normal/ annular Zernike polynomials
        if algo.getCompensatorMode() == "zer":
            xSensor, ySensor = algo._inst.getSensorCoor()
            zc = ZernikeMaskedFit(
                West, xSensor, ySensor, numTerms, algo.mask_pupil, zobsR
            )
        else:
            zc = np.zeros(numTerms)

    elif PoissonSolver == "exp":

        # Use the integration method by serial expansion to solve the
        # Poisson's equation

        # Calculate I0 and dI
        I0, dI = algo._getdIandI(I1, I2)
        store['I0'] = I0
        store['dI'] = dI

        # Get the x, y coordinate in mask. The element outside mask is 0.
        xSensor, ySensor = algo._inst.getSensorCoor()
        xSensor = xSensor * algo.mask_comp
        ySensor = ySensor * algo.mask_comp

        # Create the F matrix and Zernike-related matrixes

        # Get Zernike and gradient bases from cache.  These are each
        # (nzk, npix, npix) arrays, with the first dimension indicating
        # the Noll index.
        zk, dzkdx, dzkdy = algo._zernikeBasisCache()

        # Eqn. (19) from Xin et al., Appl. Opt. 54, 9045-9054 (2015).
        # F_j = \int (d_z I) Z_j d_Omega
        F = np.tensordot(dI, zk, axes=((0, 1), (1, 2))) * dOmega
        # Eqn. (20) from Xin et al., Appl. Opt. 54, 9045-9054 (2015).
        # M_ij = \int I (grad Z_j) . (grad Z_i) d_Omega
        #      =   \int I (dZ_i/dx) (dZ_j/dx) d_Omega
        #        + \int I (dZ_i/dy) (dZ_j/dy) d_Omega
        Mij = np.einsum("ab,iab,jab->ij", I0, dzkdx, dzkdx)
        Mij += np.einsum("ab,iab,jab->ij", I0, dzkdy, dzkdy)
        Mij *= dOmega / (apertureDiameter / 2.0) ** 2

        # Calculate dz
        focalLength = algo._inst.focalLength
        offset = algo._inst.defocalDisOffset
        dz = 2 * focalLength * (focalLength - offset) / offset

        # Define zc
        zc = np.zeros(numTerms)

        # Consider specific Zk terms only
        idx = algo.getZernikeTerms()

        # Solve the equation: M*W = F => W = M^(-1)*F
        zc_tmp = np.linalg.lstsq(Mij[:, idx][idx], F[idx], rcond=None)[0] / dz
        zc[idx] = zc_tmp

        # Estimate the wavefront surface based on z4 - z22
        # z0 - z3 are set to be 0 instead
        West = ZernikeAnnularEval(
            np.concatenate(([0, 0, 0], zc[3:])), xSensor, ySensor, zobsR
        )

    return zc, West, store

def runIt_store(algo, I1, I2, model, tol=1e-3, store=None,
               doCoCenter=True, increaseFactor=1.0):
    ''' 
    doCoCenter : whether to run coCenter step or not
    increaseFactor : the amount by which to scale the amount of radial 
        shift imparted in the cocenter step'''
    # To have the iteration time initiated from global variable is to
    # distinguish the manually and automatically iteration processes.
    itr = algo.currentItr
    if store is None:
        store = {}
    while itr <= algo.getNumOfOuterItr():

        stopItr, store = _singleItr_store(algo, I1, I2, model, tol, store,
                                         doCoCenter, increaseFactor)

        # Stop the iteration of outer loop if converged
        if stopItr:
            break

        itr += 1
    
    return algo, store



def _singleItr_store(algo, I1, I2, model, tol=1e-3, store=None,
                    doCoCenter = True, increaseFactor=1.0):
   
    #algo.debugLevel = 3 
    
    # Use the zonal mode ("zer")
    compMode = algo.getCompensatorMode()

    # Define the gain of feedbackGain
    feedbackGain = algo.getFeedbackGain()

    # Set the pre-condition
    # ... only zeroth iteration ... 
    # Rename this index (currentItr) for the simplification
    jj = algo.currentItr
    #print('iteration ', jj)
    store[jj] = {'intra':{}, 'extra':{}, 'both':{}}
    if jj == 0:

        # Check this is the first time of running iteration or not
        if I1.getImgInit() is None or I2.getImgInit() is None:

            # Check the image dimension
            if I1.getImg().shape != I2.getImg().shape:
                print(
                    "Error: The intra and extra image stamps need to be of same size."
                )


            # Calculate the pupil mask (binary matrix) and related
            # parameters
            boundaryT = algo.getBoundaryThickness()
            I1.makeMask(algo._inst, model, boundaryT, 1)
            I2.makeMask(algo._inst, model, boundaryT, 1)
            
            algo._makeMasterMask(I1, I2, algo.getPoissonSolverName())
            store[jj]['both']['mask_pupil'] = algo.mask_pupil
            store[jj]['both']['mask_comp'] = algo.mask_comp
            
        # Load the offAxis correction coefficients
        if model == "offAxis":
            offAxisPolyOrder = algo.getOffAxisPolyOrder()
            I1.setOffAxisCorr(algo._inst, offAxisPolyOrder)
            I2.setOffAxisCorr(algo._inst, offAxisPolyOrder)

        # store images before co-centering
        store[jj]['intra']['imgBeforeCocenter'] = I1.getImg()
        store[jj]['extra']['imgBeforeCocenter'] = I2.getImg()
        
        # default value 
        store[jj]['doCoCenter'] = False
        if doCoCenter : 
            for I, defocal in zip([I1,I2], ['intra','extra']):
                store[jj][defocal] = imageCoCenter_store(I, algo._inst, 
                                                     store=store[jj][defocal],
                                                     increaseFactor=increaseFactor
                                                        )
            # override default if actually done that 
            store[jj]['doCoCenter'] = True 
        
        # store images after that step whether it 
        # has been done or not... 
        store[jj]['intra']['imgAfterCocenter'] = I1.getImg()
        store[jj]['extra']['imgAfterCocenter'] = I2.getImg()

        # Update the self-initial image
        I1.updateImgInit()
        I2.updateImgInit()
    
        # Initialize the variables used in the iteration.
        algo.zcomp = np.zeros(algo.getNumOfZernikes())
        algo.zc = algo.zcomp.copy()
        store[jj]['both']['zcomp'] = algo.zcomp
        store[jj]['both']['zc'] = algo.zc

        dimOfDonut = algo._inst.dimOfDonutImg
        algo.wcomp = np.zeros((dimOfDonut, dimOfDonut)) 
        
        algo.West = algo.wcomp.copy()
        store[jj]['both']['wcomp'] = algo.wcomp
        store[jj]['both']['West'] = algo.West
        algo.caustic = False

        # ... only zeroth iteration ... 
            
            
    # Solve the transport of intensity equation (TIE)
    if not algo.caustic:

        # Reset the images before the compensation
        I1.updateImage(I1.getImgInit().copy())
        I2.updateImage(I2.getImgInit().copy())

        if compMode == "zer":

            # Zk coefficient from the previous iteration
            ztmp = algo.zc.copy()

            # Do the feedback of Zk from the lower terms first based on the
            # sequence defined in compSequence
            if jj != 0:
                compSequence = algo.getCompSequence()
                #print(compSequence)
                ztmp[int(compSequence[jj - 1]) :] = 0

            # Add partial feedback of residual estimated wavefront in Zk
            algo.zcomp = algo.zcomp + ztmp * feedbackGain

            # store the image so that nothing in the below will affect it 
            intraBeforeCompensate = I1.getImg()

            I1,  store[jj]['intra'] = compensate_store(I1, algo._inst, 
                                                       algo, algo.zcomp, 
                                                       model, store=store[jj]['intra'])
            I2,  store[jj]['extra'] = compensate_store(I2, algo._inst, 
                                                       algo, algo.zcomp, 
                                                       model, store=store[jj]['extra'])

        # Correct the defocal images if I1 and I2 are belong to different
        # sources, which is determined by the (fieldX, field Y)
        I1, I2 = algo._applyI1I2mask_pupil(I1, I2)
        store[jj]['intra']['applyI1I2mask_pupil'] = I1.getImg()
        store[jj]['extra']['applyI1I2mask_pupil'] = I2.getImg()

        #  self.zc, self.West = self._solvePoissonEq(I1, I2, jj) 
        algo.zc, algo.West, store[jj]['both'] = _solvePoissonEq_store(algo, 
                                                    I1, I2, jj, 
                                                    store[jj]['both'])


        # Record/ calculate the Zk coefficient and wavefront
        if compMode == "zer":
            algo.converge[:, jj] = algo.zcomp + algo.zc

            xoSensor, yoSensor = algo._inst.getSensorCoorAnnular()
            algo.wcomp = algo.West + ZernikeAnnularEval(
                np.concatenate(([0, 0, 0], algo.zcomp[3:])),
                xoSensor,
                yoSensor,
                algo.getObsOfZernikes(),
            )
            
            store[jj]['both']['wcomp'] = algo.wcomp
            store[jj]['both']['West'] = algo.West

    else:
        # Once we run into caustic, stop here, results may be close to real
        # aberration.
        # Continuation may lead to disastrous results.
        algo.converge[:, jj] = algo.converge[:, jj - 1]

    # Record the coefficients of normal/ annular Zernike polynomials after
    # z4 in unit of nm
    algo.zer4UpNm = algo.converge[3:, jj] * 1e9

    store[jj]['zer4UpNm'] = algo.zer4UpNm
    
    # Status of iteration
    stopItr = False

    # Calculate the difference
    if jj > 0:
        diffZk = (
            np.sum(np.abs(algo.converge[:, jj] - algo.converge[:, jj - 1])) * 1e9
        )

        # Check the Status of iteration
        if diffZk < tol:
            stopItr = True

    # Update the current iteration time
    algo.currentItr += 1

    # Show the Zk coefficients in interger in each iteration
    if algo.debugLevel >= 2:
        print("itr = %d, z4-z%d" % (jj, algo.getNumOfZernikes()))
        print(np.rint(algo.zer4UpNm))

    return stopItr, store
         
    
def get_butler_stamps(repoDir,instrument='LSSTComCam', iterN=0, detector="R22_S01",
                     dataset_type = 'donutStampsExtra', collection=''):
    
    butler = dafButler.Butler(repoDir)
    registry = butler.registry
    if collection == '':
        collection=f'ts_phosim_90060{iterN}1'
    dataId0 = dict(instrument=instrument)
    dataset = next(iter(butler.registry.queryDatasets(
                            datasetType='postISRCCD',
                            dataId=dataId0, 
                            collections=[collection]  )
                       ))
    expN = dataset.dataId["exposure"]
    # construct a dataId for zernikes and donut catalog:
    # switch exposure to visit 
    
    dataId = {'detector':detector, 'instrument':instrument,
              'visit':expN}

    donutStamps = butler.get(dataset_type, 
                              dataId=dataId, 
                              collections=[collection])  
    
    donutCatalog = butler.get('donutCatalog', 
                              dataId=dataId, 
                              collections=[collection]) 
    return donutStamps, donutCatalog

def get_butler_image(repoDir,instrument='LSSTComCam', iterN=0, detector="R22_S01",
                     collection=''):
    butler = dafButler.Butler(repoDir)
    registry = butler.registry
    if collection == '':
        collection=f'ts_phosim_90060{iterN}1'
    dataId0 = dict(instrument=instrument)
    dataset = next(iter(butler.registry.queryDatasets(
                            datasetType='postISRCCD',
                            dataId=dataId0,
                            collections=[collection]  )
                       ))
    expN = dataset.dataId["exposure"]
    dataId = {'detector':detector, 'instrument':instrument,
          'exposure':expN}
    postIsr = butler.get('postISRCCD',dataId=dataId,
                          collections=[collection])
    return postIsr

def get_comp_image(boundaryT=1,
                   optical_model="offAxis",
                   maskScalingFactorLocal =1,
                   field_xy = (1.1,1.2),
                   donut_stamp_size = 160,
                   inst_name='comcam') :
    ''' Get compensable image for field location 
    
    Parameters:
    -----------
    boundaryT : int
        Extended boundary in pixel. It defines how far the computation mask
        extends beyond the pupil mask. And, in fft, it is also the width of
        Neuman boundary where the derivative of the wavefront is set to
        zero.
    optical_model : str
        Optical model. It can be "paraxial", "onAxis", or "offAxis".
    maskScalingFactorLocal : float
        Mask scaling factor (for fast beam) for local correction.
        
    field_xy : tuple or list
            Position of donut on the focal plane in degree (field x, field y).

    '''
    # make fake donut array so no data is needed
    x = np.linspace(-2,2,donut_stamp_size)
    y = np.linspace(-2,2,donut_stamp_size)
    xx, yy = np.meshgrid(x,y)
    z = np.exp(np.cos(5*xx)-np.sin(5*yy))
     
    # initialize compensable image 
    imgExtra = CompensableImage() 

    # this is like setup in WfEstimator()
    config_dir = getConfigDir()
    inst_dir = os.path.join(config_dir, "cwfs", "instData")
    instrument = Instrument()

    camType = getCamType(inst_name)
    defocalDisInMm = getDefocalDisInMm(inst_name)
    instrument.configFromFile(donut_stamp_size, camType)
    field_distance = np.sqrt(field_xy[0]**2.+field_xy[1]**2.)
    #print(field_distance)
    # this is similar to _makeCompensableImage() in donutStamp.py, 
    imgExtra.setImg(
                field_xy,
                DefocalType.Extra,
                image=z
            )
    imgExtra.makeMask(instrument, optical_model, boundaryT, maskScalingFactorLocal)
    
    return imgExtra

def plot_imageCoCenter(algo, store, 
                      I1,I2,
                      I1imgInit,I2imgInit,
                      I1shifts, I2shifts):
    ''' Plot the action performed by imageCoCenter'''

    fig,ax = plt.subplots(2,3,figsize=(12,6))

    dimOfDonut = algo._inst.dimOfDonutImg
    stampCenter = dimOfDonut / 2 + 0.5

    for row,I,imgInit, shift in zip([0,1],
                             [I1,I2],
                             [I1imgInit,I2imgInit],
                             [I1shifts, I2shifts]
                            ):
        ax[row,0].imshow(imgInit, origin='lower')
        ax[row,1].imshow(I.getImg(), origin='lower')
        ax[row,2].imshow(imgInit-I.getImg(), origin='lower')

        ax[row,2].plot([[stampCenter,stampCenter],[stampCenter+shift[0], stampCenter]],)

        # illustrate the shift with arrows
        x0 = stampCenter
        y0 = stampCenter
        dx = 10*shift[0]
        dy = 10*shift[1]
        # x-shift
        ax[row,2].quiver(x0,    y0, dx, 0,  scale=1, units="xy", scale_units="xy",
                         width=2.5, edgecolors='orange', color='red')
        # y-shift
        ax[row,2].quiver(x0+dx, y0,  0 ,dy, scale=1, units="xy", scale_units="xy",
                         width=2.5,  edgecolors='orange', color='red')
        ax[row,2].text(1.1, 0.45, f'dx,dy: {shift} [px]',
                       transform=ax[row,2].transAxes,
                       fontsize=19,)

    row=0
    ax[row,0].set_title('Initial')
    ax[row,1].set_title('Shifted')
    ax[row,2].set_title('Initial-shifted')

    ax[0,0].text(-0.6,0.45, 'I1', fontsize=19,transform=ax[0,0].transAxes)
    ax[1,0].text(-0.6,0.45, 'I2', fontsize=19,transform=ax[1,0].transAxes)
    
def plot_algo_steps(store,iterNum=0):

    titles = {}
    if iterNum == 0 : 
        titles['imgBeforeCocenter']='before coCenter'
        titles['imgAfterCocenter']='after coCenter'
        
    # for all other iterations including zeroth one 
    titles['imgBeforeCompensate']='before compensation'
    titles['show_lutxyp']='projection of image \ncoordinates onto pupil \n(show_lutxyp)'
    titles['show_lutxyp_extracted']='cleaned-up projection'
    titles['imgAfterRecenter']='recentered image'
    titles['lutIp']='interpolated image \n(lutIp)'
    titles['Jacobian']='Jacobian'
    titles['imgAfterCompensate']='after compensation'
    titles['applyI1I2mask_pupil']='apply common pupil mask'


    rows=len(titles.keys())
    fig,ax = plt.subplots(rows,2, figsize=(6,3*rows))
    ax[0,0].text(0.8,1.2,f'iteration {iterNum}',
                transform=ax[0,0].transAxes,
                fontsize=19)
    for defocal,col in zip(['intra', 'extra'],
                           [0,1]):
        print(defocal,col)
        row=0
        for key in titles.keys():
            ax[row,col].imshow(store[iterNum][defocal][key], origin='lower')
            row+=1 
    row=0
    for key in titles.keys():
        ax[row,1].text(1.05,0.5, titles[key], fontsize=19,transform=ax[row,1].transAxes)
        row += 1 
    ax[0,0].set_title('intra (I1)')
    ax[0,1].set_title('extra (I2)')
from matplotlib import rcParams 
rcParams['ytick.labelsize'] = 15
rcParams['xtick.labelsize'] = 15
rcParams['axes.labelsize'] = 20
rcParams['axes.linewidth'] = 2
rcParams['font.size'] = 15
rcParams['axes.titlesize'] = 18

# common functions for AOS analysis
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm 
from matplotlib.ticker import MaxNLocator
from mpl_toolkits.axes_grid1 import make_axes_locatable
from astropy.table import Table
from astropy.coordinates import SkyCoord
from astropy.visualization import ZScaleInterval
from lsst.daf import butler as dafButler
from lsst.ts.wep.paramReader import ParamReader

from lsst.ts.phosim.utils.ConvertZernikesToPsfWidth import (
    convertZernikesToPsfWidth,
    getPsfGradPerZernike,
    
    
)


# run the CWFS fit ...
from lsst import cwfs
from lsst.cwfs.algorithm import Algorithm as AlgorithmCwfs
from lsst.cwfs.image import Image
from lsst.cwfs.instrument import Instrument as InstrumentCwfs
from pathlib import Path


import numpy as np # standard package for doing calculations
import matplotlib.pyplot as plt # standard package for plotting

import batoid
import galsim

import sys
sys.path.append('/sdf/data/rubin/gpfs/ncsa-home/home/scichris/aos/wfsim/')
import wfsim

from lsst.ts.wep.utility import (
    getConfigDir,
    DonutTemplateType,
    DefocalType,
    CamType,
    getCamType,
    getDefocalDisInMm,
    CentroidFindType
)
from lsst.ts.wep.cwfs.instrument import Instrument
from lsst.ts.wep.cwfs.algorithm import Algorithm
from lsst.ts.wep.cwfs.compensableImage import CompensableImage

import numpy as np

# run the CWFS fit ...
from lsst import cwfs
from lsst.cwfs.algorithm import Algorithm as AlgorithmCwfs
from lsst.cwfs.image import Image
from lsst.cwfs.instrument import Instrument as InstrumentCwfs
from pathlib import Path




import numpy as np # standard package for doing calculations
import matplotlib.pyplot as plt # standard package for plotting

import batoid
import galsim

import sys
sys.path.append('/sdf/data/rubin/gpfs/ncsa-home/home/scichris/aos/wfsim/')
import wfsim


from lsst.obs.lsst import LsstCam
import lsst.afw.image as afwImage
from lsst.afw.geom import makeSkyWcs,  makeCdMatrix
import numpy as np
from astropy.time import Time
from lsst.obs.lsst import Latiss
import lsst

import galsim
import numpy as np





def makeWcs( rotationAngleInDeg=0, pixel_scale_asec=0.1, originPixelCoord=[0,0],
           originSkyCoordDeg=[0,0]):
    '''
    rotationAngleInDeg : float, rotation angle in degrees (default: 0)
    pixel_scale: float, how many arcseconds per pixel 
    '''
    orientation = rotationAngleInDeg* lsst.geom.degrees
    flipX = True
    scale = pixel_scale_asec * lsst.geom.arcseconds  # how many arcsec per pixel 
    cdMatrix = makeCdMatrix(scale=scale, orientation=orientation, flipX=flipX)

    
    #crpix - `lsst.geom.Point2D` -  pixel coordinate of the origin 
    #crpix = wcs.getPixelOrigin() yields 
    # Point2D(2088.0000000000005, 2005.9999999999998)
    crpix =  lsst.geom.Point2D(originPixelCoord)
    
    #crval  - `lsst.geom.SpherePoint` - desired reference sky position
    #crval = wcs.getSkyOrigin() yields 
    # SpherePoint(156.712774671942*degrees, -30.072990580516898*degrees)
    ra,dec = originSkyCoordDeg
    crval = lsst.geom.SpherePoint(ra*lsst.geom.degrees, dec*lsst.geom.degrees)
    newWcs = makeSkyWcs(crpix=crpix, crval=crval, cdMatrix=cdMatrix)
    return newWcs


def make_exposure(image_array=None, focusZ=0.8):
    ''' 
    This is only for Latiss... 
    '''
    camera = Latiss().getCamera()
    
    w,h = np.shape(image_array)
    image = afwImage.ImageF(width=w, height=h)
    image.array[:] = image_array.copy()# simulator.image.array.copy()
    
    #image = afwImage.ImageF(image_array)
    
    filt_name = 'r'
    expInfo = afwImage.ExposureInfo()
    inFilter = afwImage.FilterLabel(filt_name)
    expInfo.setFilter(inFilter)
    # add focusZ ... 
    visitInfo = lsst.afw.image.VisitInfo(focusZ=focusZ)
    expInfo.setVisitInfo(visitInfo)
    
    exp = afwImage.ExposureF(afwImage.MaskedImageF(image), expInfo)

    md = exp.getMetadata()
    md.set("CHIPID", 'RXX_S00')
    detector = camera.get('RXX_S00')
    exp.setDetector(detector)
    
    # add WCS 
    wcs = makeWcs()
    exp.setWcs(wcs)
    
    # Set place holder date
    md.set("MJD-OBS", Time.now().mjd)
    md.set("OBSTYPE", "simulation")
    # arbitrary for flats, non-arbitrary for science obs 
    md.set("EXPTIME", 30)
    # need to be able to specify any filter#
    md.set("CALDATE", Time.now().mjd)
    
    
    # example LATISS keys are 
    # VisitInfo(exposureId=2022040700156, exposureTime=30, darkTime=30.315, 
    # date=2022-04-08T01:08:44.669053693, UT1=nan, ERA=3.71809 rad, 
    # boresightRaDec=(156.7127746719, -30.0729905805), 
    # boresightAzAlt=(93.4631537688, +77.5489299579), 
    # boresightAirmass=1.02402, boresightRotAngle=3.29551 rad, 
    # rotType=1, observatory=-30.2446N, -70.7494E  2663, 
    # weather=Weather(11.9, nan, 20), instrumentLabel='LATISS', 
    # id=2022040700156, focusZ=0.770146, observationType='engtest', 
    # scienceProgram='CWFS', observationReason='intra', 
    # object='HD  90557', hasSimulatedContent=false)
    
    # http://doxygen.lsst.codes/stack/doxygen/x_mainDoxyDoc/_visit_info_8cc_source.html#l00227
    #visitInfo = lsst.afw.image.VisitInfo(focusZ=focusZ)
    #exp.getInfo().setVisitInfo(visitInfo)

    
    return exp



def simulate_defocal_atmosphere(seed = 10, image_shape_x=1000,image_shape_y=1000):
    ''' 
    Simulate auxTel atmosphere only, one intra and one extra-focal simulator.
    
    This is a workaround to allow simulating different zernikes, but 
    with the same telescope setup and atmosphere.
    
    For some reason, trying to re-simulate the atmosphere & zks in a loop
    fails (as of 2/1/2023) -the simulation never finishes. 
    
    
    '''

    auxtel = batoid.Optic.fromYaml("AuxTel.yaml")
    bandpass = galsim.Bandpass("LSST_r.dat", wave_type='nm')
    rng = np.random.default_rng(seed)

    # parameters associated with the observing conditions
    obs_params = {
        # zenith=0 means straight up, while zenith=90 is the horizon
        # greater zenith angle means more atmospheric turbulence
        "zenith": 30 * galsim.degrees,

        # raw_seeing sets how much atmospheric turbulence there is at zenith=0
        # wfsim uses this to calculate how much turbulence there is at 
        # the zenith angle given above
        "raw_seeing": 0.7 * galsim.arcsec,

        # these next three parameters set attributes of the atmosphere, which
        # help determine how much turbulence there is
        "temperature": 293, # Kelvin
        "pressure": 69, # kPa
        "H2O_pressure": 1, # kPa

        # wavelength of the observed light. this is important because light of 
        # longer/shorter wavelength refracts less/more in the atmosphere
        "wavelength": bandpass.effective_wavelength, # nanometers

        # the AuxTel exposure time
        "exptime": 30, # seconds
    }

    # parameters associated with computational aspects of simulating the atmosphere
    atm_params = {
        "screen_size": 819.2,
        "screen_scale": 0.1,
        "nproc": 3,
        }

    intra = auxtel.withGloballyShiftedOptic(
        "M2", [0, 0, -0.0008]) # meters 
        
    extra = auxtel.withGloballyShiftedOptic(
        "M2", [0, 0, +0.0008])
        
    # Create an intra-focal simulator with just atmosphere:
    intra_simulator = wfsim.SimpleSimulator(
        obs_params,
        atm_params,
        intra,
        bandpass,
        shape=(image_shape_x, image_shape_y),
        rng=rng
    )
        
    # Create an extra-focal simulator with just atmosphere:
    extra_simulator = wfsim.SimpleSimulator(
        obs_params,
        atm_params,
        extra,
        bandpass,
        shape=(image_shape_x, image_shape_y),
        rng=rng
    )
        
    return intra_simulator, extra_simulator 

def simulate_zernike_screen(intra_simulator, extra_simulator, 
                            z_in = np.array([0, 0, 0, 0, 0, 0, 0, 0,0, 0, 300e-9, 0]),
                            thxDeg=0, thyDeg=39/3600,
                            outDir = 'DM-37629',
                            fname = f"auxTel_batoid_test_zk"
                           ):
    # since the wavelength is 622 nm, 
    # +/- 200 nm corresponds to +/- 0.3 waves of perturbation
    #z_in = rng.uniform(-200e-9, 200e-9, size=12)
    ''' 
    Simulate auxTel donuts with input Zernike wavefront as phase screen.
    
    NB: 
    z_in contains z0,z1,z2....z11   
    so eg. z_in = [0,0,0,200e-9,0,0...] means z3=200e-9 [m],
    i.e. 200 nm 
    
    NB: z_in must be in meters! 
    So  eg. result of ts_wep fit, which is in nanometers, 
    needs to be converted to meters ! 
    z_meters = z_nm  * 1e-9 

    thxDeg : x-position of a star in degrees 
    thyDeg : y-position of a star in degrees 
    
    '''

    auxtel = batoid.Optic.fromYaml("AuxTel.yaml")
    phase = batoid.Zernike(
        np.array(z_in),
        R_outer=0.6,
        R_inner=0.2115
    )
    perturbed_telescope = batoid.CompoundOptic(
                (
                batoid.optic.OPDScreen(
                    batoid.Plane(),
                    phase,
                    name='PhaseScreen',
                    obscuration=batoid.ObscNegation(batoid.ObscCircle(5.0)),
                    coordSys=auxtel.stopSurface.coordSys
                ),
                *auxtel.items
            ),
            name='PerturbedAuxTel',
            backDist=auxtel.backDist,
            pupilSize=auxtel.pupilSize,
            inMedium=auxtel.inMedium,
            stopSurface=auxtel.stopSurface,
            sphereRadius=auxtel.sphereRadius,
            pupilObscuration=auxtel.pupilObscuration
        )

    # set the image simulator with the perturbed telescope 
    intra_perturbed_telescope = perturbed_telescope.withGloballyShiftedOptic("M2", [0, 0, -0.0008])    
    intra_simulator.telescope = intra_perturbed_telescope
    intra_simulator.image.setZero()

    
    extra_perturbed_telescope = perturbed_telescope.withGloballyShiftedOptic("M2", [0, 0, +0.0008])   
    extra_simulator.telescope = extra_perturbed_telescope
    extra_simulator.image.setZero()
    
    # Now run the image simulation:
    wavelength  =  intra_simulator.bandpass.effective_wavelength * 1e-9 # batoid wants meters 

    # choose a position for the star
    # these angles specify the angle from the optical axis to the star
    thx = np.deg2rad(thxDeg)
    thy = np.deg2rad(thyDeg)

    # NB: read out the zernikes BEFORE the telescope is shifted by +/- 8 mm ! 
    zk = batoid.zernike(perturbed_telescope, thx, thy, wavelength, eps=0.3525)
    zk *= wavelength  #  waves -> meters 
    
    
    # create a spectrum for the star -- this is needed for chromatic refraction
    # we will randomly select a plausible star temperature, 
    # and calculate the corresponding blackbody spectrum
    rng = intra_simulator.rng
    star_temp = rng.uniform(4_000, 10_000) # Kelvin
    sed = wfsim.BBSED(star_temp) # wfsim has a function to calculate the blackbody spectrum

    # randomly choose a flux (must be an integer)
    flux = 1800000  #rng.integers(1_000_000, 2_000_000)

    intra_simulator.add_star(thx, thy, sed, flux, rng)
    extra_simulator.add_star(thx, thy, sed, flux, rng)
    
    fpath = os.path.join(outDir,fname)
    np.savez(fpath,
             intra=intra_simulator.image.array,
             extra=extra_simulator.image.array,
             zk=zk,
            )
    print(f'saved as {fname}')

    return 



def simulate_auxtel_zernikes(seed = 0, 
                             z_in = [0,0,0,0,200e-9,0,0,0,0,0,0,0],
                             fname = "auxTel_batoid_.npy", outDir = 'DM-37269',
                            image_shape_x=4000, image_shape_y=4000,
                            thxDeg=0, thyDeg=0):
    ''' 
    Simulate auxTel donuts with input Zernike wavefront as phase screen.
    
    NB: 
    z_in contains z0,z1,z2....z11   
    so eg. z_in = [0,0,0,200e-9,0,0...] means z3=200e-9 [m],
    i.e. 200 nm 
    
    NB: z_in must be in meters! 
    So  eg. result of ts_wep fit, which is in nanometers, 
    needs to be converted to meters ! 
    z_meters = z_nm  * 1e-9 


    
    '''

    auxtel = batoid.Optic.fromYaml("AuxTel.yaml")
    bandpass = galsim.Bandpass("LSST_r.dat", wave_type='nm')
    rng = np.random.default_rng(seed)

    # parameters associated with the observing conditions
    obs_params = {
        # zenith=0 means straight up, while zenith=90 is the horizon
        # greater zenith angle means more atmospheric turbulence
        "zenith": 30 * galsim.degrees,

        # raw_seeing sets how much atmospheric turbulence there is at zenith=0
        # wfsim uses this to calculate how much turbulence there is at 
        # the zenith angle given above
        "raw_seeing": 0.7 * galsim.arcsec,

        # these next three parameters set attributes of the atmosphere, which
        # help determine how much turbulence there is
        "temperature": 293, # Kelvin
        "pressure": 69, # kPa
        "H2O_pressure": 1, # kPa

        # wavelength of the observed light. this is important because light of 
        # longer/shorter wavelength refracts less/more in the atmosphere
        "wavelength": bandpass.effective_wavelength, # nanometers

        # the AuxTel exposure time
        "exptime": 30, # seconds
    }

    # parameters associated with computational aspects of simulating the atmosphere
    atm_params = {
        "screen_size": 819.2,
        "screen_scale": 0.1,
        "nproc": 3,
    }

    # since the wavelength is 622 nm, 
    # +/- 200 nm corresponds to +/- 0.3 waves of perturbation
    #z_in = rng.uniform(-200e-9, 200e-9, size=12)
    
    #z_in = np.zeros(12)
    #z_in[5] = 200e-9
    phase = batoid.Zernike(
        np.array(z_in),
        R_outer=0.6,
        R_inner=0.2115
        
    )
    perturbed = batoid.CompoundOptic(
            (
            batoid.optic.OPDScreen(
                batoid.Plane(),
                phase,
                name='PhaseScreen',
                obscuration=batoid.ObscNegation(batoid.ObscCircle(5.0)),
                coordSys=auxtel.stopSurface.coordSys
            ),
            *auxtel.items
        ),
        name='PerturbedAuxTel',
        backDist=auxtel.backDist,
        pupilSize=auxtel.pupilSize,
        inMedium=auxtel.inMedium,
        stopSurface=auxtel.stopSurface,
        sphereRadius=auxtel.sphereRadius,
        pupilObscuration=auxtel.pupilObscuration
    )


    intra = perturbed.withGloballyShiftedOptic("M2", [0, 0, -0.0008])
    extra = perturbed.withGloballyShiftedOptic("M2", [0, 0, +0.0008])


    # start the simulator 
    intra_simulator = wfsim.SimpleSimulator(
        obs_params,
        atm_params,
        intra,
        bandpass,
        shape=(image_shape_x, image_shape_y),
        rng=rng
    )

    extra_simulator = wfsim.SimpleSimulator(
        obs_params,
        atm_params,
        extra,
        bandpass,
        shape=(image_shape_x, image_shape_y),
        rng=rng
    )

    wavelength  =  bandpass.effective_wavelength * 1e-9 # batoid wants meters 
    # the fractional inner radius is 
    # eps = inner_radius / outer_radius = 0.2538 / 0.6 =  0.423 [m]


    # choose a position for the star
    # these angles specify the angle from the optical axis to the star
    thx = np.deg2rad(thxDeg)
    thy = np.deg2rad(thyDeg)


    zs = batoid.zernike(perturbed, thx, thy, wavelength, eps=0.3525)
    zs*=wavelength  #  waves -> m 
    
    # create a spectrum for the star -- this is needed for chromatic refraction
    # we will randomly select a plausible star temperature, 
    # and calculate the corresponding blackbody spectrum
    star_temp = rng.uniform(4_000, 10_000) # Kelvin
    sed = wfsim.BBSED(star_temp) # wfsim has a function to calculate the blackbody spectrum

    # randomly choose a flux (must be an integer)
    flux = 1800000#rng.integers(1_000_000, 2_000_000)

    intra_simulator.add_star(thx, thy, sed, flux, rng)
    extra_simulator.add_star(thx, thy, sed, flux, rng)

    print(f'saved {fname}')
    fpath = os.path.join(outDir,fname)
    np.savez(fpath,
        intra=intra_simulator.image.array,
        extra=extra_simulator.image.array, 
        zktruth=zs)
    return intra_simulator.image.array, extra_simulator.image.array, zs




def fit_cwfs(intra_stamp, extra_stamp, side=120, fieldXYIntra = [0.0, 0.0], fieldXYExtra=[0.0, 0.0],
            opticalModel="onAxis", obscuration=0.3525, _dz =  0.8):
    '''
     There is  "defocal offset" with a value of 32.8 mm  from 
     https://github.com/lsst-ts/ts_wep/blob/develop/policy/cwfs/instData/auxTel/instParamPipeConfig.yaml
     
     The value of _dz=0.8  is used in latiss_base_align.py;
     if I set _dz = 0.8, then   0.8*0.041  = 32.8 * 1e-3 , i.e. same as in ts_wep config ....
     
     Originally the cwfs param has:

    #Auxiliary Telescope parameters:
    Obscuration                             0.3525
    Focal_length (m)                        21.6
    Aperture_diameter (m)                   1.2
    Offset (m)                              0.0205
    Pixel_size (m)                          1.44e-5

    (from https://github.com/lsst-ts/cwfs/blob/master/data/AuxTel/AuxTel.param )

    '''
    # this value is in the latiss_base_align.py
    # if I set _dz = 0.8, then   0.8*0.041  = 32.8 * 1e-3 , i.e. same as in ts_wep config ....
    binning = 1  
    #side = 110  #  https://github.com/lsst-ts/ts_externalscripts/blob/8ba110cd64a134bf7d4efca424477c26deec96eb/tests/auxtel/test_latiss_cwfs_align.py#L525
    # from https://github.com/lsst-ts/ts_externalscripts/blob/8ba110cd64a134bf7d4efca424477c26deec96eb/python/lsst/ts/externalscripts/auxtel/latiss_base_align.py#L169

    #0.0205

    # Create configuration file with the proper parameters
    cwfs_config_template = f"""#Auxiliary Telescope parameters:
    Obscuration 				{obscuration}
    Focal_length (m)			21.6
    Aperture_diameter (m)   		1.2
    Offset (m)				{_dz * 0.041}
    Pixel_size (m)			{10e-6 *binning}
    """
    config_index = "auxtel_latiss"
    path = Path(cwfs.__file__).resolve().parents[3].joinpath("data", config_index)
    #print(path)
    if not path.exists():
        os.makedirs(path)
    dest = path.joinpath(f"{config_index}.param")
    with open(dest, "w") as fp:
        # Write the file and set the offset and pixel size parameters
        fp.write(cwfs_config_template)

    inst = InstrumentCwfs(config_index, int(side * 2 / binning))
    algo = AlgorithmCwfs("exp", inst, 1)

    # initialize intra and extra stamps 
    I1 = Image(intra_stamp, fieldXYIntra, Image.INTRA)
    I2 = Image(extra_stamp, fieldXYExtra, Image.EXTRA)


    algo.reset(I1,I2)
    algo.runIt(inst, I1, I2, opticalModel)
    zer4UpNm = algo.zer4UpNm

    # return the fit result
    return zer4UpNm




def fit_ts_wep(intra_stamp, extra_stamp, fieldXYIntra=np.zeros(2), 
               fieldXYExtra=np.zeros(2), sizeInPix = 240,
              opticalModel = 'onAxis'):
    #fieldXY = np.zeros(2)

    instName = 'auxTel'
    
     # here we make bigger stamps than the usual 200  # donut stamp size 
    configDir = getConfigDir()
    algoDir = os.path.join(configDir, "cwfs", "algo")
    tol=1e-3

    # this is part of the init
    inst = Instrument()
    algo = Algorithm(algoDir)

    # inside estimateZernikes()
    camType = getCamType(instName)

    # inside wfEsti.config
    inst.configFromFile(sizeInPix, camType)

    # choose the solver for the algorithm
    solver = 'exp' # by default
    debugLevel = 1 # 1 to 3
    algo.config(solver, inst, debugLevel=debugLevel)

    centroidFindType = CentroidFindType.RandomWalk
    imgIntra = CompensableImage(centroidFindType=centroidFindType)
    imgExtra = CompensableImage(centroidFindType=centroidFindType)

    #fieldXYIntra = fieldXY
    #fieldXYExtra = fieldXYIntra

    # now inside  `wfEsti.setImg` method,
    # which inherits from `CompensableImage`
    imgExtra.setImg(fieldXYExtra,
                    DefocalType.Extra,
                    image = extra_stamp)
  
    imgIntra.setImg(fieldXYIntra,
                    DefocalType.Intra,
                    image = intra_stamp)
    
    algo.runIt(imgIntra, imgExtra, opticalModel, tol=tol)

    zer4UpInNm = algo.getZer4UpInNm()
    
    return zer4UpInNm




def read_intra_extra(butler, expIdIntra, expIdExtra, 
                     datasetRefOrType = 'postISRCCD'):
    exposure_intra =  butler.get(datasetRefOrType, 
                                 dataId={'instrument':'LATISS', 'detector':0, 
                                     'exposure':expIdIntra}, 
                                )
    exposure_extra = butler.get(datasetRefOrType, 
                                dataId={'instrument':'LATISS', 'detector':0, 
                                     'exposure':expIdExtra}, 
                               )
    return exposure_intra, exposure_extra


def make_stamp(qm, exposure, side=120):
    result = qm.run(exposure, donutDiameter=2 * side)
    
    x,y = result.brightestObjCentroidCofM
    x = int(x)
    y = int(y)
    stamp = exposure.image.array[y-side:y+side, x-side:x+side]
    return stamp

def make_stamps(qm, intra_exposure, extra_exposure, side=120):
    
    # this is to avoid bright stars towards edges of exposures 
    # eg. 220-221,  222-223,
    # and bleeding edge in  263-264
    ycen=1900  
    xcen=2000
    w=1500


    result_intra = qm.run(intra_exposure[ycen-w:ycen+w, xcen-w:xcen+w], donutDiameter=2 * side)
    result_extra = qm.run(extra_exposure[ycen-w:ycen+w, xcen-w:xcen+w], donutDiameter=2 * side)
    
    #x,y = result.brightestObjCentroidCofM
    
    dy = (
        result_extra.brightestObjCentroidCofM[0]
        - result_intra.brightestObjCentroidCofM[0]
    )
    dx = (
        result_extra.brightestObjCentroidCofM[1]
        - result_intra.brightestObjCentroidCofM[1]
    )
    dr = np.sqrt(dy**2 + dx**2)

    position_out_of_range = dr > 100

    x,y = result_intra.brightestObjCentroidCofM
    x = int(x); y = int(y)
    stamp_intra = intra_exposure.image.array[y-side:y+side, x-side:x+side]
    
    if not position_out_of_range:
        # only use if the extra-focal position is not widely different
        x,y = result_extra.brightestObjCentroidCofM
        x = int(x); y = int(y)
    stamp_extra = extra_exposure.image.array[y-side:y+side, x-side:x+side]
    
    return stamp_intra, stamp_extra# , result

def preview_auxtel_pair(day=20210608, seqNums=[233,234], exps=None,
                            datasetRefOrType='raw', collections=['LATISS/raw/all',
                                                                 'LATISS/calib',
                                                                 'LATISS/runs/quickLook'],
                            repo='repo/main/',limits = None
                       ):

    butler = dafButler.Butler(os.path.join('/sdf/data/rubin/',repo),
                             collections=collections)
    
    # just a pair of donuts
    ncol=2
    nrows = 1

    zscale = ZScaleInterval()
    
    
    dataIds = []
    if exps is None:
        for seqNum in seqNums:
            dataId = {'day_obs': day, 'seq_num': seqNum, 'detector':0}
            dataIds.append(dataId)
    else:
        for exp in exps:
            dataId={'instrument':'LATISS', 'detector':0, 
                                         'exposure':exp}
            dataIds.append(dataId)
            
            
    # do the plotting 
    fig,axs = plt.subplots(nrows,ncol,figsize=(ncol*4,nrows*4))
    ax = np.ravel(axs)       
    for i in range(len(dataIds)): 
        dataId = dataIds[i]
        exposure = butler.get(datasetRefOrType, dataId=dataId, )#collections=collections)
        
        # get focusZ from exposureInfo rather than metadata...
        info = exposure.getInfo()
        visitInfo = info.getVisitInfo() 
        focusZ = visitInfo.focusZ
        expId = visitInfo.id
        if limits is None:
            data = exposure.image.array
        else:
            xcen,ycen,w = limits
            data = exposure.image.array[ycen-w:ycen+w, xcen-w:xcen+w]
        vmin, vmax = zscale.get_limits(data)
        ax[i].imshow(data,vmin=vmin,vmax=vmax,origin=
                  'lower')
        
        ax[i].set_title(f"{expId},\n focusz={np.round(focusZ,3)}")
        i += 1
    fig.subplots_adjust(hspace=0.25)

def preview_auxtel_exposures(year='2021', monthDay='0908', expStart=483, expEnd=490,
                            datasetRefOrType='raw', collection='LATISS/raw/all',
                            repo='repo/main/'):

    butler = dafButler.Butler(os.path.join('/sdf/data/rubin/',repo))
    
    # figure out how many images to plot
    nexp = expEnd-expStart

    # calculate how many cols and rows we need 
    if nexp > 3:
        ncol = 3
        nrows = (nexp // ncol) + 1
    else:
        ncol=nexp
        nrows = 1
        
    zscale = ZScaleInterval()
    # do the plotting 
    fig,axs = plt.subplots(nrows,ncol,figsize=(ncol*4,nrows*4))
    ax = np.ravel(axs)
    i=0
    for exp in range(expStart,expEnd):
        expN = str(exp).zfill(5)
        print(expN)
        
        
            
        exposure = butler.get(datasetRefOrType, dataId={'instrument':'LATISS', 'detector':0, 
                                         'exposure':int(f'{year}{monthDay}{expN}')
                                            },
                          collections=[collection])
        
        # get focusZ from exposureInfo rather than metadata...
        info = exposure.getInfo()
        visitInfo = info.getVisitInfo() 
        focusZ = visitInfo.focusZ
            
        data = exposure.image.array
        vmin, vmax = zscale.get_limits(data)
        ax[i].imshow(data,vmin=vmin,vmax=vmax,origin=
                  'lower')
        
        ax[i].set_title(f"{year}{monthDay}, exp {exp},\n focusz={np.round(focusZ,3)}")
        i += 1
    fig.subplots_adjust(hspace=0.25)

    # if there are more axes than exposures,
    # turn off the extra axes 
    ncells = nrows*ncol
    if ncells > nexp:
        for axis in ax[nexp:]:
            axis.axis("off")
            
            

def simulate_auxtel_zernikes(seed = 0, 
                             z_in = [0,0,0,0,200e-9,0,0,0,0,0,0,0],
                             fname = "auxTel_batoid_.npy", outDir = 'DM-37269',
                            image_shape_x=4000, image_shape_y=4000,
                            thxDeg=0, thyDeg=0):
    ''' 
    Simulate auxTel donuts with input Zernike wavefront as phase screen.
    
    NB: 
    z_in contains z0,z1,z2....z11   
    so eg. z_in = [0,0,0,200e-9,0,0...] means z3=200e-9 [m],
    i.e. 200 nm 
    
    NB: z_in must be in meters! 
    So  eg. result of ts_wep fit, which is in nanometers, 
    needs to be converted to meters ! 
    z_meters = z_nm  * 1e-9 


    
    '''

    auxtel = batoid.Optic.fromYaml("AuxTel.yaml")
    bandpass = galsim.Bandpass("LSST_r.dat", wave_type='nm')
    rng = np.random.default_rng(seed)

    # parameters associated with the observing conditions
    obs_params = {
        # zenith=0 means straight up, while zenith=90 is the horizon
        # greater zenith angle means more atmospheric turbulence
        "zenith": 30 * galsim.degrees,

        # raw_seeing sets how much atmospheric turbulence there is at zenith=0
        # wfsim uses this to calculate how much turbulence there is at 
        # the zenith angle given above
        "raw_seeing": 0.7 * galsim.arcsec,

        # these next three parameters set attributes of the atmosphere, which
        # help determine how much turbulence there is
        "temperature": 293, # Kelvin
        "pressure": 69, # kPa
        "H2O_pressure": 1, # kPa

        # wavelength of the observed light. this is important because light of 
        # longer/shorter wavelength refracts less/more in the atmosphere
        "wavelength": bandpass.effective_wavelength, # nanometers

        # the AuxTel exposure time
        "exptime": 30, # seconds
    }

    # parameters associated with computational aspects of simulating the atmosphere
    atm_params = {
        "screen_size": 819.2,
        "screen_scale": 0.1,
        "nproc": 3,
    }

    # since the wavelength is 622 nm, 
    # +/- 200 nm corresponds to +/- 0.3 waves of perturbation
    #z_in = rng.uniform(-200e-9, 200e-9, size=12)
    
    #z_in = np.zeros(12)
    #z_in[5] = 200e-9
    phase = batoid.Zernike(
        np.array(z_in),
        R_outer=0.6,
        R_inner=0.2115
        
    )
    perturbed = batoid.CompoundOptic(
            (
            batoid.optic.OPDScreen(
                batoid.Plane(),
                phase,
                name='PhaseScreen',
                obscuration=batoid.ObscNegation(batoid.ObscCircle(5.0)),
                coordSys=auxtel.stopSurface.coordSys
            ),
            *auxtel.items
        ),
        name='PerturbedAuxTel',
        backDist=auxtel.backDist,
        pupilSize=auxtel.pupilSize,
        inMedium=auxtel.inMedium,
        stopSurface=auxtel.stopSurface,
        sphereRadius=auxtel.sphereRadius,
        pupilObscuration=auxtel.pupilObscuration
    )


    intra = perturbed.withGloballyShiftedOptic("M2", [0, 0, -0.0008])
    extra = perturbed.withGloballyShiftedOptic("M2", [0, 0, +0.0008])


    # start the simulator 
    intra_simulator = wfsim.SimpleSimulator(
        obs_params,
        atm_params,
        intra,
        bandpass,
        shape=(image_shape_x, image_shape_y),
        rng=rng
    )

    extra_simulator = wfsim.SimpleSimulator(
        obs_params,
        atm_params,
        extra,
        bandpass,
        shape=(image_shape_x, image_shape_y),
        rng=rng
    )

    wavelength  =  bandpass.effective_wavelength * 1e-9 # batoid wants meters 
    # the fractional inner radius is 
    # eps = inner_radius / outer_radius = 0.2538 / 0.6 =  0.423 [m]


    # choose a position for the star
    # these angles specify the angle from the optical axis to the star
    thx = np.deg2rad(thxDeg)
    thy = np.deg2rad(thyDeg)


    zs = batoid.zernike(perturbed, thx, thy, wavelength, eps=0.3525)
    zs*=wavelength  #  waves -> m 
    
    # create a spectrum for the star -- this is needed for chromatic refraction
    # we will randomly select a plausible star temperature, 
    # and calculate the corresponding blackbody spectrum
    star_temp = rng.uniform(4_000, 10_000) # Kelvin
    sed = wfsim.BBSED(star_temp) # wfsim has a function to calculate the blackbody spectrum

    # randomly choose a flux (must be an integer)
    flux = 1800000#rng.integers(1_000_000, 2_000_000)

    intra_simulator.add_star(thx, thy, sed, flux, rng)
    extra_simulator.add_star(thx, thy, sed, flux, rng)

    print(f'saved {fname}')
    fpath = os.path.join(outDir,fname)
    np.savez(fpath,
        intra=intra_simulator.image.array,
        extra=extra_simulator.image.array, 
        zktruth=zs)
    return intra_simulator.image.array, extra_simulator.image.array, zs



# def fit_cwfs(intra_stamp, extra_stamp, side=120, fieldXYIntra = [0.0, 0.0], fieldXYExtra=[0.0, 0.0],
#             opticalModel="onAxis", obscuration=0.3525, _dz =  0.8):
#     '''
#      There is  "defocal offset" with a value of 32.8 mm  from 
#      https://github.com/lsst-ts/ts_wep/blob/develop/policy/cwfs/instData/auxTel/instParamPipeConfig.yaml
     
#      The value of _dz=0.8  is used in latiss_base_align.py;
#      if I set _dz = 0.8, then   0.8*0.041  = 32.8 * 1e-3 , i.e. same as in ts_wep config ....
     
#      Originally the cwfs param has:

#     #Auxiliary Telescope parameters:
#     Obscuration                             0.3525
#     Focal_length (m)                        21.6
#     Aperture_diameter (m)                   1.2
#     Offset (m)                              0.0205
#     Pixel_size (m)                          1.44e-5

#     (from https://github.com/lsst-ts/cwfs/blob/master/data/AuxTel/AuxTel.param )

#     '''
#     # this value is in the latiss_base_align.py
#     # if I set _dz = 0.8, then   0.8*0.041  = 32.8 * 1e-3 , i.e. same as in ts_wep config ....
#     binning = 1  
#     #side = 110  #  https://github.com/lsst-ts/ts_externalscripts/blob/8ba110cd64a134bf7d4efca424477c26deec96eb/tests/auxtel/test_latiss_cwfs_align.py#L525
#     # from https://github.com/lsst-ts/ts_externalscripts/blob/8ba110cd64a134bf7d4efca424477c26deec96eb/python/lsst/ts/externalscripts/auxtel/latiss_base_align.py#L169

#     #0.0205

#     # Create configuration file with the proper parameters
#     cwfs_config_template = f"""#Auxiliary Telescope parameters:
#     Obscuration 				{obscuration}
#     Focal_length (m)			21.6
#     Aperture_diameter (m)   		1.2
#     Offset (m)				{_dz * 0.041}
#     Pixel_size (m)			{10e-6 *binning}
#     """
#     config_index = "auxtel_latiss"
#     path = Path(cwfs.__file__).resolve().parents[3].joinpath("data", config_index)
#     #print(path)
#     if not path.exists():
#         os.makedirs(path)
#     dest = path.joinpath(f"{config_index}.param")
#     with open(dest, "w") as fp:
#         # Write the file and set the offset and pixel size parameters
#         fp.write(cwfs_config_template)

#     inst = InstrumentCwfs(config_index, int(side * 2 / binning))
#     algo = AlgorithmCwfs("exp", inst, 1)

#     # initialize intra and extra stamps 
#     I1 = Image(intra_stamp, fieldXYIntra, Image.INTRA)
#     I2 = Image(extra_stamp, fieldXYExtra, Image.EXTRA)


#     algo.reset(I1,I2)
#     algo.runIt(inst, I1, I2, opticalModel)
#     zer4UpNm = algo.zer4UpNm

#     # return the fit result
#     return zer4UpNm




# def fit_ts_wep(intra_stamp, extra_stamp, fieldXYIntra=np.zeros(2), 
#                fieldXYExtra=np.zeros(2), sizeInPix = 240,
#               opticalModel = 'onAxis'):
#     #fieldXY = np.zeros(2)

#     instName = 'auxTel'
    
#      # here we make bigger stamps than the usual 200  # donut stamp size 
#     configDir = getConfigDir()
#     algoDir = os.path.join(configDir, "cwfs", "algo")
#     tol=1e-3

#     # this is part of the init
#     inst = Instrument()
#     algo = Algorithm(algoDir)

#     # inside estimateZernikes()
#     camType = getCamType(instName)

#     # inside wfEsti.config
#     inst.configFromFile(sizeInPix, camType)

#     # choose the solver for the algorithm
#     solver = 'exp' # by default
#     debugLevel = 1 # 1 to 3
#     algo.config(solver, inst, debugLevel=debugLevel)

#     centroidFindType = CentroidFindType.RandomWalk
#     imgIntra = CompensableImage(centroidFindType=centroidFindType)
#     imgExtra = CompensableImage(centroidFindType=centroidFindType)

#     #fieldXYIntra = fieldXY
#     #fieldXYExtra = fieldXYIntra

#     # now inside  `wfEsti.setImg` method,
#     # which inherits from `CompensableImage`
#     imgExtra.setImg(fieldXYExtra,
#                     DefocalType.Extra,
#                     image = extra_stamp)
  
#     imgIntra.setImg(fieldXYIntra,
#                     DefocalType.Intra,
#                     image = intra_stamp)
    
#     algo.runIt(imgIntra, imgExtra, opticalModel, tol=tol)

#     zer4UpInNm = algo.getZer4UpInNm()
    
#     return zer4UpInNm




def get_xy_from_yaml(get_actuators=True):
    '''
    Get x,y positions of either actuators, or grid space
    
    For locations of 156 axial actuators,
    use the default 
    
    gridFileName = 'M1M3_1um_156_force.yaml',
    mirrorDataDir = '/project/scichris/aos/ts_ofc/policy/M1M3/'
    
    
    For locations of 5256 grid points, use 
    gridFileName = 'M1M3_1um_156_grid.yaml'
    mirrorDataDir = '/project/scichris/aos/ts_phosim/policy/M1M3/'
    
    '''
    if get_actuators:
        print("Obtaining x,y locations for 156 axial actuators ")
        gridFileName = 'M1M3_1um_156_force.yaml'
        mirrorDataDir = '/project/scichris/aos/ts_ofc/policy/M1M3/'
    else:
        print("Obtaining x,y locations for 5256 grid locations ")
        gridFileName = 'M1M3_1um_156_grid.yaml'
        mirrorDataDir = '/project/scichris/aos/ts_phosim/policy/M1M3/'
        
    gridFile = ParamReader()
    
    gridFilePath = os.path.join(mirrorDataDir, gridFileName)
    print('from ', gridFilePath)
    gridFile.setFilePath(gridFilePath)
    daat = gridFile.getMatContent()

#     x=[]
#     y=[]
#     for row in daat:
#         x.append(row[1])
#         y.append(row[2])
        
    x = daat[:,1]
    y = daat[:,2]
    
    return x,y

def plotZernikeImage(repoDir = '/repo/main/',
                       collection='u/scichris/Latiss/test',
                       instrument = 'LATISS',
                       detector = 0,
                       titleAdd=''
                       ):
    '''
    repoDir : str
        Path to a directory containing butler.yaml file. 
        For Latiss tests its /repo/main,  while for 
        imgCloseLoop runs it is based on a path to the output 
        directory+'phosimData' eg.
        "/project/scichris/aos/rotation_DM-31532/\
         Ns_rotCam_30_c/phosimData/"
    collections: str
        Collection name containing the following dataTypes:
        postISRCCD, zernikeEstimateRaw, donutCatalog
    
        We assume just one collection containing all data.
        For Latiss tests, collection is u/scichris/Latiss/test
        
        For ts_phosim runs collections are usually 
        f'ts_phosim_90060{iterN}1', where "iterN" is iteration
        number.
    instrument: str
        Name of instrument for which the imgCloseLoop was run,
        eg. LsstCam, LsstComCam, or LATISS 
    detector: str or int 
        Name of detector. For LATISS it is "0", for 
        LsstComCam eg. "R22_S10"; must match the collection name,
        i.e. 
    titleAdd: str
        A title to be added to the default main figure title
        
    Notes: 
    
    example call to imgCloseLoop:
    
    python /project/scichris/aos/ts_phosim/bin.src/imgCloseLoop.py \
    --inst comcam --numOfProc 20 --boresightDeg 0.03 -0.02 --rotCam 0 \
    --skyFile /project/scichris/aos/rotation_DM-31532/comCam_grid.txt \
    --output /project/scichris/aos/rotation_DM-31532/Ns_rotCam_0/
    
    example call to pipetask (running ISR, donut detection, and 
    zernike estimation on a pair of auxTel exposures)
    
    pipetask run -d "exposure IN (2021090800487..2021090800488)  \
    AND instrument='LATISS' AND visit_system=0" \
    -b /repo/main/butler.yaml -i LATISS/raw/all,LATISS/calib  \
    -o u/scichris/Latiss/test \
    -p /project/scichris/aos/ts_wep/tests/testData/\
    pipelineConfigs/testLatissPipeline.yaml --register-dataset-types 

    '''

    # read in the data from the butler 
    butler = dafButler.Butler(repoDir)
    registry = butler.registry

    
    
    dataId0 = dict(instrument=instrument)
    dataset = next(iter(butler.registry.queryDatasets(
                        datasetType='postISRCCD',
                        dataId=dataId0, 
                        collections=[collection]  )))

    expN = dataset.dataId["exposure"]
    
    # construct a dataId  for postISR 
    dataId = {'detector':detector, 'instrument':instrument,
              'exposure':expN}
    print(dataId)
    # read the postISR exposure 
    postIsrExp = butler.get('postISRCCD', dataId,
               collections=[collection])

    # construct a dataId for zernikes and donut catalog:
    # switch exposure to visit 
    dataId = {'detector':detector, 'instrument':instrument,
              'visit':expN}
    print(dataId)
    # the raw Zernikes 
    zkRaw =  butler.get('zernikeEstimateRaw', dataId=dataId,
                        collections=[collection])

    # the donut source catalog 
    srcCat= butler.get('donutCatalog', dataId=dataId, collections=[collection])

    # since we queried by detector, sources in that catalog are 
    # only for that detector., and its a pandas Df 
    exposureName = postIsrExp.getDetector().getName()
    #expCatalog = srcCat.query(f'detector == "{exposureName}"')

    # plot the figure ...   
    fig = plt.figure(figsize=(14, 5))

    ####################################
    ### left - plot the fit results  ###
    #################################### 

    #add_axes([xmin,ymin,dx,dy]) 
    ax1 = fig.add_axes([0,0,0.6,1])  

    for i in range(len(zkRaw)):
        ax1.plot(np.arange(4, 23),1000*zkRaw[i], 
                 '-d', label=f'donut {i}')

    ax1.set_xlabel('Zernike Number',)
    ax1.set_ylabel('Zernike Coefficient [nanometers]', )
    ax1.legend(fontsize=14, loc='center left', bbox_to_anchor=[.65, 0.65])
    ax1.set_xticks(np.arange(4,23)[::2])
    ax1.grid()

    ax1.set_title(f'{instrument} {collection} {titleAdd}', fontsize=18)


    ###############################################
    ### right - plot the postISR image ###
    ###############################################


    ax2 = fig.add_axes([0.6,0,0.4,1])
    exposure_intra = postIsrExp
    zscale = ZScaleInterval()
    data = exposure_intra.image.array
    vmin, vmax = zscale.get_limits(data)

    ax2.imshow(data, origin='lower', vmin=vmin, vmax=vmax)

    nrows = len(srcCat)

    xs  = list(srcCat.centroid_x)
    ys = list(srcCat.centroid_y)
    for i in range(nrows):

        x = xs[i]
        y = ys[i]

        # plot the cross marking that the donut was used 
        ax2.scatter(x,y,s=200,marker='+',c='m', lw=4)

        # plot the donut number on the plot 
        xtext,ytext = x,y
        ytext -= 60
        if xtext+100 > 4096:
            xtext -= 250
        if len(str(i))>1: # move to the left label thats too long 
            #print(i, 'moving')
            xtext -=340 
        else:
            xtext -=260
        ax2.text(xtext, ytext, f'{i}', fontsize=17, c='white' )    
    ax2.yaxis.tick_right()
    ax2.set_xlabel('x [px]')
    ax2.set_ylabel('y [px]')
    ax2.yaxis.set_label_position("right")
    ax2.set_title(f'{exposureName}')
    
    plt.show()
    
    # plot donuts on a separate figure 
    extraFocalStamps = butler.get('donutStampsExtra', 
                              dataId=dataId, 
                              collections=[collection])
    nDonuts = len(extraFocalStamps)
    ncols=4
    nrows= nDonuts//ncols
    if nrows*ncols<nDonuts:
        nrows+=1
    fig,axs = plt.subplots(nrows, ncols, figsize=(3*ncols, 3*nrows))
    ax = np.ravel(axs)
    for i in range(nDonuts):
        donut = extraFocalStamps[i]
        ax[i].imshow(donut.stamp_im.image.array, origin='lower')
        ax[i].text(80,80, f'{i}', fontsize=17, c='white')
    fig.subplots_adjust(hspace=0.35)  
    
    # if there are more axes than donuts,
    # turn off the extra axes 
    ncells = nrows*ncols
    if ncells > nDonuts:
        for axis in ax[nDonuts:]:
            axis.axis("off")
            

def plotIterationSummary(data_dir, iterNum=5, num_ccds=9, suptitle='', figname='1.png',
                         opdPssnFileName='PSSN.txt', 
                         plot_opd=True,
                         plot_rms_opd_zk=True,
                         plot_pssn=True, plot_fwhm=True,
                         plot_raw_zk = False, aos_fwhm_arcsec_level = 0.089,
                         plot_aos_fwhm_line = True, R_inner = 0.61,
                         plot_in_asec = True
                        ):
    '''Convenience function to make a 4-panel plot informing about the 
     convergence of AOS loop plotting :
     1) the OPD data in terms of Zernikes
     2) RMS WFS vs OPD in Zernikes
     3) PSSN as a function of iteration number 
     4) FWHM as a function of iteration number
     
    Parameters:
    ----------
    data_dir : str, a path to the AOS loop directory, eg. 
        '/epyc/users/suberlak/Commissioning/aos/ts_phosim/notebooks/analysis_scripts/baselineTestComCam_qbkg'
    iterNum : float, a number of iterations (usually 5)
    num_ccds : float, a number of CCDs (field positions) for OPD evaluation. 
        For comcam it's usually 9 (one field point per CCD), for lsstcam/lsstfamcam 
        it's 31 (field points scattered across the full array), for corner sensors it's 4 
    suptitle : str, a suptitle to apply to the figure
    figname : str, a filename to save the plot as 
    testLabel: str, a label to identify tests; by default in AOS loop '1', so that 
        we expect opd.zer.1 and wfs.zer.1   files to be present in iterX/img/ directory
        where X=0,1,2,3,4 etc. 
    R_inner : float, the fractional obscuration passed to Galsim.zernike.Zernike, 
        used in scaling with R_outer = 1 by default.  R_inner  = 0.61 for lsst,  
        0.3525 for auxTel, see
        https://github.com/lsst-ts/ts_wep/blob/develop/policy/cwfs/instData/auxTel/instParamPipeConfig.yaml
        https://github.com/lsst-ts/ts_wep/blob/develop/policy/cwfs/instData/lsst/instParamPipeConfig.yaml
        
    Returns:
    --------
    None
     
    '''
    # calculate waves to fwhm conversion 
    zDicList4_37 = []
    for j in range(4, 37):
        z = galsim.zernike.Zernike([0]*j+[1], R_inner=R_inner, ) # that's what Josh used earlier ... 
        # R_outer  = 1.0 by default, so if that's used, then
        # R_inner would be fractional, as in 
        # https://github.com/lsst-ts/ts_wep/blob/develop/policy/cwfs/instData/lsst/instParamPipeConfig.yaml
        zkAnnular = np.sqrt(np.sum(z.gradX.coef**2) + np.sum(z.gradY.coef**2))
        zDicList4_37.append(zkAnnular)
        #print(f"{j:>2}   {zkAnnular:>6.3f}")
    rms_coeff = zDicList4_37[:19] 

    # Read in the data 
    opdDataDic = {}
    wfsDataDic = {}
    pssn_data = []
    fwhm_data = []
    for iterCount in range(iterNum):
        # load the OPD data 
        opdZkFilePath = os.path.join(data_dir, f'iter{iterCount}/img/opd.zer')
        opdData = np.loadtxt(opdZkFilePath)
        opdDataDic[iterCount] = opdData
        
        
        # load the wavefront error data 
        wfsZkFilePath = os.path.join(data_dir, f'iter{iterCount}/img/wfs.zer')
        wfsData = np.loadtxt(wfsZkFilePath)
        
        wfsDataDic[iterCount] = wfsData

        # load the PSSN and FWHM data 
        pssn_filename = os.path.join(data_dir, 'iter%i' % iterCount, 
                                     'img/%s'%opdPssnFileName)
        pssn_file_array = np.genfromtxt(pssn_filename)
        pssn_data.append(pssn_file_array[0])
        fwhm_data.append(pssn_file_array[1])
    pssn_data = np.array(pssn_data)
    fwhm_data = np.array(fwhm_data)
    
    
    # define the plotting space

    # figure out how many panels are needed
    n_panels  = np.sum([plot_opd, plot_rms_opd_zk, plot_pssn, plot_fwhm, plot_raw_zk])
    n_cols = 2
    n_rows = n_panels // n_cols # no remainder division
    if n_rows*n_cols < n_panels:
        n_rows += 1 
     
    fig,axs = plt.subplots(n_rows,n_cols,figsize=(8*n_cols,5*n_rows))
    ax = np.ravel(axs)

    # 0: plot the OPD 

    # to compare the two, need  to somehow "average" the OPD ? 
    # plot the values of zernikes at different field points...
    k = 0 
    if plot_opd:
 
        for iterCount in range(iterNum):
            opdData = opdDataDic[iterCount]
            if plot_in_asec:
                #opd_fwhm = (1000*opdData/750.)*rms_coeff
                opd_fwhm = convertZernikesToPsfWidth(opdData)
                ax[k].plot(np.mean(opd_fwhm, axis=0), lw=3,ls='-' ,)
            else:
                ax[k].plot(1000*np.mean(opdData, axis=0), lw=3,ls='-' ,)
                
        ax[k].set_xlabel('Zernike Numer')

        if plot_in_asec: 
            ax[k].set_ylabel('OPD FWHM [asec]')
        else:
            ax[k].set_ylabel('Wavefront Error (OPD) [nm]')
        k += 1 
                       
    if plot_rms_opd_zk:
        if plot_in_asec : 
            
            # 1: plot Zernike wavefront errors  vs OPD 
            for iterCount in range(iterNum):
                wfsData = wfsDataDic[iterCount]
                opdData = opdDataDic[iterCount]
                
                #opd_fwhm = (1000*opdData/750.)*rms_coeff
                #wfsData_fwhm = (1000*wfsData/750.)*rms_coeff
                wfsData_fwhm = convertZernikesToPsfWidth(wfsData)
                opd_fwhm  = convertZernikesToPsfWidth(opdData)
                
                meanOpdData_fwhm = np.mean(opd_fwhm, axis=0)
                zernikeErrorsDiff_fwhm = np.sqrt((wfsData_fwhm - meanOpdData_fwhm)**2.)
                #zernikeErrors_fwhm = np.transpose(zernikeErrorsDiff_fwhm, axes=(1,0))
                zernikeRms_fwhm = np.sqrt(np.mean(np.square(zernikeErrorsDiff_fwhm), axis=0))
                #print(zernikeRms_fwhm)

                ax[k].plot(np.arange(19)+4, zernikeRms_fwhm, 
                                      '-o', lw=3, label='iter%d'%iterCount)
            ax[k].set_ylabel('RMS(WFS-OPD) FWHM [asec]', size=18)
    
        else:
            # 1: plot Zernike wavefront errors  vs OPD 
            for iterCount in range(iterNum):
                wfsData = wfsDataDic[iterCount]
                opdData = opdDataDic[iterCount]
                # do the difference with mean OPD ... 
                meanOpdData = np.mean(opdData, axis=0)
                zernikeErrorsDiff = np.sqrt((wfsData - meanOpdData)**2.)

                #zernikeErrors = np.transpose(zernikeErrorsDiff, axes=(1,0))

                zernikeRms = np.sqrt(np.mean(np.square(zernikeErrorsDiff), axis=0))

                ax[k].plot(np.arange(19)+4, 1000*zernikeRms, 
                              '-o', lw=3, label='iter%d'%iterCount)
            ax[k].set_ylabel('RMS(WFS-OPD) [nm]', size=18)

        ax[k].set_xlabel('Zernike Number', size=18)
        ax[k].legend(fontsize=16)
        ax[k].set_title('Zernike Errors ', size=18)
        k += 1 
                       
    # plot the raw Zk estimate
    if plot_raw_zk:
        for iterCount in range(iterNum):
            wfsData = wfsDataDic[iterCount]
            #opdData = opdDataDic[iterCount]
            # do the difference with mean OPD ... 
            #meanOpdData = np.mean(opdData, axis=0)
            #zernikeErrorsDiff = np.sqrt((wfsData - meanOpdData)**2.)
            #zernikeErrors = np.transpose(zernikeErrorsDiff, axes=(1,0))
            #zernikeRms = np.sqrt(np.mean(np.square(zernikeErrors), axis=1))
            meanZernikes = np.mean(wfsData, axis=0)
            ax[k].plot(np.arange(19)+4, 1000*meanZernikes, 
                          '-o', lw=3, label='iter%d'%iterCount)

        ax[k].set_xlabel('Zernike Number', size=18)
        ax[k].set_ylabel('Raw zernike estimate[nm]', size=18)

        ax[k].legend(fontsize=16)
        ax[k].set_title('Zernikes ', size=18)
        k+=1 
                       
    # 2: plot PSSN 
    if plot_pssn:
        for i in range(num_ccds):
            ax[k].plot(np.arange(iterNum), pssn_data[:,i], c='b', marker='x')
        ax[k].plot(np.arange(iterNum), pssn_data[:,num_ccds], lw=4, marker='+',
                 ms=20, markeredgewidth=5, c='r', label='GQ PSSN')
        ax[k].legend()
        ax[k].set_xlabel('Iteration')
        ax[k].set_ylabel('PSSN')
        ax[k].set_title('PSSN')
        k+=1

    # 3: plot the FWHM 
    if plot_fwhm:
        for i in range(num_ccds):
            ax[k].plot(np.arange(iterNum), fwhm_data[:,i], c='b', marker='x')
        ax[k].plot(np.arange(iterNum), fwhm_data[:,num_ccds], lw=4, marker='+',
                 ms=20, markeredgewidth=5, c='r', label='GQ FWHM_eff')
        if plot_aos_fwhm_line:
            ax[k].axhline(aos_fwhm_arcsec_level, c='orange', ls='--',lw=3, 
                          label=f'{aos_fwhm_arcsec_level} line')
        ax[k].legend()
        
        ax[k].set_xlabel('Iteration')
        ax[k].set_ylabel('FWHM_eff [asec] ')
        ax[k].set_title('FWHM_eff')
        k+=1

    # on all : turn on the grid and set the x-label to be on integers only
    # since we're plotting Zernikes and iteration, both of which are integers
    for i in range(len(ax)):
        ax[i].grid()
        ax[i].xaxis.set_major_locator(MaxNLocator(integer=True))
    fig.subplots_adjust(hspace=0.3)
    fig.suptitle(suptitle, fontsize=20)
    plt.savefig(figname, bbox_inches='tight')
    print('Saved fig as %s'%figname)


# Two convenience functions to define x,y coordinates of points
# that look like an arrow, and points that 
# trace the outline of the sensor 

def pixel_arrow(x_vertex=1500, y_vertex=3000, width=1100, 
                spacing=300, diag_spacing=200, xmin=0, xmax=2000,
                ymin=0, ymax=4072 , xy_offset = 1300 ,print_shape=True
               ):
    #x_vertex, y_vertex = 1500,3000  
    # width = 1100; spacing = 300
    xPx = np.zeros(0)
    yPx = np.zeros(0)
    # vertical part
    ys = np.arange(y_vertex-width,y_vertex, spacing )
    xs = x_vertex*np.ones_like(ys)
    print(xs,ys)
    xPx = np.append(xPx, xs)
    yPx = np.append(yPx, ys)


    # horizontal part 
    xh = np.arange(x_vertex-width,x_vertex, spacing)
    yh = y_vertex*np.ones_like(xh)
    print(xh, yh)
    xPx = np.append(xPx, xh)
    yPx = np.append(yPx, yh)


    # diagonal part:
    x_start, y_start = x_vertex-xy_offset, y_vertex-xy_offset

    a = (y_start-y_vertex)/(x_start-x_vertex)
    b = y_vertex-a*x_vertex
    print('y=%.2fx %.2f'%(a,b))

    #diag_spacing = 200
    xd = np.arange(x_start, x_vertex,diag_spacing)
    yd = a*xd+b
    print(xd,yd)
    xPx = np.append(xPx, xd)
    yPx = np.append(yPx, yd)

    # append vertex too 
    xPx = np.append(xPx, x_vertex)
    yPx = np.append(yPx, y_vertex)

    if print_shape:
        # plot what I expect on  a single WFS half-chip 
        fig,ax = plt.subplots(1,1,figsize=((4./2000)*xmax,(8./4072)*ymax))
        ax.scatter(xs,ys)
        ax.scatter(xh,yh)
        ax.scatter(xd,yd)
        ax.scatter(x_vertex,y_vertex)
        #xmin,xmax = 0,2000
        #ymin,ymax = 0,4072
        ax.set_xlim(xmin,xmax)
        ax.set_ylim(ymin,ymax)
        ax.grid()
        ax.set_xlabel('x [px]')
        ax.set_ylabel('y [px]')

        ax.scatter(xPx, yPx)
    print(xPx, yPx)
    return xPx, yPx 


def pixel_outline(xmin=0,  xmax=2000, ymin=0, ymax=4072, dx=100 , dy=100, off =15,
                 print_shape=True):

    x0,x1 = xmin,xmax
    y0,y1 = ymin,ymax

    # initialize as empty arrays 
    xPx = np.zeros(0)
    yPx = np.zeros(0)

    # bottom part
    x = np.arange(x0,x1,dx)
    y = np.zeros_like(x)
    xPx = np.append(xPx, x)
    yPx = np.append(yPx, y)

    # right 
    y = np.arange(y0,y1,dy)
    x = np.ones_like(y) * x1
    xPx = np.append(xPx, x)
    yPx = np.append(yPx, y)

    # top 
    x = np.arange(x0,x1,dx)
    y = np.ones_like(x)* y1
    xPx = np.append(xPx, x)
    yPx = np.append(yPx, y)

    # left 
    y = np.arange(y0,y1,dy)
    x = np.zeros_like(y)
    xPx = np.append(xPx, x)
    yPx = np.append(yPx, y)

    if print_shape:
        # plot what I expect on  a single WFS half-chip 
        fig,ax = plt.subplots(1,1,figsize=((4./2000)*xmax,(8./4072)*ymax))
        ax.scatter(xPx,yPx)
        ax.set_xlim(xmin-off,xmax+off)
        ax.set_ylim(ymin-off,ymax+off)
        ax.grid()
        ax.set_xlabel('x [px]')
        ax.set_ylabel('y [px]')
    return xPx, yPx




def getRaDecFromGaiaField(field='high'):
    ''' A helper function to translate the GAIA field name to
    field  ICRS  coordinates  in degrees 

    Parameters:
    ----------
    field : str, field name, default 'high'. Other possible 
        names include 'med', 'low', 'Baade', 'Pleiades'

    Returns:
    --------
    raInDeg, decInDeg : floats, field coordinates in degrees
    '''
    path_to_ts_phosim = '/astro/store/epyc/users/suberlak/Commissioning/aos/ts_phosim/'
    path_to_notebooks = 'notebooks/analysis_notebooks/'
    path_to_field_desc = os.path.join(path_to_ts_phosim,path_to_notebooks,
        'GAIA_DR2_Galactic_fields.txt' )
    gt = Table.read(path_to_field_desc, format='ascii')
    gaia_coords = SkyCoord(l=gt['l_deg'],b=gt['b_deg'], 
                       frame='galactic', unit='deg')
    # convert them to equatorial 
    gt['ra_deg']= gaia_coords.icrs.ra.deg
    gt['dec_deg'] = gaia_coords.icrs.dec.deg
    # return just the floats
    raInDeg = gt['ra_deg'][gt['name'] == field][0]
    decInDeg = gt['dec_deg'][gt['name'] == field][0]
    print('For this field, the raInDeg=%.3f, decInDeg=%.3f'%(raInDeg,decInDeg))
    return raInDeg, decInDeg

def readImage(data_dir, focalType = 'extra', obsId=None, raft = None,
              detector = None, detNum = None, verbose=True,
              data_id = None, rerun='run1', imgType = 'postISR', iterNum = 0):
    ''' A function to read the post ISR image for a given CCD (sensor) 
    using Butler (so it has to be a butler repository). 

    Parameters:
    -----------
    data_dir : the location of postISR data: a path that contains 
               /input/rerun/run1/postISRCCD/
    focalType: 'extra' or 'intra' 
    obsId: id corresponding to the focal type, if comcam, then read from a dictionary:
         {'intra':9006001,  'extra':9006002} , otherwise set manually 
    detector: ccd name, one of ['S00', 'S01', 'S02','S10', 'S11', 'S12', 'S20', 'S21', 'S22']. 
         'S00' by default 
    raft : str, by default 'R22'  (comcam)
    detNum : int, detector Id - by default it's read from a following dictionary for comCam
             {'S00':90, 'S01':91, 'S02':92, 'S10':93, 'S11':94, 'S12':95, 'S20':96, 'S21':97, 'S22':98}
    verbose : boolean, True by default  - whether to print info about the progress of reading images 
    rerun : str, by default run1, could be run2, etc.
    imgType:  postISR (by default),   or raw 
    iterNum : iteration number, 0 by default : translates to the obsId used from the 
            /input/rerun/run1/postISRCCD/
    
    Returns:
    --------
    image :  an NxM array with the CCD postISR image 

    '''
    # Read in the postISR image using the Butler 
    # if Butler args are no provided, attempting to 
    # guess based on the following:
    if data_id is None:
        try:
            assert (raft is not None) and (detector is not None)
            sensor = raft+'_'+detector 
        except AssertionError:
            print(assert_err)
            print('\n raft (eg. R22) is None, or detector (eg. S00) is None')
            return 

        # this applies to ComCam ...
        detNumDict = {'R22_S00':90, 'R22_S01':91, 'R22_S02':92,   # ComCam detector ids 
                          'R22_S10':93, 'R22_S11':94, 'R22_S12':95, 
                          'R22_S20':96, 'R22_S21':97, 'R22_S22':98,
                          'R00_S22':197,'R04_S20':204,    # WFS detector ids 
                          'R40_S02':209, 'R44_S00':216
                      }
        if not detNum:
            detNum = detNumDict[sensor]    
        # these are decided in baseComcamLoop.py or baseWfsLoop.py 
        obsIdDic = {'focal':int('90060%d0'%iterNum), 'extra':int('90060%d1'%iterNum),  
                    'intra':int('90060%d2'%iterNum)}
        if not obsId: # if not provided, reading it from a dict, based on the focal type
            obsId = obsIdDic[focalType]
        
        # assemble data_id arguments for Butler 
        data_id = {'visit': obsId, 'filter': 'g', 'raftName': raft, 
                   'detectorName': detector, 'detector': detNum
                  }
    else:
        print('Using provided data_id for Butler')
    print('data_id is')
    print(data_id)
    # Read each figure as a postage stamp, store data to an array 
    if imgType == 'postISR':
        repo_dir = os.path.join(data_dir, 'input/rerun/', rerun)
        
    elif imgType == 'raw':
        repo_dir = os.path.join(data_dir, 'input/')
        
        
    print('Reading %s images from the following repo_dir:'%imgType)
    print(repo_dir)
    
    butler = dafPersist.Butler(repo_dir)

    # show what keys are needed by the `postISRCCD` data type.... 
    # butler.getKeys('postISRCCD')
    # yields {'visit': int, 'filter': str,'raftName': str, 'detectorName': str, 'detector': int}
    butlerImgType = {'postISR':'postISRCCD', 'raw':'raw'}
    butlerImg = butlerImgType[imgType]
    post = butler.get(butlerImg, **data_id) 

    # store in a dictionary
    image = post.image.array

    if verbose: print('Done\n')
    
    return image 






def readCentroidInfo(data_dir, focalType='extra', raft='R22',detector='S00'):
    ''' Read the centroid file info 

    Parameters:
    ------------
    data_dir : the location of postISR data: a path that contains 
                   /input/rerun/run1/postISRCCD/
    focalType: 'extra' or 'intra' 
    
    Returns:
    ---------
    centroids : an astroPy Table with a list of centroids 
    centFlag:  a flag that is False if the file is not found , True  otherwise
    '''
    # read centroid info 
    centr_dir = os.path.join(data_dir, 'iter0','img',focalType)      
    print('Reading centroid files from %s'%centr_dir)
    print('The following files are available:')
    pattern = 'centroid_lsst_'
    word = '%s_%s'%(raft,detector)
    for x in os.listdir(centr_dir):
        if x.startswith(pattern): 
            print(x)
            loc = x.find(word)
            if loc>0:
                fname = x 
    print('Using %s '%fname)

    centroid = Table.read(centr_dir+'/'+fname, format='ascii')
    centFlag = True
   
    return centroid, centFlag    

def readPostageStars(postage_dir,fname='postagedonutStarsExtraIntra.txt'):
    '''
    Read the postage image stars catalog. 
    While the postage stamps are saved in 
    WepController.py, getDonutMap(), 
 
    the catalog is saved at the next stage, when 
    WepController.py, calcWfErr(), 
    calculates the wavefront error based on the donut map 

    So if there is an error in that stage, the catalog is 
    not made.


    '''
    try:
        postage = Table.read(os.path.join(postage_dir,fname), format='ascii')
        print('Reading info about postage-stamp images from %s'%fname)
        postFlag = True
    except FileNotFoundError as fnf_error:
        print(fnf_error)
        postFlag = False
        postage = None
    return postage, postFlag

# helper funtion for the colorbar 
# from https://joseph-long.com/writing/colorbars/
def colorbar(mappable,ax):
    last_axes = plt.gca()
    fig = ax.figure
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    cbar = fig.colorbar(mappable, cax=cax)
    plt.sca(last_axes)
    return cbar


def plotImage(image,ax=None, log=False, sensor='R22_S00', focalType='extra',
             postage=None,postFlag=False, centroid=None, centFlag=False, 
              Nstars=2, starMarker='redCross',starMarkerArgs=None,
             centMarkerArgs = None,centMarker='redCross',
             starLabelArgs=None, plotArgs=None, imgType='postISR',
             addColorbar = True, addTitle=False, title=''):
    ''' A function  to plot a CCD image

    Parameters:
    -----------
    image : an NxM image array with pixel values . 
        It will be transposed for plotting 
    ax : plt.axis,  an axis to plot the image on
    log : bool, False by default - whether to pot log(counts) or not 
    

    Returns:
    --------
    None

    '''
    
    if ax is None:
        fig, ax = plt.subplots(1,1,figsize=(10,10))
        
    # plot the image 
    if log : 
        
        plottable = np.log10(image.T)
        cbar_label = r'$\log_{10}(\mathrm{counts})$'
    else:
        plottable = image.T
        cbar_label = r'$\mathrm{counts}$'
    

    if plotArgs is None:
        plotArgs = {}
    img = ax.imshow(plottable, origin='lower', **plotArgs)
    if addColorbar : 
        cbar= colorbar(mappable=img, ax=ax)
        cbar.set_label(label=cbar_label, weight='normal', )
    ax.set_xlabel('x [px]')
    ax.set_ylabel('y [px]')
    if addTitle:
        if len(title)>1: # use the title from the provided string 
            ax.set_title(title)
        else: # the default title
            ax.set_title('%s image, sensor %s, %s-focal'%(imgType, sensor,focalType))

    # try to figure out how many stars there are from postage file 
    if postFlag: 
        m1 = postage['abbrevDetectorName'] == sensor
        m2 = postage['focalPlane'] == focalType
        mask = m1 * m2 
        Nstars = len(postage[mask])

    starMarkerArgDict = {'yellowSquare': {'marker':'s', 'markersize':40, 'markerfacecolor':'none',
                                          'markeredgewidth':2.5, 'markeredgecolor':'y',
                                         } ,
                        'redCross':{'marker':'+', 'markersize':10,'markeredgecolor':'red', 
                                    'markerfacecolor':'red', 'markeredgewidth':2,
                                   }
                        }
    if starMarkerArgs is None:
        starMarkerArgs = starMarkerArgDict[starMarker]
    if centMarkerArgs is None:
        centMarkerArgs = starMarkerArgDict[centMarker]
    if starLabelArgs  is None:
        starLabelArgs = {'fontsize':16, 'color':'white'}
   

    for i in range(Nstars):
        if postFlag:
            x,y = postage[mask]['xpos'][i], postage[mask]['ypos'][i]
            starId = postage[mask]['starId'][i]
            #print(starId,x,y)
            ax.plot(x,y,  **starMarkerArgs)
            ax.text(x-40, y-180, starId, **starLabelArgs)

        if centFlag:
            #get the centroid location for that star
            starCentroid = centroid[centroid['SourceID'].data.astype(int) == starId]
            xCen, yCen = starCentroid['AvgX'].data  , starCentroid['AvgY'].data 
            
            #plot it with a cross 
            ax.plot(xCen, yCen, **centMarkerArgs)

    plt.tight_layout()

    
    
    
def plotZernikesAndCCD(image, rmsErrors, sepInPerc=10, testLabel='sep', xlims=[1525,2025], ylims=[750,1250],
                      sensor = 'R22_S00', focalType='extra',savefig=True, magPrimary=16, mag=15):
    '''  Function to plot both rms zernike errors and CCD image as a two-panel plot,
    restricting the CCD image to show only the relevant donut 
    
    Parameters:
    -----------
    image: NxM array with CCD data to plot 
    rmsErrors : an array with size 
    
    Returns:
    -------
    None
    '''
    xmin,xmax = xlims[0], xlims[1]
    ymin,ymax = ylims[0], ylims[1]

    fig, ax = plt.subplots(1,2,figsize=(16,6))
    

    figtitle = 'img_AOS_'
    suptitle = '%s,   '%sensor
    
    if testLabel == 'sep':
        sepInRadii = sepInPercToRadii(sepInPerc)
        print(sepInRadii)
        suptitle += 'Star Sep=%.1f donut radii'%sepInRadii
        figtitle += 'singleAmpSep_'
        
    if testLabel == 'mag':
        suptitle += r'$\Delta =%d$ mag' % (magPrimary-mag)
        figtitle += 'singleAmpMag_'
        
    if testLabel == 'gaia':
        suptitle += 'GAIA DR2'
        figtitle += 'gaia_'
        
    figtitle += sensor+'_'+focalType+'_ZerCCD.png'
        
    if np.shape(rmsErrors)[0] == 19 : 
        ax[0].plot(np.arange(19)+4, rmsErrors, 
         '-o', lw=3, )# color = cmap(colors[i]))
    else:
        print('Need the Zernike rms errors to be an array with 19 elements')

    ax[0].set_xlabel('Zernike Number', size=18)
    ax[0].set_ylabel('RMS WFS vs OPD (microns)', size=18)  
    ax[0].set_title(suptitle,  size=18)

    # plot the postage stamp
    img = ax[1].imshow(np.log10(image[ymin:ymax, xmin:xmax]), vmin = 0.01,
                       cmap=cm.get_cmap('Greys'),origin='lower')     
    ax[1].set_xlabel('x [px]')
    ax[1].set_ylabel('y [px]')
    ax[1].set_title('postISR image')
    plt.tight_layout()
    if savefig:
        plt.savefig(figtitle,
                bbox_inches='tight', dpi=150)

    
def sepInPercToRadii(sepInPerc):
    ''' Function to convert separation in percentage of amplifier ra span
    to donut radius 
    
    Parameters:
    ----------
    sepInPerc : int - separation in percent (eg. 10)
    
    Returns:
    ---------
    sepInRadii : float  - separation in donut radii
    '''
                    
    yPxAmp = 2048 # in pixels
    donutPxRadius = 66 # px            
    sepInPx = sepInPerc*0.01*yPxAmp 
    sepInRadii = sepInPx / donutPxRadius
                    
    return sepInRadii             


def plotPostageStamps(postage_dir, sensor='R22_S00', focalType='extra', Nstars=3,
                      cbarX0Y0DxDy = [0.13, 0.06, 0.76, 0.01],
                      sepInPerc=3, testLabel=None,suptitle=None,
                      magPrimary=16, mag = 15, filename = None
                     ):

    
    
    imgTypes = ['singleSciImg','imgDeblend_full', 'imgDeblend_resized']
    print('Using postage images from %s'%postage_dir)
    
    if suptitle is None:
        suptitle = '%s %s-focal, '%(sensor,focalType)
    
    if testLabel == 'sep':
        sepInRadii = sepInPercToRadii(sepInPerc)
        suptitle += 'sep=%.1f donut radii'%sepInRadii

    elif testLabel == 'mag':
        suptitle += r'$\Delta =%d$ mag' % (magPrimary-mag)
            
    elif testLabel == 'gaia':
        suptitle += 'GAIA DR2'
        
        
    if filename is None:
        filename = 'img_AOS_'
        if testLabel == 'sep':
            filename += 'singleAmpSep_'

        elif testLabel == 'mag':
            filename += 'singleAmpMag_'

        elif testLabel == 'gaia':
            filename += 'gaia_'
            
        filename += sensor+'_'+focalType+'_postageImg.png'
        
    print('Searching in %s directory'%postage_dir)
    print('\nAvailable postage stamp images for sensor %s: '%sensor)
    available = {}
    for imgType in imgTypes:
        available[imgType] = []
        # filename pattern to test how many are available ...
        pattern = focalType+'_'+imgType
        # i.e. eg. 'extra_imgDeblend...'
        print('\nLooking for files that start with "%s" and contain "%s"...'%(pattern, sensor))
        for x in os.listdir(postage_dir):
            if x.startswith(pattern) and (x.rfind(sensor)>0):
                #print(x)
                available[imgType].append(x)
                
        #print summary of what we found
        Nfound =len(available[imgType]) 
        if  Nfound > 5:
            print('\nFound %d %s postage stamp images '%(Nfound, imgType))
            print('first 5: ')
            print(available[imgType][:5])
        else:
            print('\nFound %d %s postage stamp images '%(Nfound, imgType))
            print(available[imgType])


    # revise the number of stars available vs those requested ... 
    Navailable = min([len(available[key]) for key in available.keys()])
    if Nstars is None:
        Nstars = Navailable
    if Nstars > Nfound:
        print('Only found %d '%Nfound)
        Nstars = Navailable

    # start plotting
    fig,ax = plt.subplots(Nstars,len(imgTypes),figsize=(12,4*Nstars))
    
    # handle the case of only 1 star with postage stamp image 
    # artificially adding a row of ones,
    # so that the minimal shape of [row,col] is preserved 
    if Nstars<2: 
        ax = np.append(ax,[1,1,1])
        ax = ax.reshape(2,3)
    
    for col in range(len(imgTypes)): # columns : each imgType is one column 
        imgType = imgTypes[col]
        ax[0,col].set_title(imgTypes[col], fontsize=17)
        for row in range(Nstars): # Nstars   rows : one per star
            fname = available[imgType][row]
            word = 'id' ; loc = fname.find(word)
            start = loc+len(word)+1
            stop = loc+len(word)+2
            starId = fname[start:start+2]
            print('Loading %s'%fname)
            image = np.loadtxt(os.path.join(postage_dir,fname))
            if image.ndim == 2  :
                mappable = ax[row,col].imshow(image, origin='lower')
                ax[row,col].text(0.1,0.1,'star %d, id %s'%(row,starId) , 
                                 fontsize=17, color='white', 
                                 transform=ax[row,col].transAxes)
            else: 
                ax[row,col].remove()
                ax[row,col].text(0.2,0.5, 'image.ndim < 2 ',fontsize=15,
                            transform=ax[row,col].transAxes)
     
    # that's for horizontal cbar on the bottom 
    cbar_ax = fig.add_axes(cbarX0Y0DxDy)     #  (x0 ,y0  , dx,  dy )  
    cbar = fig.colorbar(mappable, cax = cbar_ax,  orientation='horizontal')                    
    cbar.set_label(label='counts',weight='normal', fontsize=17)

    fig.suptitle(suptitle, fontsize=17)
    plt.savefig(filename, 
                bbox_inches='tight', dpi=100)
    print('Saved as %s'%filename)
    
    
    
    
def make_healpix_table(r_max=24.5):
    ''' A convenience function 
    to read in the MAF simulation data,
    and given the limiting r magnitude, 
    return stellar density per healpixel,
    and the fraction of healpixels with higher
    density , together with ra,dec coord 
    of each healpixel. We use the constraint 
    r < r_max, with  65 magnitude bins between 
    15 and 28 mag every 0.2 mag. 
    
    '''
    # the data consists of 
    # data['starDensity'],  expressed as stars / sq. deg  ,  per pixel, per magnitude
    # data['bins'], defining the magnitude ranges for each of the 65 magnitude bins 
    # data['overMaxMask'], which tells where there are more than 1e6 stars 
    data = np.load('starDensity_r_nside_64.npz')


    # Cumulative number counts, units of stars/sq deg. Array at healpix locations
    # magnitude bins 
    mag_bins = data['bins'].copy()
    # pixels where there were so many  (1e6 ) stars some were skipped
    mask = data['overMaxMask'].copy()
    # in this simulation none were skipped : 
    # np.sum(mask) = 0

    # select only bins up to r_max - then selecting the final bin will 
    # give us the source count up to depth of r_max mag 
    bright_mag_idx, = np.where(mag_bins<r_max)
    print('Selecting only the source density up \
    to the depth of ', r_max, ' mag')
    faintest_mag_idx = bright_mag_idx[-1]

    # Since the data is already cumulative, just choose the  last bin: 
    # this will have the number of stars up to the faintest magnitude 
    # bin in a given  healpixel 
    starDensity_lt_245 = data['starDensity'][:,faintest_mag_idx]
    # len(starDensity_lt_245) = len(data['starDensity]) = 49142

    # Generate the ra, dec array from healpy
    nside = hp.npix2nside(np.size(mask))
    lat,ra = hp.pix2ang(nside, np.arange(np.size(mask)))
    dec = np.pi/2-lat

    # only select those healpixels for which we have any simulation data ...
    m = starDensity_lt_245 > 0

    density = starDensity_lt_245[m]
    ra = ra[m]
    dec = dec[m]

    # For each pixel calculate how many pixels have a higher or equal density 
    N_px_greater  = np.zeros_like(density)
    for i in range(len(density)):
        N_px_greater[i]=np.sum(density>=density[i])

    # calculate the fraction of pixels that have a higher density (by area)
    frac_greater  = N_px_greater /  len(density)

    # Make an AstroPy table with healpix data...

    healpix_table = Table([density, ra,dec, N_px_greater, frac_greater], 
                          names=('source_density','ra_rad','dec_rad', 'N_px_greater', 
                                 'frac_greater'))
    return healpix_table , nside
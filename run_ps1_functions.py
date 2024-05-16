import subprocess
import os
import argparse
import time
from astropy.table import Table
import numpy as np 

from lsst.obs.base import createInitialSkyWcsFromBoresight
from lsst.afw.cameraGeom import FOCAL_PLANE, PIXELS
from lsst.obs.lsst import LsstCam, LsstComCam
import lsst.geom


from lsst.daf import butler as dafButler
import lsst.afw.cameraGeom.utils as cameraGeomUtils
import lsst.afw.display as afwDisplay
from astropy.visualization import ZScaleInterval

import matplotlib.pyplot as plt
from matplotlib import rcParams 
import pandas as pd
# this is a place to put all functions 
# used by at least two of the 
# ps1-simulation suite,
# including 
# run_ps1_1_catalogs
# run_ps1_phosim
# run_ps1_3_butler
# run_ps1_slurm


def plot_butler_with_wcs(repo_dir, detector='R22_S10', 
                         inst_file ='/project/scichris/aos/images/lsstCam/lsstCam_entire_tiled_LSST.inst',
                        xmax=4000, ymax=4000, coords='CCS',
                        collections=['run1'],
                        xsize=12, ysize=12):
    zscale = ZScaleInterval()

    instrument = 'LSSTCam'
    #repo_dir = '/project/scichris/aos/images/lsstCam/letter_lsstCam_entire_tiled_LSST_all_rot/DATA/'
    butler = dafButler.Butler(repo_dir)#,collections=collections)# collections=[f'{instrument}/raw/all',
                                       #              f'{instrument}/calib','run1',
                                       #              f'{instrument}/calib/unbounded'],
                                 #)
    dataId0 = dict(instrument=instrument)
    dataset = next(iter(butler.registry.queryDatasets(
                            datasetType='postISRCCD',
                            dataId=dataId0,
                            collections=collections )
                       ))
    expN = dataset.dataId["exposure"]
    dataId = {'detector':detector, 'instrument':instrument,
          'exposure':expN}

    exposure = butler.get('postISRCCD', dataId=dataId,
              collections=collections)
    detector  = exposure.getDetector()
    data = exposure.image.array

    vmin, vmax = zscale.get_limits(data)

    fig = plt.figure(figsize=(xsize, ysize))

    # plot in DVCS 
    if coords == 'CCS':
        data = data.T
    # otherwise,  for coords = 'DVCS' nothing needs be done 
    
    plt.imshow(data, origin='lower', vmin=vmin,vmax=vmax,cmap='gray')
    plt.colorbar()
    plt.xlabel('x [px]')
    plt.ylabel('y [px]')

    plt.title(f'{instrument}, {detector.getName()} ({detector.getSerial()[:3]}), {coords}')


    # read in the input instance catalog 
    # that is the reference catalog 
    #cat = np.genfromtxt(inst_file,
    #                        skip_header=16)

    #cat_df = pd.DataFrame(cat[:, 1:5], columns=['id', 'ra', 'dec', 'g'])
    cat = Table.read(inst_file, format='ascii')
    # get the wcs 
    wcs = exposure.getWcs()

    # calculate the position given the x,y 
    x,y= wcs.skyToPixelArray(cat['ra'], cat['dec'],degrees=True)
    
    if coords == 'CCS':
        x,y=y,x
        
    # plot the WCS
    plt.scatter(x,y,  facecolors='none', edgecolors='r', marker='o', s=50, label='phoSim WCS')

    plt.xlim(0,ymax)
    plt.ylim(0,xmax)
    plt.legend(fontsize=18)
    detectorName, detectorType = detector.getName(), detector.getSerial()[:3] 
    #plt.savefig(f'letter_lsstCam_entire_tiled_LSST_all_rot{detectorName}_{detectorType}.jpeg',
    #           bbox_inches='tight', dpi=150)
    
    

def ccd_xy_to_radec(x_px=[0], y_px=[0], boresight_ra=0, boresight_dec=0, rotskypos=0,
                     sensorNameList = ['R22_S11']):
    # Use the actual elements of SkySim and WcsSol , so 
    # it's more transparent  how CCD coordinates get translated into Ra, Deg catalog 

    rotSkyPos=rotskypos
    # get all sensors from the LsstCam : 
    #camera = obs_lsst.lsstCamMapper.LsstCamMapper().camera
    camera = LsstCam().getCamera()
    #setObsMetaData in bsc/WcsSol.py 
    boresightPointing = lsst.geom.SpherePoint(boresight_ra, 
                                              boresight_dec, 
                                              lsst.geom.degrees)
    centerCcd= "R22_S11"
    skyWcs = createInitialSkyWcsFromBoresight(
                boresightPointing,
                (90-rotSkyPos) * lsst.geom.degrees,
                camera[centerCcd],
                flipX=False,
            )

    # Get the pixel positions in DM team
    # pixelDmX, pixelDmY = self._sourProc.camXY2DmXY(xInpixelInCam, yInPixelInCam)
    # transpose - from DVCS to CCS 
    xInPixelInCam = x_px
    yInPixelInCam = y_px
    pixelDmX, pixelDmY = yInPixelInCam,  xInPixelInCam


    raList = []
    decList = []
    xPxList = []
    yPxList = []
    centerChip = camera[centerCcd]
    for sensorName in  sensorNameList:
        # same sensorName for each pixel in the list 
        chipNames =[sensorName for x in range(len(x_px))]
        for chipX, chipY, ccdX, ccdY, chipName in zip(pixelDmX, 
                                                      pixelDmY, 
                                                      xInPixelInCam,
                                                      yInPixelInCam,
                                                      chipNames
                                                     ):
            #print(chipX,chipY,chipName, raPt,decPt)
            cameraChip = camera[chipName]
            # Get x,y on specified detector in terms of mm from center of cam
            camXyMm = cameraChip.transform(
                lsst.geom.Point2D(chipX, chipY), PIXELS, FOCAL_PLANE
            )
            # Convert mm to pixels
            camPoint = centerChip.transform(camXyMm, FOCAL_PLANE, PIXELS)

            # Calculate correct ra, dec
            raPt, decPt = skyWcs.pixelToSky(camPoint)

            raList.append(raPt.asDegrees())
            decList.append(decPt.asDegrees())
            xPxList.append(ccdX)
            yPxList.append(ccdY)

    raInDeg, declInDeg =  np.array([raList, decList])
    
    return raInDeg, declInDeg , xPxList, yPxList


def pixel_L(xmin=500,ymin=200,width=800,height=1200,xspacing=220,yspacing=200):
    xmax = xmin+width
    ymax = ymin+height
    
    # initialize array storing coords for all pixels 
    xPx = np.zeros(0)
    yPx = np.zeros(0)

    # bott horizontal line
    x_bot = np.arange(xmin,xmax,xspacing)
    y_bot = np.ones_like(x_bot)*ymin
    xPx = np.append(xPx, x_bot)
    yPx = np.append(yPx, y_bot)

    # vertical line
    y_vert = np.arange(ymin+200,ymax-100,yspacing)
    x_vert = np.ones_like(y_vert)*xmin
    xPx = np.append(xPx, x_vert)
    yPx = np.append(yPx, y_vert)

    return xPx, yPx

def pixel_S(x0=0,y0=0):
    xPx = np.array([1000,
                    900,
                    750,
                    550,
                    350,
                    250,
                    200,
                    300,
                    500,
                    750,
                    900,
                    1000,
                    900,
                   700,
                   500,
                   340,
                  205,
                   ]
                  )
    yPx = np.array([1750,
                   1900,
                   2000,
                   2000,
                   1900,
                   1750,
                   1500,
                   1250,
                   1100,
                   1000,
                   850,
                   600,
                   350,
                   215,
                   125,
                   185,
                   305,
                   ])
    xPx += x0
    return xPx, yPx



def pixel_T(xmin=500,ymin=200,width=800,height=1200,xspacing=220,yspacing=200):
    xmax = xmin+width
    ymax = ymin+height-200
    
    # initialize array storing coords for all pixels 
    xPx = np.zeros(0)
    yPx = np.zeros(0)

    # top horizontal line
    x_top = np.arange(xmin,xmax,xspacing)
    y_top = np.ones_like(x_top)*ymax
    xPx = np.append(xPx, x_top)
    yPx = np.append(yPx, y_top)

    # vertical line
    y_vert = np.arange(ymin,ymax,yspacing)
    x_vert = np.ones_like(y_vert)*((xmin+xmax)/2.)
    xPx = np.append(xPx, x_vert)
    yPx = np.append(yPx, y_vert)

    return xPx, yPx

def pixel_LSST(x0=0,y0=0,xf=1.0,yf=1.0):
    
    xPx = np.zeros(0)
    yPx = np.zeros(0)
    
    x,y = pixel_L(xmin=200,ymin=200,xspacing=100, yspacing=200, width=800, height=2000 )
    xPx = np.append(xPx, x)
    yPx = np.append(yPx, y)
    
    x,y = pixel_S(x0=900)
    xPx = np.append(xPx, x)
    yPx = np.append(yPx, y)
    
    x,y = pixel_S(x0=1900)
    xPx = np.append(xPx, x)
    yPx = np.append(yPx, y)
    
    x,y = pixel_T(xmin=3000,ymin=200,xspacing=100, yspacing=200, width=950, height=2000)
    xPx = np.append(xPx, x)
    yPx = np.append(yPx, y)
    
    
    xPx *= xf
    yPx *= yf 
    
    xPx += x0
    yPx += y0 
    
    
    
    return xPx, yPx 
    

def get_camera_sensors(instrument):
    instruments = ['lsstCam', 'comCam']
    if instrument in instruments:
        if instrument == 'lsstCam':
            camera = LsstCam().getCamera()
        if instrument == 'comCam':
            camera = LsstComCam().getCamera()
        sensorsAll = np.sort(list(camera.getNameMap().keys()))
        return sensorsAll
    else:
        raise ValueError(f'{instrument} is not a recognized instrument name; use one of ', instruments)
        
def write_to_file(out_file, content):
    with open(out_file, "w") as output:
        for line in content:
            output.write(line)
            

def readFile(pathToFile, headerString = '#'):
    ''' Read file line-by-line  as two lists,
    one containing the header lines, 
    the other the content lines 
    
    Parameters:
    -----------
    pathToFile: `str`
        absolute path to file 
        
    Returns:
    ----------
    `list`, `list` 
        lists containing the header lines 
        and the content lines 
    '''
    with open(pathToFile) as f:
        allLines = f.readlines()

    headerLines=[]
    contentLines=[]

    # store the header part and content separately 
    for line in allLines:
        if line.startswith('#'):
            headerLines.append(line)
        else:
            contentLines.append(line)

    return headerLines, contentLines 


def update_cmd_mirror_paths(bkgnd, pert, cmd_dir, mirror_dir):
      
    cmd_file = f'{bkgnd}BkgndPert{pert}.cmd'

    path_to_cmd = os.path.join(cmd_dir, cmd_file)
    surfacemap_dir = os.path.join(f'imgCloseLoop_0-{pert}','iter0','pert')

    header, content = readFile(path_to_cmd)  
    #print(content)

    new_content = content.copy()
    for i in range(len(content)):
        line = content[i]
        newline = line
        if line.startswith('surfacemap'):
            print(line) 
            # split the original line 
            splitline = line.split()
            mirror_file = splitline[2].split('/')[-1]
            path_to_mirror = os.path.join(mirror_dir, surfacemap_dir, mirror_file)
            #print(path_to_mirror)
            # replace that with a new line with an  updated path 
            newSplitLine = splitline[:2] # "eg. surfacemap 0"
            # eg. /project/scichris/aos/AOS/DM-28360/imgCloseLoop_0-00/iter0/pert/M1res.txt
            #print(newSplitLine)
            newSplitLine.append(path_to_mirror)
            newSplitLine.append(splitline[-1]) # eg. 1
            newline = ' '.join(newSplitLine)
            newline += ' \n'
            print('-->', newline)
        new_content[i] = newline
    
    return new_content
        
        
def update_inst_file_obshistid(obshistid, inst_file,  root_dir, seeing=0.69):
    
    path_to_inst_file = os.path.join(root_dir, inst_file)

    header, content = readFile(path_to_inst_file)
    new_content = content.copy()
    for i in range(len(content)):
        # first update obshistid
        if content[i].find('obshistid')>0:
            split_line = content[i].split(' ')
            split_line[1] = f'{obshistid}\n'
            new_content[i] = ' '.join(split_line)
            
        # second update the seeing 
        if content[i].find('rawseeing')>0:
            split_line = content[i].split(' ')
            split_line[1] = f'{seeing}\n'
            new_content[i] = ' '.join(split_line)
            
            continue
            
    new_inst_file = inst_file[:-(len('.inst'))]+f'_{obshistid}.inst'
    out_file = os.path.join(root_dir, new_inst_file)
    write_to_file(out_file, new_content)
    print(f'Updated {inst_file} to {new_inst_file} in {root_dir}')

    
            
def invert_dict(dic):
    invDic = {}
    for key, value in zip(dic.keys(), dic.values()):
        invDic[value] = key
    return invDic



def get_inst_dict():
    return {"comCam": 0, "lsstCam": 1, "wfs": 2}


def get_field_dict():
    return {"high": 0, "med": 1, "low": 2, "Baade": 3}


def get_position_dict():
    return {"focal": 0, "extra": 1, "intra": 2}


def get_cmd_dict():
    return {
        "noBkgndPert00": 0,
        "noBkgndPert05": 1,
        "qckBkgndPert00": 2,
        "qckBkgndPert05": 3,
        "noBkgnd": 4,
        "qckBkgnd": 5,
    }


def calculate_obshistid(instrument, field, position, cmd_file, run):

    instDict = get_inst_dict()
    fieldDict = get_field_dict()
    positionDict = get_position_dict()

    if cmd_file.find("_") > 0:
        # eg 'noBkgndPert00_NCSA.cmd', 'noBkgndPert00_hyak.cmd'
        # i.e. with corrected surfacemap paths
        cmd = cmd_file.split("_")[0]
    else:  # eg. 'noBkgndPert00.cmd'
        # i.e. with original paths
        cmd = cmd_file.split(".")[0]
    cmdDict = get_cmd_dict()

    first = instDict[instrument]
    second = fieldDict[field]
    third = positionDict[position]
    fourth = cmdDict[cmd]
    obshistid = f"90{first}{second}{third}{fourth}{run}"
    return obshistid


def invert_obshistid(obshistid):
    ''' Given obshistid, invert the logic and find the 
    instrument, field, position, cmd_file, run
    
    Parameters:
    -----------
    obshistid: str
         The obs hist id, eg. "9020031"
        
    Returns:
    --------
    inverted_dict:  dic
         A dictionary containing keys that resolve the obshistid:
         instrument, field, position, cmd, run
    '''
    instDictInv = invert_dict(get_inst_dict())
    fieldDictInv = invert_dict(get_field_dict())
    positionDictInv = invert_dict(get_position_dict())
    cmdDictInv = invert_dict(get_cmd_dict())
    
   # take last five digits
    digits = obshistid[-5:]

    first=int(digits[0])
    second=int(digits[1])
    third=int(digits[2])
    fourth=int(digits[3])
    run=int(digits[4])


    instrument = instDictInv[first]
    field = fieldDictInv[second]
    position= positionDictInv[third]
    cmd = cmdDictInv[fourth]
    
    return {'instrument':instrument, 'field':field, 
            'position':position, 'cmd':cmd, 
            'run':run
           }

def sensor_list_to_string(sensorNameList):
    sensors = ""
    for sensor in sensorNameList:
        sensors += "%s|" % sensor
    return sensors


def get_field_ra_dec():
    """Read the ra,dec coordinates of simulated fields
    for creation of phosim files. Restructure as
    dictionary for easier access of each value.

    Returns:
    -------
    gt_dict: a dictionary where keys are field names, eg.
            high, med, low, Baade, so that ra,dec can be
            accessed via eg. gt_dict['high']['ra']
    """
    filename = "/project/scichris/aos/ps1_query_coordinates.txt"
    gt = Table.read(filename, format="ascii")

    # rewrite as dict for easier access of coordinates
    # of a specific field by name
    gt_dict = {}
    for i in range(len(gt)):
        gt_dict[gt["name"][i]] = {"ra": gt["ra_deg"][i],
                                  "dec": gt["dec_deg"][i]
                                  }
    return gt_dict

def pixel_letter_F(xmin=500,ymin=200,width=800,height=1200,xspacing=220,yspacing=200):
    xmax = xmin+width
    ymax = ymin+height
    # initialize array storing coords for all pixels 
    xPx = np.zeros(0)
    yPx = np.zeros(0)

    # middle horizontal line
    x_mid = np.arange(xmin,xmax,xspacing)
    y_mid = np.ones_like(x_mid) * ((ymin+ymax)/2.)
    xPx = np.append(xPx, x_mid)
    yPx = np.append(yPx, y_mid)
    
    # top horizontal line
    x_top = x_mid.copy()
    y_top = np.ones_like(x_top)*ymax
    xPx = np.append(xPx, x_top)
    yPx = np.append(yPx, y_top)

    # vertical line
    y_vert = np.arange(ymin+200,ymax-100,yspacing)
    x_vert = np.ones_like(y_vert)*xmin
    xPx = np.append(xPx, x_vert)
    yPx = np.append(yPx, y_vert)

    return xPx, yPx


def write_phosim_header(
    output,
    ra,
    dec,
    camconfig=3,
    opsim_filter=3,
    mjd=59580.0,
    exposure=0.25,
    obsid=9006002,
    position='focal',
    seeing=0.69
):
    """Write to file handle the phosim header"""
    output.write("Opsim_obshistid {}\n".format(obsid))
    output.write("Opsim_filter {}\n".format(opsim_filter))
    output.write("mjd {}\n".format(mjd))
    output.write("SIM_SEED {}\n".format(1000))
    output.write("rightascension {}\n".format(ra))
    output.write("declination {}\n".format(dec))
    output.write("rotskypos {}\n".format(0.000000))
    output.write("rottelpos {}\n".format(0))
    output.write("SIM_VISTIME {}\n".format(exposure))
    output.write("SIM_NSNAP {}\n".format(1))
    output.write("moonphase {}\n".format(0.0))
    output.write("moonalt {}\n".format(-90))
    output.write("sunalt {}\n".format(-90))
    output.write("Opsim_rawseeing {}\n".format(seeing))
    if position == 'extra':
        output.write("move 10 -1500.0000\n")  # write the defocal movement
    elif position == 'intra':
        output.write("move 10  1500.0000\n")  
    output.write("camconfig {}\n".format(camconfig))
    


# NB:  here we only do 0.25 sec exposure to quickly test if the image is properly made 
# in generating large catalogs, change to 15 sec. 
def writePhosimHeader(output, ra, dec, rotskypos=0, rottelpos=0, 
                      camconfig=3, Opsim_filter=3, mjd=59580., exposure=0.25,
                     obsid=9006002, position='focal',
                     opsim_rawseeing=0.69):
    '''Write to file handle the phosim header'''
    output.write("Opsim_obshistid {}\n".format(obsid))
    output.write("Opsim_filter {}\n".format(Opsim_filter))
    output.write("mjd {}\n".format(mjd))
    output.write("SIM_SEED {}\n".format(1000))
    output.write("rightascension {}\n".format(ra))
    output.write("declination {}\n".format(dec))
    output.write("rotskypos {}\n".format(rotskypos))
    output.write("rottelpos {}\n".format(rottelpos))
    output.write("SIM_VISTIME {}\n".format(exposure))
    output.write("SIM_NSNAP {}\n".format(1))
    output.write("moonphase {}\n".format(0.0))
    output.write("moonalt {}\n".format(-90))
    output.write("sunalt {}\n".format(-90))
    output.write("Opsim_rawseeing {}\n".format(opsim_rawseeing))
    if position == 'extra':
        output.write("move 10 -1500.0000\n")  # write the defocal movement
    elif position == 'intra':
        output.write("move 10  1500.0000\n")  
    output.write("camconfig {}\n".format(camconfig))

def writePhosimInstFile(catalog, ra, dec, rotskypos=0,  phosimFile="stars.inst", 
                        passband='r', outDir="./", camconfig=3, opd=False,
                        mjd=59580,exposure=.25, obsid=9006002, position='focal',
                        constMag=15. ):
    '''Generate a phosim instance catalog from a Pandas dataframe or Astropy Table '''
    
    passbands = {"u":"umeanpsfmag", 
                 "g":"gmeanpsfmag", 
                 "r":"rmeanpsfmag", 
                 "i":"imeanpsfmag", 
                 "z":"zmeanpsfmag", 
                 "y":"ymeanpsfmag", 
                }
    
    outFile = os.path.join(outDir, phosimFile)
    
    filterID = list(passbands.keys()).index(passband)
    with open(outFile,'w') as output:
        writePhosimHeader(output, ra, dec, rotskypos=rotskypos, camconfig=camconfig, 
                          Opsim_filter=filterID, mjd=mjd,
                         exposure=exposure, obsid=obsid, position=position)
        if (opd==True):
            output.write("opd 0 {} {} 500".format(ra,dec))
        else:
            for idx in range(len(catalog)):
                row = catalog[idx]
                output.write("object {} {} {} {} ../sky/sed_flat.txt 0.0 0.0 0.0 0.0 0.0 0.0 star 12.0 none none\n".format(
                    int(row["objid"]),row["ra"],row["dec"], constMag))
    print('Saved as ', outFile)
    return outFile
 
def plot_cameraGeom(repo_dir = os.path.join('/project/scichris/aos/images/lsstCam/letter_lsstCam_entire_tiledF/DATA/'),
                   detectorNameList = [ "R04_SW0", "R04_SW1",
                         "R44_SW0", "R44_SW1",
                         "R00_SW0", "R00_SW1",
                         "R40_SW0", "R40_SW1",
                      
                       ] ,
                    instrument = 'LSSTCam',
                    binSize=16,
                    collections = ['run1']
                   ):
    afwDisplay.setDefaultBackend("matplotlib")

    #repo_dir = os.path.join('/project/scichris/aos/images/lsstCam/letter_lsstCam_entire_tiledF/DATA/')

    # need to specify the calib collections to be able to access the camera 
    
    butler = dafButler.Butler(repo_dir)# collections=[f'{instrument}/raw/all',
                                        ##             f'{instrument}/calib','run1',
                                         #            f'{instrument}/calib/unbounded']
                             #)

    dataId0 = dict(instrument=instrument)
    dataset = next(iter(
                        butler.registry.queryDatasets(
                            datasetType='postISRCCD', 
                            collections=collections,
                            dataId=dataId0  )
                        )
                   )

    exposure = dataset.dataId["exposure"]

    camera  = butler.get("camera", instrument=instrument, exposure=exposure,
                        collections = f'{instrument}/calib/unbounded')

    fig = plt.figure(figsize=(15,15))
    fig.subplots_adjust(hspace=0., wspace=0.0, )

    ax = fig.add_subplot(1, 1, 1)

    disp = afwDisplay.Display(fig)
    #disp.scale('asinh', 'zscale', Q=2)
    disp.scale('linear', 'minmax')
    #disp.scale('asinh', 5, 7000, Q=2)

    disp.setImageColormap('viridis' if True else 'gray')
    dataType = "postISRCCD"


    # use that  to limit what's shown
    #detectorNameList = ['R22_S11', 'R22_S20', 'R22_S10']# , 'R34_S20'] 

    # set to None to plot everything
    
    mos = cameraGeomUtils.showCamera(camera,
                                 cameraGeomUtils.ButlerImage(butler, dataType, 
                                                             instrument=instrument, 
                                                             exposure=exposure,
                                                             verbose=True),
                                 binSize=binSize, 
                                 detectorNameList=detectorNameList, 
                                 display=disp, overlay=True, title=f'{instrument}' )

    disp.show_colorbar(False)
    ax.axis("off")

    #plt.savefig('LsstCam_letter_entire_tiledF_wfs.png', bbox_inches='tight')
    return mos   
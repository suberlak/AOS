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
from lsst.ts.wep.ParamReader import ParamReader

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
                         plot_raw_zk = False, aos_fwhm_arcsec_level = 0.089
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
        
    Returns:
    --------
    None
     
    '''
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
        #     for i in range(np.shape(opdData)[0]):
        #         ax.plot(opdData[i,:], lw=1,ls='--')
            # plot the average of these ... 
            ax[k].plot(1000*np.mean(opdData, axis=0), lw=3,ls='-' ,)

        ax[k].set_xlabel('Zernike #')
        ax[k].set_ylabel('wavefront error (OPD) '+r'$[nm]$')
        k += 1 
                       
    if plot_rms_opd_zk:
        # 1: plot Zernike wavefront errors  vs OPD 
        for iterCount in range(iterNum):
            wfsData = wfsDataDic[iterCount]
            opdData = opdDataDic[iterCount]
            # do the difference with mean OPD ... 
            meanOpdData = np.mean(opdData, axis=0)
            zernikeErrorsDiff = np.sqrt((wfsData - meanOpdData)**2.)

            zernikeErrors = np.transpose(zernikeErrorsDiff, axes=(1,0))

            zernikeRms = np.sqrt(np.mean(np.square(zernikeErrors), axis=1))

            ax[k].plot(np.arange(19)+4, 1000*zernikeRms, 
                          '-o', lw=3, label='iter%d'%iterCount)

        ax[k].set_xlabel('Zernike Number', size=18)
        ax[k].set_ylabel('RMS(WFS-OPD) [nm]', size=18)

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
        ax[k].axhline(aos_fwhm_arcsec_level, c='orange', ls='--',lw=3, label=f'{aos_fwhm_arcsec_level} line')
        ax[k].legend()
        
        ax[k].set_xlabel('Iteration')
        ax[k].set_ylabel('FWHM_eff (arcseconds)')
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
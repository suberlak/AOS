from astropy.table import Table
from lsst.ts.phosim.SkySim import SkySim
import os
import pandas as pd
import numpy as np 
import argparse
import run_ps1_functions as func
from lsst.obs.lsst import LsstCam, LsstComCam

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
    # NB: no defocal movement for OPD! 
#     if position == 'extra':
#         output.write("move 10 -1500.0000\n")  # write the defocal movement
#     elif position == 'intra':
#         output.write("move 10  1500.0000\n")  
    output.write("camconfig {}\n".format(camconfig))


def write_opd_inst_file(
    panda_cat,
    ra,
    dec,
    inst_path=os.path.join("./","opd.inst"),
    passband="r",
    camconfig=3,
    exposure=15,
    obsid=9006002,
    position="focal",
    mjd=59580,
    magcol=None
    
):
    """Generate a phosim OPD instance catalog"""

    passbands = {
        "u": "umeanpsfmag",
        "g": "gmeanpsfmag",
        "r": "rmeanpsfmag",
        "i": "imeanpsfmag",
        "z": "zmeanpsfmag",
        "y": "ymeanpsfmag",
    }

    out_file = inst_path

    filterid = list(passbands.keys()).index(passband)
    if magcol is None:
        magcol = passbands[passband]

    with open(out_file, "w") as output:
        write_phosim_header(
            output,
            ra,
            dec,
            camconfig=camconfig,
            opsim_filter=filterid,
            mjd=mjd,
            exposure=exposure,
            obsid=obsid,
            position=position )
        
        for index, row in panda_cat.iterrows():
            output.write("opd {} {} {} 500 \n".format(
                            int(row["objid"]),
                                row["ra"], row["dec"])
                        )
            

    print("Saved as ", out_file)

def get_all_sensor_names_xy(camera):
    
    sensorNames = []
    xCenter, yCenter = [], []
    for detector in camera:
        
        bbox = detector.getBBox()
        xCen, yCen = bbox.centerX, bbox.centerY
        xCenter.append(xCen)
        yCenter.append(yCen)
        sensorNames.append(detector.getName())
        
    return sensorNames, xCenter, yCenter


def get_wfs_sensor_names_xy():
    sensorNames = [  "R04_SW0", "R04_SW1",
                     "R44_SW0", "R44_SW1",
                     "R00_SW0", "R00_SW1",
                     "R40_SW0", "R40_SW1"
                   ]
    xCenter = []
    yCenter = []
    camera = LsstCam().getCamera()
    for sensor in sensorNames:
        detector = camera.get(sensor)
        bbox = detector.getBBox()
        xCen, yCen = bbox.centerX, bbox.centerY
        xCenter.append(xCen)
        yCenter.append(yCen)
        
    return sensorNames, xCenter, yCenter


def get_opd_ra_dec(instrument, raField, decField, rotskypos):
    ''' Provide phoSim with ra,dec locations to evaluate OPD.
    The OPD expects field location, not sky location.
    '''
    if instrument == 'comCam':
        sensorNames, xCenter, yCenter = get_all_sensor_names_xy(LsstComCam().getCamera())

    elif instrument == 'lsstCam':
        sensorNames, xCenter, yCenter = get_all_sensor_names_xy(LsstCam().getCamera())

    elif instrument == 'wfs':
        sensorNames, xCenter, yCenter = get_wfs_sensor_names_xy()

    # Declare the SkySim()
    skySim = SkySim()
    
    # Set the observation information
    skySim.setObservationMetaData(raField, decField, rotskypos)
        
    xPx = xCenter
    yPx = yCenter
    starMag = 15


    starId = 0 
    # we have only one x,y point per sensor 
    for i in range(len(sensorNames)):
        skySim.addStarByChipPos(sensorNames[i], starId, xCenter[i],
                                yCenter[i], starMag)
        starId += 1 

    raList , decList = skySim.getRaDecInDeg()

    # Note:  ra for phosim is (0 <= ra <= 360). But the 
    # field position might be < 0 (-180 <= field_ra <= 180).
    # So here I change 
    # the (0,360) range to (-180,180) range:
    m =raList>180
    raList[m] = raList[m] - 360
    
    mags = skySim.getStarMag()
    coords = Table(data=[raList, decList, xCenter, yCenter, mags], 
                 names=['ra','dec', 'xPx', 'yPx','r' ]  )
        
 
    
    # add a column with object id 
    coords['objid'] = np.arange(len(coords))

    panda_cat = coords.to_pandas()
    
    return panda_cat



def main(
    instruments,
    fields,
    positions,
    perts,
    bkgnds,
    root_dir,
    exposure,
    dry_run,
    run=1
):
    

    # get ra,dec coordinates of fields 
    gt_dict = func.get_field_ra_dec()
    
    for instrument in instruments:
        for field in fields:
            for position in positions:
                for pert in ['00','05']:
                    for bkgnd in ['qck','no']:
                        cmd_file = f'{bkgnd}BkgndPert{pert}_NCSA.cmd'
                        print(instrument, field, position, pert, bkgnd)
                        
                        # obtain ra,dec coords of the OPD
                        # given the instrument and pointing
                        # assuming we take the center of each CCD 
                        
                        # NOTE: this is ONLY used for the header of hte inst file
                        # the actual OPD field positions are irrespective of the 
                        # boresight
                        raField = gt_dict[field]["ra"] 
                        decField = gt_dict[field]["dec"]

                        obshistid = func.calculate_obshistid(instrument, field, position, cmd_file, run)
                        opd_fname = f'opd_{instrument}_{field}_{position}_{obshistid}.inst'
                        print(opd_fname)
                        inst_path = os.path.join(root_dir, opd_fname)
                        
                        if not dry_run:
                            panda_cat = get_opd_ra_dec(instrument, raField=0, decField=0,rotskypos=0)
                            write_opd_inst_file(panda_cat, raField, decField,
                                inst_path, passband="r", camconfig=3,
                                exposure=exposure, obsid=obshistid, position=position, mjd=59580,
                                magcol=None)



            

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Create phosim instance catalogs for OPD generation."
    )
    parser.add_argument(
        "--instruments",
        nargs="+",
        default=["comCam"],
        help='A list of instruments based on which to generate OPD. \
        We can use "lsstCam", "comCam" or "wfs" (default:comCam)',
    )
    parser.add_argument(
        "--fields",
        nargs="+",
        default=["high"],
        help='A list of field names to generate, \
eg. "high", "med", "low", "Baade" (default: high)',
    )
    parser.add_argument(
        "--positions",
        nargs="+",
        default=["focal"],
        help='A list of positions to simulate, eg. "focal", "intra", "extra". (default: focal)',
    )
    parser.add_argument(
        "--bkgnds",
        nargs="+",
        default=["no", "qck"],
        help='List of backgrounds to use for cmd file, eg. "no", "qck". (default: all )',
    )
    parser.add_argument(
        "--perts",
        nargs="+",
        default=["00","05"],
        help='A list of perturbations to simulate, eg. "00", "05". (default: all) ',
    )
    parser.add_argument(
        "--root_dir",
        type=str,
        default="/project/scichris/aos/AOS/DM-28360/",
        help="Absolute path to the work directory where .inst  \
files should be stored.(default: /project/scichris/aos/AOS/DM-28360/) ",
    )
    
    parser.add_argument(
        "--exposure",
        nargs=1,
        type=float,
        default=15,
        help="Exposure time for the instance catalogs, stored as SIM_VISTIME \
in the .inst file (default: 15)",
    )
    parser.add_argument(
        "--dry_run",
        default=False,
        action="store_true",
        help="Do not run any calculations, just print commands used.",
    )
        

    args = parser.parse_args()
    main(
        instruments=args.instruments,
        fields=args.fields,
        positions=args.positions,
        perts=args.perts,
        bkgnds=args.bkgnds,
        root_dir=args.root_dir,
        exposure=args.exposure,
        dry_run=args.dry_run
    )

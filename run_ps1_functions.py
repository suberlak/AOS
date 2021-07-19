import subprocess
import os
import argparse
import time
from astropy.table import Table

# this is a place to put all functions 
# used by at least two of the 
# ps1-simulation suite,
# including 
# run_ps1_1_catalogs
# run_ps1_phosim
# run_ps1_3_butler
# run_ps1_slurm


def write_to_file(out_file, content):
    with open(out_file, "w") as output:
        for line in content:
            output.write(line)
            
            
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
    
    

    
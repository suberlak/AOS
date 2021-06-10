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

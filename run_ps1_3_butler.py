import yaml
import os 
import subprocess 
import argparse

# write the pipeline yaml file
def write_pipeline_yaml(out_dir, instrument='LsstComCam', file_name='testPipeline.yaml'):
    '''
    Write a YAML file that describes the pipeline to be 
    executed on raws ingested to a gen3 repo. 
    '''
    dict_file = {'description':'ISR basic processing pipeline',
                 'instrument':f'lsst.obs.lsst.{instrument}',
                 'tasks':
                     {'isr': 
                          {'class':'lsst.ip.isr.isrTask.IsrTask',
                           'config':
                             {
                              'connections.outputExposure': 'postISRCCD',
                              'doBias':False,
                              'doVariance': False,
                              'doLinearize': False,
                              'doCrosstalk': False,
                              'doDefect': False,
                              'doNanMasking': False,
                              'doInterpolate': False,
                              'doBrighterFatter': False,
                              'doDark': False,
                              'doFlat': False,
                              'doApplyGains': True,
                              'doFringe': False,
                              'doOverscan': True
                              }

                           }
                     }
                 }


    out_file = os.path.join(out_dir, file_name)
    print('Saved as ', out_file)
    with open(out_file, 'w') as file:
        documents = yaml.dump(dict_file, file)
        
        
        
def write_isr_script(out_dir, instrument="LsstComCam"):
    '''
    Write to file a script to create butler repo
    and do ISR.
    '''
    outFile = os.path.join(out_dir, 'runIsr.sh')
    
    with open(outFile,'w') as output:
        # saves a file with specific permissions
        # this one needs to be executable
        # at least by the user and group
        os.chmod(outFile, 0o776) 
        
        output.write("#!/bin/bash\n")
        output.write("butler create DATA\n")
        output.write(f"butler register-instrument DATA/ lsst.obs.lsst.{instrument}\n")
        output.write("butler ingest-raws DATA repackaged/\n")
        output.write(f"butler define-visits DATA/ lsst.obs.lsst.{instrument}\n")
        output.write(f"butler write-curated-calibrations DATA/ lsst.obs.lsst.{instrument}\n")
        
        if instrument == "LsstComCam":
            instRepo = "LSSTComCam"
        elif instrument == "LsstCam":
            instRepo = "LSSTCam"
            
        output.write(f"pipetask run -j 9 -b DATA/ -i {instRepo}/raw/all,{instRepo}/calib \
        -p testPipeline.yaml --register-dataset-types --output-run run1")
        
    print(f'Saved as {outFile}')

def get_butler_instrument(instrument):
    if instrument == 'lsstCam':
        butler_instrument = 'LsstCam'
    elif instrument == 'comCam':
        butler_instrument = 'LsstComCam'
    return butler_instrument
    
def main(
    instruments=["comCam"],
    fields=["high"],
    positions=["focal"],
):
    
    root_dir = "/project/scichris/aos/ps1_phosim/"

    default_dir = os.getcwd()
    for instrument in instruments:
        for field in fields:
            for position in positions: 
                print(f"\n Running for {instrument} {field} {position}")
                # build the name of input directory with 
                # repackaged simulated files
                work_dir = os.path.join(root_dir, instrument, field, position)
                
                # check that there are any files in /repackaged/ dir 
                repackaged_dir = os.path.join(work_dir,'repackaged')
                if not os.path.exists(repackaged_dir):
                    raise RuntimeError("There is no /repackaged/ directory.")
                elif len(os.listdir(repackaged_dir)) < 1 : 
                    raise RuntimeError("There is a /repackaged/ dir \
but there are no files in it.")

                # if the two conditions are met we can 
                # proceed to butler ingestion of raw files 
                butler_instrument = get_butler_instrument(instrument)
                write_pipeline_yaml(work_dir, butler_instrument)
                write_isr_script(work_dir, butler_instrument)

                os.chdir(work_dir)
                # run the script to create butler and ingest 
                print('Running sh  ./runIsr.sh')
                subprocess.call(['sh', './runIsr.sh']) 
                os.chdir(default_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Butler ingestion of phosim simulated data using PS1 catalogs.\
This program writes and executes butler scripts that create a gen3 repo,\
register appropriate instrument, ingest raws, define visits, register calibs,\
and do the instrument signature removal.")

    parser.add_argument(
        "--instruments",
        nargs="+",
        default=["comCam"],
        help='A list of instruments, eg. "lsstCam", "comCam" ',
    )
    parser.add_argument(
        "--fields",
        nargs="+",
        default=["high"],
        help='A list of field names to generate, \
eg. "high", "med", "low", "Baade"',
    )
    parser.add_argument(
        "--positions",
        nargs="+",
        default=["defocal"],
        help='A list of positions to simulate, eg. "focal", "defocal". ',
    )
    args = parser.parse_args()
    main(
    instruments=args.instruments,
    fields=args.fields,
    positions=args.positions,
    )
import yaml
import os
import subprocess
import argparse

# write the pipeline yaml file
def write_pipeline_yaml(
    out_dir, instrument="LsstComCam", file_name="testPipeline.yaml"
):
    """
    Write a YAML file that describes the pipeline to be
    executed on raws ingested to a gen3 repo.
    """
    dict_file = {
        "description": "ISR basic processing pipeline",
        "instrument": f"lsst.obs.lsst.{instrument}",
        "tasks": {
            "isr": {
                "class": "lsst.ip.isr.isrTask.IsrTask",
                "config": {
                    "connections.outputExposure": "postISRCCD",
                    "doBias": False,
                    "doVariance": False,
                    "doLinearize": False,
                    "doCrosstalk": False,
                    "doDefect": False,
                    "doNanMasking": False,
                    "doInterpolate": False,
                    "doBrighterFatter": False,
                    "doDark": False,
                    "doFlat": False,
                    "doApplyGains": True,
                    "doFringe": False,
                    "doOverscan": True,
                },
            }
        },
    }

    out_file = os.path.join(out_dir, file_name)
    print("Saved as ", out_file)
    with open(out_file, "w") as file:
        documents = yaml.dump(dict_file, file)


def write_isr_script(out_dir, instrument="LsstComCam"):
    """
    Write to file a script to create butler repo
    and do ISR.
    """
    outFile = os.path.join(out_dir, "runIsr.sh")

    with open(outFile, "w") as output:
        # saves a file with specific permissions
        # this one needs to be executable
        # at least by the user and group
        os.chmod(outFile, 0o776)

        output.write("#!/bin/bash\n")
        output.write("butler create DATA\n")
        output.write(f"butler register-instrument DATA/ lsst.obs.lsst.{instrument}\n")
        output.write("butler ingest-raws DATA repackaged/\n")
        output.write(f"butler define-visits DATA/ lsst.obs.lsst.{instrument}\n")
        output.write(
            f"butler write-curated-calibrations DATA/ lsst.obs.lsst.{instrument}\n"
        )

        if instrument == "LsstComCam":
            instRepo = "LSSTComCam"
        elif instrument == "LsstCam":
            instRepo = "LSSTCam"

        output.write(
            f"pipetask run -j 9 -b DATA/ -i {instRepo}/raw/all,{instRepo}/calib \
        -p testPipeline.yaml --register-dataset-types --output-run run1"
        )

    print(f"Saved as {outFile}")


def get_butler_instrument(instrument):
    if instrument == "lsstCam":
        butler_instrument = "LsstCam"
    elif instrument == "comCam":
        butler_instrument = "LsstComCam"
    return butler_instrument


def calculate_obshistid(instrument, field, position, cmd_file, run):

    instDict = {"comCam": 0, "lsstCam": 1}
    fieldDict = {"high": 0, "med": 1, "low": 2, "Baade": 3}
    positionDict = {"focal": 0, "extra": 1, "intra": 2}

    if cmd_file.find("_") > 0:
        # eg 'noBkgndPert00_NCSA.cmd', 'noBkgndPert00_hyak.cmd'
        # i.e. with corrected surfacemap paths
        cmd = cmd_file.split("_")[0]
    else:  # eg. 'noBkgndPert00.cmd'
        # i.e. with original paths
        cmd = cmd_file.split(".")[0]
    cmdDict = {
        "noBkgndPert00": 0,
        "noBkgndPert05": 1,
        "qckBkgndPert00": 2,
        "qckBkgndPert05": 3,
        "noBkgnd": 4,
        "qckBkgnd": 5,
    }
    first = instDict[instrument]
    second = fieldDict[field]
    third = positionDict[position]
    fourth = cmdDict[cmd]
    obshistid = f"90{first}{second}{third}{fourth}{run}"
    return obshistid


def find_dirs(
    root_dir="/project/scichris/aos/AOS/DM-28360/",
    instruments=["comCam"],
    fields=["med"],
    positions=["focal", "extra", "intra"],
):

    all_dirs = []

    for instrument in instruments:
        for field in fields:
            for position in positions:
                work_dir = os.path.join(root_dir, instrument, field, position)
                print(work_dir)
                sub_dir_names = os.listdir(work_dir)
                for sub_dir in sub_dir_names:
                    all_dirs.append(os.path.join(work_dir, sub_dir))
                # add
                print(" ", sub_dirs)

    return all_dirs


def main(instruments, fields, positions, cmd_files, root_dir, run, dry_run):

    default_dir = os.getcwd() # get current working directory 
    
    
    
    
    
    for instrument in instruments:
        for field in fields:
            for position in positions:
                for cmd_file in cmd_files:
                    print(f"\n Running for {instrument} {field} {position}")
                    # build the name of input directory with
                    # repackaged simulated files
                    # work_dir = os.path.join(root_dir, instrument, field, position)

                    obshistid = calculate_obshistid(
                        instrument, field, position, cmd_file, run
                    )

                    work_dir = os.path.join(
                        root_dir, instrument, field, position, str(obshistid)
                    )

                    # check that there are any files in /repackaged/ dir
                    repackaged_dir = os.path.join(work_dir, "repackaged")
                    if not os.path.exists(repackaged_dir):
                        raise RuntimeError("There is no /repackaged/ directory.")
                    elif len(os.listdir(repackaged_dir)) < 1:
                        raise RuntimeError(
                            "There is a /repackaged/ dir but there are no files in it."
                        )

                    # if the two conditions are met we can
                    # proceed to butler ingestion of raw files
                    butler_instrument = get_butler_instrument(instrument)
                    write_pipeline_yaml(work_dir, butler_instrument)
                    write_isr_script(work_dir, butler_instrument)
                    
                    if not dry_run:
                        os.chdir(work_dir)
                        # run the script to create butler and ingest
                        print("Running sh  ./runIsr.sh")
                        subprocess.call(["sh", "./runIsr.sh"])
                        os.chdir(default_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Butler ingestion of phosim simulated data using PS1 catalogs.\
This program writes and executes butler scripts that create a gen3 repo,\
register appropriate instrument, ingest raws, define visits, register calibs,\
and do the instrument signature removal."
    )

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
        default=["focal"],
        help='A list of positions to simulate, eg. "focal", "extra", "intra". ',
    )
    parser.add_argument(
        "--cmd_files",
        type=str,
        default=["noBkgndPert00_NCSA.cmd",  "noBkgndPert05_NCSA.cmd",
                 "qckBkgndPert00_NCSA.cmd", "qckBkgndPert05_NCSA.cmd"
                ],
        help="Name of the physics command file(s) used by phosim",
    )
    parser.add_argument(
        "--root_dir",
        type=str,
        default="/project/scichris/aos/AOS/DM-28360/",
        help="Absolute path to the work directory where .cmd and .inst \
files can be found. That is also where the output \
under  {instrument}{field}{position}{obshistid}\
will be written",
    )
    parser.add_argument(
        "--run",
        "-r",
        nargs=1,
        type=int,
        default=1,
        help="Run number, assuming instrument, field, position, cmd_file\
are the same (default:1).",
    )
    parser.add_argument(
        "--dry_run",
        default=False,
        action="store_true",
        help="Do not run any simulation, just print commands used.",
    )

    args = parser.parse_args()
    main(
        instruments=args.instruments,
        fields=args.fields,
        positions=args.positions,
        cmd_files=args.cmd_files,
        root_dir=args.root_dir,
        run=args.run,
        dry_run=args.dry_run
    )

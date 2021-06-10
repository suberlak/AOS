import yaml
import os
import subprocess
import argparse
import run_ps1_functions as func

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



def invert_obshistid(obshistid):
    """Given obshistid, invert the logic and find the
    instrument, field, position, cmd_file, run
    """
    instDictInv = func.invert_dict(func.get_inst_dict())
    fieldDictInv = func.invert_dict(func.get_field_dict())
    positionDictInv = func.invert_dict(func.get_position_dict())
    cmdDictInv = func.invert_dict(func.get_cmd_dict())

    # take last five digits
    digits = obshistid[-5:]

    first = int(digits[0])
    second = int(digits[1])
    third = int(digits[2])
    fourth = int(digits[3])
    run = int(digits[4])

    instrument = instDictInv[first]
    field = fieldDictInv[second]
    position = positionDictInv[third]
    cmd = cmdDictInv[fourth]

    return {
        "instrument": instrument,
        "field": field,
        "position": position,
        "cmd": cmd,
        "run": run,
    }


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
                print(" ", sub_dir)

    return all_dirs


def get_dirs_obshistids(instruments, fields, positions, cmd_files, root_dir, run):
    obshistids = []
    work_dirs = []

    for instrument in instruments:
        for field in fields:
            for position in positions:
                for cmd_file in cmd_files:
                    # print(f"\n Running for {instrument} {field} {position}")
                    # build the name of input directory with
                    # repackaged simulated files
                    # work_dir = os.path.join(root_dir, instrument, field, position)

                    obshistid = func.calculate_obshistid(
                        instrument, field, position, cmd_file, run
                    )

                    work_dir = os.path.join(
                        root_dir, instrument, field, position, str(obshistid)
                    )
                    print(
                        f"\n {instrument} {field} {position} {cmd_file} is {obshistid} and {work_dir}"
                    )

                    obshistids.append(obshistid)
                    work_dirs.append(work_dir)
    return work_dirs, obshistids


def get_work_dirs_from_obshistids(obshistids, root_dir):
    work_dirs = []
    for obshistid in obshistids:
        dic = func.invert_obshistid(obshistid)
        work_dir = os.path.join(
            root_dir, dic["instrument"], dic["field"], dic["position"], str(obshistid)
        )
        work_dirs.append(work_dir)
    return work_dirs


def get_instruments_from_obshistids(obshistids):
    instruments = []
    for obshistid in obshistids:
        dic = func.invert_obshistid(obshistid)
        # print(dic)
        instruments.append(dic["instrument"])
    return instruments


def main(
    instruments,
    fields,
    positions,
    cmd_files,
    root_dir,
    run,
    dry_run,
    obshistids,
    clobber,
):

    default_dir = os.getcwd()  # get current working directory

    # if obshistids are not provided, generate them from provided
    # instrument, field, position, cmd_file
    if len(obshistids) < 1:
        print(
            "Building obshistids and work_dirs from the provided names of \
        instruments, fields, positions, cmd_files."
        )
        work_dirs, obshistids = get_dirs_obshistids(
            instruments, fields, positions, cmd_files, root_dir, run
        )

    # otherwise, use the given obshistids to resolve all other named parameters,
    # and find work_dirs
    elif len(obshistids) > 0:
        print(
            "Resolving obshistids to get names of \
        instruments, fields, positions, cmd_files."
        )

        work_dirs = get_work_dirs_from_obshistids(obshistids, root_dir)
        instruments = get_instruments_from_obshistids(obshistids)

    else:
        print(
            "Either obshistids or names of instruments, fields, positions, \
        cmd files must be provided"
        )

    # ingest and run ISR for each pair of work_dir, obshistid :

    for work_dir, obshistid, instrument in zip(work_dirs, obshistids, instruments):
        print(f"Ingesting and running ISR for {work_dir}")

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

            # if clobber, then delete already existing repo there
            repo_dir = os.path.join(work_dir, "DATA")
            if os.path.exists(repo_dir):
                print(f"There is already {repo_dir}")
                if clobber:
                    print(f"Deleting {repo_dir}")
                    # os.system('cmd /k "date"')
                    subprocess.call(f"rm -rf {repo_dir}", shell=True)
            # run the script to create butler and ingest
            print("Running sh  ./runIsr.sh")
            subprocess.call(["sh", "./runIsr.sh"])
            os.chdir(default_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Butler ingestion of phosim simulated data using PS1 catalogs.\
\
This program writes and executes butler scripts that create a gen3 repo, \
register appropriate instrument, ingest raws, define visits, register calibs, \
and do the instrument signature removal. \
The path to raw (repackaged) files is based on the obshistid, which \
encodes the information about instrument, field, position, cmd_file used. \
\
Two modes of operation are possible:\
\
a) --obshistids (-i) arg is provided, then --instruments, --fields, \
--positions, --cmd_files are ignored:\
obshistid --> {instrument,field,position,cmd_file}, work_dir \
\
b) -instruments, --fields, --positions, --cmd_files are provided, \
and based on these we calculate obshistid: \
{instrument,field,position,cmd_file} --> obshistid ,  work_dir \
\
"
    )

    parser.add_argument(
        "--instruments",
        nargs="+",
        default=["comCam"],
        help='A list of instruments, eg. "lsstCam", "comCam" (default: comCam)',
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
        help='A list of positions to simulate, eg. "focal", "extra", "intra". (default: focal)',
    )
    parser.add_argument(
        "--cmd_files",
        type=str,
        default=[
            "noBkgndPert00_NCSA.cmd",
            "noBkgndPert05_NCSA.cmd",
            "qckBkgndPert00_NCSA.cmd",
            "qckBkgndPert05_NCSA.cmd",
        ],
        help="Name of the physics command file(s) used by phosim (default: all four - {no/qck bkgnd}x{Pert00,Pert05})",
    )
    parser.add_argument(
        "--root_dir",
        type=str,
        default="/project/scichris/aos/AOS/DM-28360/",
        help="Absolute path to the work directory where .cmd and .inst \
files can be found. That is also where the output \
under  {instrument}{field}{position}{obshistid}\
will be written (default: /project/scichris/aos/AOS/DM-28360/)",
    )
    parser.add_argument(
        "--run",
        "-r",
        nargs=1,
        type=int,
        default=1,
        help="Run number for the ISR, assuming instrument, field, position, cmd_file\
are the same (default:1).",
    )
    parser.add_argument(
        "--dry_run",
        default=False,
        action="store_true",
        help="Do not run any simulation, just print commands used. (default: False)",
    )
    parser.add_argument(
        "--clobber",
        default=False,
        action="store_true",
        help="Delete the existing DATA gen3 repo. (default: False)",
    )

    parser.add_argument(
        "--obshistids",
        "-i",
        nargs="+",
        default=[],
        help="A list of 7-digit obshistids to ingest and ISR, eg. [9000001 9000101] (default:[]).\
If provided, ignoring values of instrument, field, position, cmd_file, since we \
use the numeric value of obshistid to generate these. \
Eg. 9001001 is comCam med focal noBkgndPert00.cmd. (default: None)",
    )

    args = parser.parse_args()
    main(
        instruments=args.instruments,
        fields=args.fields,
        positions=args.positions,
        cmd_files=args.cmd_files,
        root_dir=args.root_dir,
        run=args.run,
        dry_run=args.dry_run,
        obshistids=args.obshistids,
        clobber=args.clobber,
    )

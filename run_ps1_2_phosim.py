import subprocess
import os
import argparse


def write_readme(work_dir, phosim_command, inst_file,
                 cmd_file, repackager_command):
    """
    Store information about the .inst and .cmd
    phoSim input files used to run the simulation
    """
    filename = "README.txt"
    outfile = os.path.join(work_dir, filename)
    with open(outfile, "w") as output:

        output.write("Files in /raw/ were created with this command:\n\n\n")
        output.write(phosim_command)
        output.write("\n\n\n")
        output.write(f"The instance catalog was {inst_file}\n")
        output.write("\n")
        output.write(f"The physics command catalog was {cmd_file}\n")
        output.write("\n\n")
        s = "Files from /raw/ were repackaged to /repackaged/ with \n"
        output.write(s)
        output.write(repackager_command)


def main(
    instruments=["comCam"],
    fields=["high"],
    positions=["focal"],
    phosim_t=1,
    phosim_p=25,
    cmd="noBkgnd.cmd",
):

    # define paths
    phosim_path = "/project/scichris/aos/phosim_syseng4/phosim.py"
    root_dir = "/project/scichris/aos/ps1_phosim/"

    for instrument in instruments:
        for field in fields:
            for position in positions:
                print("\n", instrument, field, position)
                # define instance catalog and physics command file
                inst_file = os.path.join(
                    root_dir, f"stars_{instrument}_PS1_{field}_{position}.inst"
                )
                cmd_file = os.path.join(root_dir, cmd)

                if instrument == "lsstCam":
                    instr = "lsst"
                elif instrument == "comCam":
                    instr = "comcam"

                # ensure the output path exists
                work_dir = os.path.join(root_dir, instrument, field, position)
                out_dir = os.path.join(work_dir, "raw")
                if not os.path.exists(out_dir):
                    os.makedirs(out_dir)
                log_file = os.path.join(work_dir, "starPhoSim.log")

                # run phosim
                command = f"python {phosim_path} {inst_file} -i {instr} -e 1 \
    -c {cmd_file} -p {phosim_p} -t {phosim_t} -o {out_dir} > {log_file} 2>&1"
                phosim_command = command

                print(f"\nRunning via subprocess: \n {command}\n")
                if subprocess.call(command, shell=True) != 0:
                    raise RuntimeError("Error running: %s" % command)

                # repackage the output
                repackaged_dir = os.path.join(work_dir, "repackaged")
                command = f"phosim_repackager.py {out_dir} \
    --out_dir {repackaged_dir} --inst {instr}"
                print(f"\nRunning via subprocess: \n {command}\n")
                repackager_command = command
                if subprocess.call(command, shell=True) != 0:
                    raise RuntimeError("Error running: %s" % command)

                # store names of all files used by phosim to README file
                # for good bookkeeping
                write_readme(
                    work_dir, phosim_command,
                    inst_file, cmd_file, repackager_command
                )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run phosim for PS1 source catalogs")
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
    parser.add_argument(
        "--phosim_t",
        "-t",
        nargs=1,
        type=int,
        default=1,
        help="Phosim argument to multi-thread the calculation (M) on \
a per-astronomical-source basis. \
Note that there should be M*N cores available.",
    )
    parser.add_argument(
        "--phosim_p",
        "-p",
        nargs=1,
        type=int,
        default=25,
        help="Phosim argument to run N copies of raytrace. \
Note that there should be M*N cores available.",
    )

    args = parser.parse_args()
    main(
        instruments=args.instruments,
        fields=args.fields,
        positions=args.positions,
        phosim_t=args.phosim_t[0],
        phosim_p=args.phosim_p[0],
    )

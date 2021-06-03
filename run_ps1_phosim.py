import subprocess
import os
import argparse
import time
import run_ps1_functions as func

def write_readme(
    work_dir, phosim_command, inst_file, cmd_file, repackager_command, ttl_time
):
    """
    Store information about the .inst and .cmd
    phoSim input files used to run the simulation
    """
    filename = "README.txt"
    out_file = os.path.join(work_dir, filename)
    content = ["Files in /raw/ were created with this command:\n\n\n",
              phosim_command,
              "\n\n\n",
              f"The instance catalog was {inst_file}\n",
              "\n",
              f"The physics command catalog was {cmd_file}\n",
              "\n\n",
              "Files from /raw/ were repackaged to /repackaged/ with \n",
              repackager_command,
              f"Running phosim took {ttl_time} seconds"
              ]
    func.write_to_file(out_file, content)


def main(
    instruments,
    fields,
    positions,
    phosim_t,
    phosim_p,
    phosim_s,
    cmd_file,
    phosim_path,
    root_dir,
    run,
    dry_run,
):

    for instrument in instruments:
        for field in fields:
            for position in positions:
                print("\n", instrument, field, position)
                # define instance catalog and physics command file
                inst_file = f"stars_{instrument}_PS1_{field}_{position}.inst"
                inst_file_path = os.path.join(root_dir, inst_file)

                cmd_file_path = os.path.join(root_dir, cmd_file)

                obshistid = func.calculate_obshistid(
                    instrument, field, position, cmd_file, run
                )
                new_inst_file = inst_file[: -(len(".inst"))] + f"_{obshistid}.inst"
                inst_file_path = os.path.join(root_dir, new_inst_file)

                if instrument == "lsstCam":
                    instr = "lsst"
                elif instrument == "comCam":
                    instr = "comcam"

                # ensure the output path exists
                work_dir = os.path.join(
                    root_dir, instrument, field, position, str(obshistid)
                )
                out_dir = os.path.join(work_dir, "raw")
                if not dry_run:
                    if not os.path.exists(out_dir):
                        os.makedirs(out_dir)
                log_file = os.path.join(work_dir, "starPhoSim.log")

                # run phosim: beginning of the command
                command = f"python {phosim_path} {inst_file_path} -i {instr} -e 1 \
    -c {cmd_file_path} -w {out_dir} -p {phosim_p}  -o {out_dir} "

                # add sensor selection
                if len(phosim_s[0]) > 0:
                    sensor_names = func.sensor_list_to_string(phosim_s)
                    command += f' -s "{sensor_names}"'

                # end of the command
                command += f"> {log_file} 2>&1"
                phosim_command = command

                print(f"\nRunning via subprocess: \n {command}\n")
                t0 = time.time()
                if not dry_run:
                    if subprocess.call(command, shell=True) != 0:
                        raise RuntimeError("Error running: %s" % command)
                ttl_time = time.time() - t0
                print(f"Running phosim took {ttl_time:.3f} seconds")

                # repackage the output
                focuszDict = {"focal": 0, "intra": 1500, "extra": -1500}
                focusz = focuszDict[position]
                repackaged_dir = os.path.join(work_dir, "repackaged")
                command = f"phosim_repackager.py {out_dir} \
    --out_dir {repackaged_dir} --inst {instr} --focusz {focusz}"

                print(f"\nRunning via subprocess: \n {command}\n")
                repackager_command = command
                if not dry_run:
                    if subprocess.call(command, shell=True) != 0:
                        raise RuntimeError("Error running: %s" % command)

                # store names of all files used by phosim to README file
                # for good bookkeeping
                if not dry_run:
                    write_readme(
                        work_dir,
                        phosim_command,
                        inst_file_path,
                        cmd_file_path,
                        repackager_command,
                        ttl_time,
                    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run phosim for PS1 source catalogs")
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
        "--phosim_t",
        "-t",
        nargs=1,
        default=[1],
        help="Phosim argument to multi-thread the calculation (M) on \
a per-astronomical-source basis. \
Note that there should be M*N cores available.",
    )
    parser.add_argument(
        "--phosim_p",
        "-p",
        nargs=1,
        default=[25],
        help="Phosim argument to run N copies of raytrace. \
Note that there should be M*N cores available.",
    )
    parser.add_argument(
        "--phosim_s",
        "-s",
        nargs="+",
        default=[""],
        help="Phosim argument to specify which sensors to simulate. Eg. R22_S00 R22_S11 \
(default:"
        ", i.e. not specify anything)",
    )

    parser.add_argument(
        "--cmd_file",
        type=str,
        default="noBkgnd.cmd",
        help="Name of the physics command file used by phosim",
    )

    parser.add_argument(
        "--phosim_path",
        type=str,
        default="/project/scichris/aos/phosim_syseng4/phosim.py",
        help="Absolute path to phosim.py",
    )

    parser.add_argument(
        "--root_dir",
        type=str,
        default="/project/scichris/aos/ps1_phosim/",
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
        cmd_file=args.cmd_file,
        phosim_path=args.phosim_path,
        phosim_t=args.phosim_t[0],
        phosim_p=args.phosim_p[0],
        phosim_s=args.phosim_s,
        root_dir=args.root_dir,
        run=args.run,
        dry_run=args.dry_run,
    )

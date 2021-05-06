import os
import subprocess
import argparse
import time




def write_readme(work_dir, phosim_command, inst_file,
                 cmd_file,ttl_time):
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
        output.write("\n\n")
        output.write(f"Running phosim took {ttl_time} seconds")
        
        
        
def main(
    dir_name="test",
    phosim_p=24,
    cmd="noBkgnd.cmd",
    inst="stars_comCam_PS1_high_focal.inst",
    phosim_path="/project/scichris/aos/phosim_syseng4/phosim.py",
    root_dir="/project/scichris/aos/ps1_phosim/",
):

    instrument = "comCam"
    instr = "comcam"
    field = "high"
    position = "focal"
    sensor = "R22_S11"  # pick just one CCD

    print("\n", instrument, field, position)

    # define instance catalog and physics command file
    inst_file = os.path.join(root_dir, inst)
    cmd_file = os.path.join(root_dir, cmd)
    print(f"\nThe root_dir is {root_dir}")
    print(f"\nUsing {inst_file} and {cmd_file}")

    # ensure the output path exists
    work_dir = os.path.join(root_dir, "benchmark", dir_name)
    out_dir = os.path.join(work_dir, "raw")
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    print(f"\nSaving output in {work_dir}")
    log_file = os.path.join(work_dir, "starPhoSim.log")

    # run phosim
    command = f"python {phosim_path} {inst_file} -i {instr} -e 1 \
-c {cmd_file} -p {phosim_p} -w {work_dir} \
-s {sensor} -o {out_dir} > {log_file} 2>&1"

    print(f"\nRunning via subprocess: \n {command}\n")
    t0 = time.time()
    if subprocess.call(command, shell=True) != 0:
        raise RuntimeError("Error running: %s" % command)
    ttl_time = time.time() - t0
    print(f"Running phosim took {ttl_time:.3f} seconds")

    write_readme(work_dir, command, inst_file,
                 cmd_file, ttl_time)
   
    # repackage the output
    repackaged_dir = os.path.join(work_dir, "repackaged")
    command = f"phosim_repackager.py {out_dir} \
    --out_dir {repackaged_dir} --inst {instr}"
    print(f"\nRunning via subprocess: \n {command}\n")
    if subprocess.call(command, shell=True) != 0:
        raise RuntimeError("Error running: %s" % command)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run phosim for PS1 source catalogs")
    parser.add_argument(
        "--dir_name",
        default="test",
        help="Name of root dir under \
/project/scichris/aos/ps1_phosim/benchmark/",
    )
    parser.add_argument(
        "--phosim_p",
        "-p",
        nargs=1,
        type=int,
        default=[25],
        help="Phosim argument to run N copies of raytrace. \
Note that there should be N cores available.",
    )
    parser.add_argument(
        "--cmd",
        type=str,
        default="noBkgnd.cmd",
        help="Name of the physics command file used by phosim",
    )
    parser.add_argument(
        "--inst",
        type=str,
        default="stars_comCam_PS1_high_focal.inst",
        help="Name of the instance catalog file used by phosim",
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
files can be found. That is also where the output under \
benchmark/dir_name/ will be written",
    )

    args = parser.parse_args()
    main(
        dir_name=args.dir_name,
        phosim_p=args.phosim_p[0],
        cmd=args.cmd,
        inst=args.inst,
        phosim_path=args.phosim_path,
        root_dir=args.root_dir,
    )

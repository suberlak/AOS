from astropy.table import Table
import subprocess
import os

# read in the file containing coordinates
# for the catalog query
filename = "/project/scichris/aos/ps1_query_coordinates.txt"
gt = Table.read(filename, format="ascii")

# rewrite as dict for easier access of coordinates
# of a specific field by name
gt_dict = {}
for i in range(len(gt)):
    gt_dict[gt["name"][i]] = {"ra": gt["ra_deg"][i], "dec": gt["dec_deg"][i]}
all_fields = list(gt_dict.keys())

#  define which fields, instruments, and piston positions to simulate
instruments = ["comCam", "lsstCam"]  # ['comCam', 'lsstCam']#
fields = all_fields# ["high"]  # 
positions = ["focal", "defocal"]  # ['focal', 'defocal']

# define paths
phosim_path = "/project/scichris/aos/phosim_syseng4/phosim.py"
root_dir = "/project/scichris/aos/ps1_phosim/"

# choose cmd file 
cmd = "noBkgnd.cmd"  # or quick background ... 

def write_readme(work_dir, phosim_command, inst_file, cmd_file,
                repackager_command):
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
        output.write("Files from /raw/ were repackaged to /repackaged/ with \n")
        output.write(repackager_command)
        


for instrument in instruments:
    for field in fields:
        for position in positions:

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
-c {cmd_file} -p 25 -o {out_dir} > {log_file} 2>&1"
            phosim_command = command
            
            print(f"\nRunning via subprocess: \n {command}\n")
            if subprocess.call(command, shell=True) != 0:
                raise RuntimeError("Error running: %s" % command)

            # repackage the output
            repackaged_dir = os.path.join(work_dir, "repackaged")
            command = f"phosim_repackager.py {out_dir} --out_dir {repackaged_dir} --inst {instr}"
            print(f"\nRunning via subprocess: \n {command}\n")
            repackager_command = command
            if subprocess.call(command, shell=True) != 0:
                raise RuntimeError("Error running: %s" % command)

            # store names of all files used by phosim to README file
            # for good bookkeeping
            write_readme(work_dir, phosim_command, inst_file, cmd_file, 
                         repackager_command)

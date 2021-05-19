import subprocess
import argparse
import os


def write_to_file(out_file, content):
    with open(out_file, "w") as output:
        for line in content:
            output.write(line)


def write_slurm_script(
    python_script,
    instrument,
    field,
    position,
    phosim_p,
    cmd_file,
    root_dir,
    nodes,
    ntasks,
    job_name,
    slurm_file_name,
    slurm_path,
    time_limit,
):

    cmd = f"python {python_script} --instruments {instrument} --fields {field} \
    --positions {position} --phosim_p {phosim_p} --cmd_file {cmd_file} --root_dir {root_dir} \n"

    content = [
        "#!/bin/bash -l \n",
        "#SBATCH --partition normal \n" f"#SBATCH --nodes {nodes} \n",
        f"#SBATCH --ntasks {ntasks} \n",
        f"#SBATCH -t {time_limit}:00:00 \n",
        f"#SBATCH --job-name {job_name} \n",
        'echo "starting at `date` on `hostname`" \n',
        "pwd \n",
        cmd,
        'echo "ended at `date` on `hostname`" \n',
    ]
    out_file = os.path.join(slurm_path, slurm_file_name)
    write_to_file(out_file, content)

    return out_file


def submit_slurm_job(slurm_file):
    print(f"Running sbatch  {slurm_file}")
    subprocess.call(["sbatch", slurm_file])


def main(
    instruments,
    fields,
    positions,
    phosim_p,
    backgrounds,
    perts,
    suffix,
    python_script,
    root_dir,
    nodes,
    ntasks,
    dry_run,
    job_prefix,
    slurm_file,
    slurm_path,
    time_limit,
):

    counter = 0
    for instrument in instruments:
        for field in fields:
            for bkgnd in backgrounds:
                for pert in perts:
                    for position in positions:
                        print(f"\n{instrument} {field} {bkgnd} {pert} {position}")
                        cmd_file = f"{bkgnd}BkgndPert{pert}_{suffix}.cmd"
                        print(f"cmd_file:{cmd_file}")
                        job_name = f"{job_prefix}{counter}"  # eg. comHi12
                        print(f"job_name:{job_name}")
                        #  eg. runSlurmComHi12.sl
                        slurm_file_name = f"{slurm_file}{job_name}.sl"
                        print(f"slurm_file_name:{slurm_file_name}")
                        slurm_file_path = write_slurm_script(
                            python_script=python_script,
                            instrument=instrument,
                            field=field,
                            position=position,
                            phosim_p=phosim_p,
                            cmd_file=cmd_file,
                            root_dir=root_dir,
                            nodes=nodes,
                            ntasks=ntasks,
                            job_name=job_name,
                            slurm_file_name=slurm_file_name,
                            slurm_path=slurm_path,
                            time_limit=time_limit,
                        )
                        if not dry_run:
                            submit_slurm_job(slurm_file_path)
                        counter += 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Submit slurm jobs for simulating \
        PS1 source catalogs, calling python_script"
    )

    parser.add_argument(
        "--python_script",
        type=str,
        default="/project/scichris/aos/AOS/run_ps1_phosim.py",
        help="Absolute path to the python script that calls phosim.py\
(default: /project/scichris/aos/AOS/run_ps1_phosim.py)",
    )
    parser.add_argument(
        "--instruments",
        nargs="+",
        default=["comCam"],
        help='A list of instruments, eg. "lsstCam", "comCam"\
(default: "comCam")',
    )

    parser.add_argument(
        "--fields",
        nargs="+",
        default=["high"],
        help='A list of field names to generate, \
eg. "high", "med", "low", "Baade"\
(default: "high")',
    )

    parser.add_argument(
        "--positions",
        nargs="+",
        default=["focal", "extra", "intra"],
        help='A list of positions to simulate, eg. "focal", "extra", "intra".\
(default: all of them)',
    )
    parser.add_argument(
        "--backgrounds",
        nargs="+",
        default=["qck", "no"],
        help='A list of background prefixes to simulate, eg. "qck"(bkgnd), "no"(bgnd).\
(default: all of them)',
    )
    parser.add_argument(
        "--perts",
        nargs="+",
        default=["00", "05"],
        help='A list of perturbations to simulate, eg. "00" (small), "05" (large).\
(default: all of them)',
    )
    parser.add_argument(
        "--suffix",
        type=str,
        default="NCSA",
        help="Suffix for the cmd file, composed of \
{bkgnd}BkgndPert{pert}_{suffix}.cmd",
    )
    parser.add_argument(
        "--phosim_p",
        "-p",
        type=int,
        default=24,
        help="Phosim argument to run N copies of raytrace. \
Note that there should be M*N cores available, so here \
phosim_p <= nodes * ntasks, eg. if nodes=1,\
ntasks=24, phosim_p <= 24 (default: 24)",
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
        "--nodes",
        nargs=1,
        type=int,
        default=1,
        help="Number of nodes, parsed as #SBATCH --nodes {nodes} (default:1).",
    )
    parser.add_argument(
        "--ntasks",
        nargs=1,
        type=int,
        default=24,
        help="Number of tasks per node, usually one task per CPU, \
parsed as #SBATCH --ntasks {ntasks}, so \
eg. ntasks=24 with 24 CPU node (default:24).",
    )

    parser.add_argument(
        "--time_limit",
        type=int,
        default=300,
        help="Time limit for the job (hours), \
parsed as #SBATCH -t {time_limit}:00:00 {default:300}",
    )
    parser.add_argument(
        "--job_prefix",
        type=str,
        default="comHi",
        help="Prefix for the slurm job, parsed as \
#SBATCH --job-name {job_name} (default: comHi)",
    )
    parser.add_argument(
        "--slurm_file",
        type=str,
        default="runSlurm",
        help="Prefix for the slurm script filename,\
used as runSlurm{counter}.sl (default: runSlurm)",
    )
    parser.add_argument(
        "--slurm_path",
        type=str,
        default="/project/scichris/aos/AOS/",
        help="Path where slurm file should be saved,\
(default: /project/scichris/aos/AOS/)",
    )
    parser.add_argument(
        "--dry_run",
        default=False,
        action="store_true",
        help="Do not run actually submit any jobs, just\
write the slurm jobs and print the arguments parsed (default: False)",
    )

    args = parser.parse_args()

    main(
        instruments=args.instruments,
        fields=args.fields,
        positions=args.positions,
        phosim_p=args.phosim_p,
        backgrounds=args.backgrounds,
        perts=args.perts,
        suffix=args.suffix,
        python_script=args.python_script,
        root_dir=args.root_dir,
        nodes=args.nodes,
        ntasks=args.ntasks,
        job_prefix=args.job_prefix,
        slurm_file=args.slurm_file,
        slurm_path=args.slurm_path,
        dry_run=args.dry_run,
        time_limit=args.time_limit,
    )

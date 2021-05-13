import subprocess


def write_to_file(out_file, content):
    with open(out_file, "w") as output:
        for line in content:
            output.write(line)


def write_slurm_script(python_script = '/project/scichris/aos/AOS/run_ps1_phosim.py',
                   instrument = 'comCam',
                   field = 'high',
                   position = 'extra',
                   phosim_p = 24,
                   cmd_file = "noBkgndPert05_NCSA.cmd",
                   root_dir = '/project/scichris/aos/AOS/DM-28360/',
                   nodes = 1, ntasks=24, job_name="comHi12",
                   slurm_file_name = "runSlurmTest.sl"
                  ):

    cmd = f"python {python_script} --instruments {instrument} --fields {field} \
    --positions {position} --phosim_p {phosim_p} --cmd_file {cmd_file} --root_dir {root_dir} \n"
    
    content = ["#!/bin/bash -l \n",
               "#SBATCH --partition normal \n"
                f"#SBATCH --nodes {nodes} \n",
                f"#SBATCH --ntasks {ntasks} \n",
                "#SBATCH -t 1000:00:00 \n",
                f"#SBATCH --job-name {job_name} \n",
                'echo "starting at `date` on `hostname`" \n',
                'pwd \n',
                 cmd,
                'echo "ended at `date` on `hostname`" \n']
    out_file = f'/project/scichris/aos/AOS/{slurm_file_name}'
    write_to_file(out_file, content)

    return out_file

slurm_file = write_slurm_script()

def submit_slurm_job(slurm_file):
    print(f'Running sbatch  {slurm_file}')
    subprocess.call(['sbatch', slurm_file]) 

submit_slurm_job(slurm_file)

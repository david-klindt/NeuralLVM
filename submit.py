"""Minimal script to submit a job array to lsf."""
import argparse
import subprocess
from datetime import datetime
from pathlib import Path
import os
from os.path import join
import shutil


def get_project_root():
    """Get the absolute path to the repo's root"""
    return os.path.dirname(os.path.realpath(__file__))


def submit_lsf_job(lsf_file_string, bsub_args):
    """
    Run `bsub` to submit `lsf_file_string` with the lsf options given in `bsub_args`.
    """
    args = ["bsub"] + (bsub_args.split(" ") if bsub_args else [])
    with subprocess.Popen(
        args, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.STDOUT
    ) as process:
        process.stdin.write(bytes(lsf_file_string, "utf-8"))
        print(process.communicate()[0].decode().rstrip())


def main():
    """Entry point function."""

    root = get_project_root()
    save_path_base = '/home/scl1pal/projects/NeuralLVM/exp/'

    parser = argparse.ArgumentParser(
        description="Minimal script to submit a job to lsf scheduler"
    )
    parser.add_argument(
        "--command_file",
        type=str,
        default='run.sh',
        help="A file where each line contains the command for a single job in the array",
    )
    parser.add_argument(
        "--name", default='run', type=str
    )
    parser.add_argument(
        "--commit",
        type=str,
        default="dev_lukas",
        help="Which commit or branch to checkout before running the job.",
    )
    parser.add_argument(
        "--bsub_args",
        help="String of arbitrary bsub options, which override the defaults in lsf.bsub",
    )

    args = parser.parse_args()

    # number of jobs
    with open(args.command_file, encoding="utf-8") as cmd_f:
        n_jobs = sum(1 for _ in cmd_f)
    print('n jobs', n_jobs)
    # submit script template
    lsf_file = "lsf.bsub"
    print('root', root)

    target = join(save_path_base, f'{args.name}_{datetime.now().strftime("%Y-%m-%d-%H-%M-%S")}')
    os.makedirs(join(target, 'lsf_logs'))

    shutil.copytree(join(root, 'NeuralLVM'), join(target, 'NeuralLVM'))
    shutil.copy(join(root, 'run.sh'), join(target))

    print('target', Path(root).exists(), target)
    with open(lsf_file, "r", encoding="utf-8") as infile:
        lsf_file_string = infile.read()
    lsf_file_string = lsf_file_string.format(
        root=target, n=n_jobs, command=args.command_file
    )
    submit_lsf_job(lsf_file_string, args.bsub_args)


if __name__ == "__main__":
    main()

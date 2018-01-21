import argparse
import glob
import subprocess

parser = argparse.ArgumentParser()
parser.add_argument("-id", "--job_id", default='char_0', type=str, help="parent encoder of job")
parser.add_argument("-schedules", "--schedules", default=1, type=int, help="Number of additional jobs")

args = parser.parse_args()
run_id = args.job_id


files = glob.glob("schedulers/" + run_id + "*")
filename = "schedulers/" + run_id + "_0.sh"
with open(filename, 'r') as f:
    template = f.read()
num_files = len(files)
# Write all the new files
for i in range(args.schedules):
    file_num = i + num_files
    op = template.replace(
        "#SBATCH -o logs/" + run_id + "_0.log",
        "#SBATCH -o logs/" + run_id + "_" + str(file_num) + ".log"
    )
    with open('schedulers/' + run_id + "_" + str(file_num) + ".sh", 'w') as f:
        f.write(op)

# Schedule all the new jobs
for i in range(args.schedules):
    file_num = i + num_files
    command = \
        "sbatch -J " + run_id + " -d singleton " + "schedulers/" + run_id + "_" + str(file_num) + ".sh"
    print("Scheduling job #%d" % file_num)
    print(subprocess.check_output(command, shell=True))

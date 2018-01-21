import argparse

import datetime
import json
import subprocess
import sys
import yaml

parser = argparse.ArgumentParser()
parser.add_argument("-bc", "--base_config", default="char.yml", type=str, help="Base configuration to build upon")
parser.add_argument("-dd", "--data_dir", default="/share/data/lang/users/kalpesh/swbd", type=str, help="Data directory")
parser.add_argument("-ds", "--dataset", default="swbd.char", type=str, help="Dataset to operate on")
parser.add_argument("-id", "--job_id", default="char_0", type=str, help="Job ID")
parser.add_argument("-lm", "--loss_mode", default="l1", type=str, help="loss mode to run experiment on")
parser.add_argument("-extra", "--extra", default="", type=str, help="extra constants")
parser.add_argument("-dc", "--config", default="{}", type=str, help="Changed parameters for config")
parser.add_argument("-schedules", "--schedules", default=1, type=int, help="# of schedules")

args = parser.parse_args()

with open("config/%s" % args.base_config, 'r') as f:
    data = yaml.load(f)

changed_parameters = json.loads(args.config)
for k, v in changed_parameters.items():
    data[k] = v

schedule_script = "#!/bin/bash\n\n#SBATCH -p speech-gpu\n#SBATCH -o logs/{5}.log\n\n" + \
    "export LD_PRELOAD=\"/share/data/speech/Software/tcmalloc/lib/libtcmalloc.so\"\n" + \
    "/share/data/speech/Software/anaconda/bin/python \\\n" + \
    "/share/data/lang/users/kalpesh/Language-Models/final_code/codes/{0}/train.py --data_dir {1} --dataset {2} --loss_mode {3} {4}  \\\n" + \
    "--job_id job_{0} --config /share/data/lang/users/kalpesh/Language-Models/final_code/config/{0}.yml{6}\n"

# Copying version of code
files = ['train.py', 'model', 'config', 'utils']
for f in files:
    command = "mkdir codes/{0}; cp -r /share/data/lang/users/kalpesh/Language-Models/final_code/{1} codes/{0}/".format(args.job_id, f)
    subprocess.check_output(command, shell=True)


with open('config/%s.yml' % args.job_id, 'w') as f:
    yaml.dump(data, f, default_flow_style=False)

for i in range(args.schedules):
    with open('schedulers/%s_%d.sh' % (args.job_id, i), 'w') as f:
        f.write(schedule_script.format(
            args.job_id, args.data_dir, args.dataset, args.loss_mode, args.extra, args.job_id + "_%d" % i, ""
        ))

with open('schedulers/%s_test.sh' % args.job_id, 'w') as f:
    f.write(schedule_script.format(
        args.job_id, args.data_dir, args.dataset, args.loss_mode, args.extra, args.job_id + "_%d" % i, " --mode test"
    ))

with open('schedulers/%s_valid.sh' % args.job_id, 'w') as f:
    f.write(schedule_script.format(
        args.job_id, args.data_dir, args.dataset, args.loss_mode, args.extra, args.job_id + "_%d" % i, " --mode valid"
    ))

with open('schedulers/%s_generate.sh' % args.job_id, 'w') as f:
    f.write(schedule_script.format(
        args.job_id, args.data_dir, args.dataset, args.loss_mode, args.extra, args.job_id + "_%d" % i, " --mode generate"
    ))


for i in range(args.schedules):
    command = "sbatch -J {0} -d singleton schedulers/{0}_{1}.sh".format(args.job_id, i)

    print(subprocess.check_output(command, shell=True))

schedule_command = "python " + " ".join(sys.argv)
output = datetime.datetime.now().strftime("%Y-%m-%d %H:%M") + "\t" + args.job_id + "\n" + schedule_command + "\n"
with open("experiments.txt", "a") as f:
    f.write(output)

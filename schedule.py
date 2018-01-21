import datetime
import itertools
import subprocess
import yaml

# Adam optimizer scheduler script
optimizer = 'sgd'
lr_values = [1.0, 0.5, 0.1, 0.05]
lr_decay_values = [0.6, 0.8, 0.9]

lists = [lr_values, lr_decay_values]
lists = list(itertools.product(*lists))

with open("config/default.yml", 'r') as f:
    data = yaml.load(f)

data['optimizer'] = optimizer

schedule_script = "#!/bin/bash\n\n#SBATCH -p speech-gpu\n#SBATCH -o logs/{0}.log\n\n" + \
    "export LD_PRELOAD=\"/share/data/speech/Software/tcmalloc/lib/libtcmalloc.so\"\n" + \
    "/share/data/speech/Software/anaconda/bin/python \\\n" + \
    "/share/data/lang/users/kalpesh/Language-Models/final_code/train.py --loss_mode l1  \\\n" + \
    "--job_id job_{0} --config /share/data/lang/users/kalpesh/Language-Models/final_code/config/{0}.yml\n"

for lr, lr_decay in lists:
    data['lr'] = lr
    data['lr_decay'] = lr_decay
    name = "l1_%s_%s_%s" % (optimizer, lr, lr_decay)

    with open('config/%s.yml' % name, 'w') as f:
        yaml.dump(data, f, default_flow_style=False)

    with open('schedulers/%s.sh' % name, 'w') as f:
        f.write(schedule_script.format(name))

    command = "sbatch -J {0} schedulers/{0}.sh".format(name)

    print(subprocess.check_output(command, shell=True))

    output = datetime.datetime.now().strftime("%Y-%m-%d %H:%M") + "\t" + name + "\n"
    with open("experiments.txt", "a") as f:
        f.write(output)

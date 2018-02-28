"""This file builds up a grid search of parameters."""
import os
import itertools
import time
from config.arguments import parser


rnn_size = [128, 256, 400]
keep_prob = [0.6, 0.7, 0.8]
lr = [0.001, 0.002, 0.005]
l1 = 1
l2 = 0

lists = [rnn_size, keep_prob, lr]
lists = list(itertools.product(*lists))

SUMMARY = "experiments.log"
PERPLEXITY = "plot_data_eval.csv"
JOBS_PARALLEL = 2
SKIP_TILL = 0
NUM_EPOCHS = 5


def check_running():
    """Check whether jobs still running."""
    global SUMMARY
    global JOBS_PARALLEL
    global NUM_EPOCHS
    with open(SUMMARY, 'r') as f:
        data = f.read()

    jobs = data.split("\n\n")
    save_dirs = []
    for job in jobs:
        if job.strip() == "":
            continue
        command = job.split("\n")[-1].split(' ')
        save_dirs.append(command[command.index('--save_dir') + 1])
    relevant = save_dirs[-1 * JOBS_PARALLEL:]
    for r in relevant:
        path = os.path.join(r, PERPLEXITY)
        if os.path.exists(path) is False:
            return True
        with open(path, 'r') as f:
            data = f.readlines()
            if len(data) < NUM_EPOCHS:
                return True
    return False


args = parser.parse_args()

for index, c in enumerate(lists):
    if index < SKIP_TILL:
        continue
    command = "python schedule.py " + \
              "--data_dir " + args.data_dir + " " + \
              "--filename " + args.filename + " " + \
              "--eval_text " + args.eval_text + " " + \
              "--save_dir save/ " + \
              "--rnn_size " + str(c[0]) + " " + \
              "--keep_prob " + str(c[1]) + " " + \
              "--learning_rate " + str(c[2]) + " " + \
              "--loss_mode l1 " + \
              "--num_epochs " + str(NUM_EPOCHS)
    os.system(command)
    if index % JOBS_PARALLEL == (JOBS_PARALLEL - 1):
        # Time to pause and wait for completion
        # Every one hour, check for completion of all jobs
        running = True
        time_taken = 0
        while running is True:
            if time_taken > 0:
                print "Running for " + str(time_taken / 4.0) + "hours"
            running = check_running()
            time.sleep(900)
            time_taken += 1

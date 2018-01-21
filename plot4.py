import itertools
import os
import re

import matplotlib
matplotlib.use('Agg')

from matplotlib import pyplot
from matplotlib.backends.backend_pdf import PdfPages



pattern = re.compile(
    r'Perplexity\safter\s(\d*)\ssteps\s-\s(\d*.?\d*)\s'
)

pp = PdfPages('test.pdf')
pyplot.figure()
pyplot.clf()


# # Adam optimizer scheduler script
# optimizer = ['sgd']
# lr_values = [1.0, 0.5, 0.1, 0.05]
# lr_decay_values = [0.6, 0.8, 0.9]

# lists = [optimizer, lr_values, lr_decay_values]
# lists1 = list(itertools.product(*lists))

# # Adam optimizer scheduler script
# optimizer = ['adam']
# lr_values = [0.005, 0.001, 0.0005, 0.0001]
# lr_decay_values = [0.6, 0.8, 0.9]

# lists = [optimizer, lr_values, lr_decay_values]
# lists2 = list(itertools.product(*lists))

# lists = lists1 + lists2

colors = [
    '#e6194b', '#3cb44b', '#ffe119', '#0082c8', '#f58231', '#911eb4',
    '#46f0f0', '#f032e6', '#008080', '#800000', '#808000', '#808080'
]

# jobs = ["ptb_l1", "ptb_l1_l3_norm_80_2", "ptb_l1_l3_norm_999", "ptb_l1_l3_norm_50", "ptb_l1_l3_norm_80", "ptb_l1_l3_norm_95", "ptb_l1_l3_norm_99", "l1_sgd_1.0_0.8", "ptb_l1_unstable"]

jobs = ["ptb_l1_orig_plus_gen", "ptb_l1"]

# for i, (optimizer, lr, lr_decay) in enumerate(lists):
for i, job in enumerate(jobs):
    # name = "l1_%s_%s_%s" % (optimizer, lr, lr_decay)
    if job[0] == 'p':
        data = ""
        for j in range(10):
            if os.path.exists('logs/%s_%d.log' % (job, j)):
                with open('logs/%s_%d.log' % (job, j), 'r') as f:
                    data += f.read()
    else:
        with open('logs/%s.log' % job, 'r') as f:
            data = f.read()
    matches = re.findall(pattern, data)
    x = []
    y = []
    for match in matches:
        x.append(int(match[0]) / 1326)
        y.append(min(120, float(match[1])))
    pyplot.plot(x, y, label=job, color=colors[i])

art = []
lgd = pyplot.legend(loc=9, bbox_to_anchor=(0.5, -0.1), ncol=2)
art.append(lgd)

pyplot.xlabel('Epoch')
pyplot.ylabel('Perplexity')
pp.savefig(additional_artists=art, bbox_inches="tight")
pp.close()

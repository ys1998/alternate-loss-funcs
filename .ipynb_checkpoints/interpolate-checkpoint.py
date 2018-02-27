"""Interpolation code."""
import argparse
import math
parser = argparse.ArgumentParser()
parser.add_argument('--model1', type=str, default='model1.txt',
                    help='Filename having textual data in data_dir.')
parser.add_argument('--model2', type=str, default='output.txt',
                    help='Filename having textual data in data_dir.')
parser.add_argument('--constant', type=float, default=0.5,
                    help='Filename having textual data in data_dir.')
parser.add_argument('--mode', type=str, default="interpolate", choices=["interpolate", "max", "tune"],
                    help='Filename having textual data in data_dir.')
args = parser.parse_args()
constant = args.constant

with open(args.model1, 'r') as f:
    data = f.read().split('\n')
with open(args.model2, 'r') as f:
    data2 = f.read().split('\n')

if args.mode == 'tune':
    best_ppl = 100000000
    best_constant = 0.0
    for c in range(0, 101):
        print str(c) + " / 100"
        constant = c / 100.0
        ce = 0
        total = 0
        for i, token in enumerate(data):
            if len(token.strip()) == 0:
                continue
            if token.split()[0] != data2[i].split()[0]:
                print "Error"
            if token.split()[0] != '<s>':
                total += 1
            prob = constant * float(token.split()[1]) + (1 - constant) * float(data2[i].split()[1])
            ce += math.log(prob)

        ce = ce / len(data)
        ppl = math.exp(-1 * ce)
        if ppl < best_ppl:
            best_ppl = ppl
            best_constant = constant
    print "Best ppl - " + str(best_ppl)
    print "Best constant - " + str(best_constant)
else:
    ce = 0
    total = 0
    for i, token in enumerate(data):
        if len(token.strip()) == 0:
            continue
        if token.split()[0] != data2[i].split()[0]:
            print "Error"
        if token.split()[0] != '<s>':
            total += 1
        if args.mode == 'interpolate':
            prob = constant * float(token.split()[1]) + (1 - constant) * float(data2[i].split()[1])
        else:
            prob = max(float(token.split()[1]), float(data2[i].split()[1]))
        ce += math.log(prob)

    ce = ce / len(data)
    ppl = math.exp(-1 * ce)
    print ppl

"""Interpolation code."""
import argparse
import math
parser = argparse.ArgumentParser()
parser.add_argument('--model1', type=str, default='model1.txt',
                    help='Filename having textual data in data_dir.')
parser.add_argument('--model2', type=str, default='output.txt',
                    help='Filename having textual data in data_dir.')
args = parser.parse_args()

with open(args.model1, 'r') as f:
    data = f.read().split('\n')
with open(args.model2, 'r') as f:
    data2 = f.read().split('\n')

for i, token in enumerate(data):
    if len(token.strip()) == 0:
        continue
    if token.split()[0] != data2[i].split()[0]:
        print "Error"
    print token.split()[1] + " " + data2[i].split()[1]

import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--filename', type=str, default='LM',
                    help='Filename having textual data in data_dir.')
parser.add_argument('--limit', type=int, default=10,
                    help='Filename having textual data in data_dir.')

args = parser.parse_args()

with open(args.filename, 'r') as f:
    data = f.read()

data = data.replace('\n', ' </s> ')
data = data.split()

chars = {}

for i in data:
    if i in chars:
        chars[i] += 1
    else:
        chars[i] = 1

print(chars)

for k, _ in sorted(chars.items(), key=lambda x: x[1])[:args.limit]:
    ngrams = {}
    for i, d in enumerate(data):
        if d == k and i != 0:
            key = data[i - 1] + ' ' + d
            if key in ngrams:
                ngrams[key] += 1
            else:
                ngrams[key] = 1
    print(ngrams)
    print("The number of ngrams are %d" % len(ngrams))

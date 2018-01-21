import argparse
import codecs


def produce_ascii(number):
    output = ''
    while number > 0:
        output += chr(97 + (number % 26))
        number = number / 26
    return output


parser = argparse.ArgumentParser()
parser.add_argument('--file', type=str, default=None, help='The location of all the data')
parser.add_argument('--vocab', type=str, default=None, help='The location of all the data')
parser.add_argument('--map', type=str, default=None, help='The location of all the data')

args = parser.parse_args()

token_map = {}

if args.map is None:
    # Map not yet generated
    with codecs.open(args.vocab, 'r', 'utf-8') as f:
        vocab = f.read().split()

    for i, v in enumerate(vocab):
        if v == '<s>' or v == '</s>' or v == '<unk>':
            token_map[v] = v
        else:
            token_map[v] = produce_ascii(i)

    # confirming valid token map
    tokens = {}
    for i in token_map.values():
        if i in tokens:
            print("Hell on earth")
        else:
            tokens[i] = 1

    output = ""
    for k, v in token_map.items():
        output += k + " " + v + "\n"

    with codecs.open('map.txt', 'w', 'utf-8') as f:
        f.write(output)
else:
    with codecs.open(args.map, 'r', 'utf-8') as f:
        data = f.read().split()
    for i in range(0, len(data), 2):
        token_map[data[i]] = data[i + 1]

if args.file is not None:
    with codecs.open(args.file, 'r', 'utf-8') as f:
        data = f.read().split('\n')

    output = ""
    for line in data:
        if len(line) == 0:
            continue
        output += ' '.join([token_map[x] for x in line.split()])
        output += '\n'

    with open(args.file + ".map", 'w') as f:
        f.write(output)

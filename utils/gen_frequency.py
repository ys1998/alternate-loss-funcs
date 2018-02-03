"""This file uses SRILM to generate frequency distributions."""
import argparse
import codecs
import collections
import json
import multiprocessing
import numpy as np
import os
import re
import subprocess
import sys
import time

from processor import DataLoader
from helpers import Node
from six.moves import cPickle

# This RE is used to parse output produced by SRILM
regex = re.compile(r'\sp\(\s(.*)\s\|.*\]\s(.*)\s\[')


def load_counts(count_file, depth):
    """Read a counts file and build a dictionary."""
    with codecs.open(count_file, 'r', 'utf-8') as f:
        lines = f.readlines()
    counts = {}
    # Converting text count file into a dictionary
    # This dictionary is used in this function and later in code
    for line in lines:
        words = line.split()
        counts[" ".join(words[:-1])] = int(words[-1])
    # Building trie structure of data
    # Also building context dictionary for top-n words
    # Here `depth` represents the height of the trie node
    base = Node('', depth)
    total_keys = len(counts)
    k = 0
    for key, value in counts.items():
        k += 1
        if k % 100000 == 0:
            print(str(k) + " / " + str(total_keys) + " keys done")
        # `active` is always a node in the trie
        active = base
        tokens = key.split()
        for i, token in enumerate(tokens):
            # Storing the counts in a frequency dictionary.
            # Here the keys represent depth. For example,
            # {0: 4, 1: 23, 2: 34, 3: 4} represents -
            # 4 tokens do not have any children
            # 23 tokens below `active` with height 1,
            # 34 tokens below `active` with height 2,
            # 4 tokens below `active` with height 3
            # Note that `tokens` used above is not the same as # of branches
            # They are scaled by frequencies of occurence of that branch
            active.frequency[len(tokens) - i] += value
            # Stepping deeper into the trie
            if token in active.children:
                active = active.children[token]
            else:
                new = Node('', depth)
                active.children[token] = new
                active = new
        # The gram has finished, so it makes sense to update the leaf node's
        # frequency value with 0 children
        active.frequency[0] += value
    return base, counts


def get_constants(count_graph, gram):
    """The function returns interpolation constants."""
    # This formula will be outlined in the paper.
    word = gram[-1]
    context = gram[:-1]
    active = count_graph
    # Traverse trie to get to ending of context
    for token in context:
        active = active.children[token]
    # Number of occurences of `gram`
    # frequency[0] has been taken since we don't want add counts of
    # N-grams longer than `gram`, but prefixed by same gram
    # Similar logic for denominator
    numerator = float(active.children[word].frequency[0])
    denominator = float(active.frequency[1])
    return numerator / denominator


def get_children(count_graph, context):
    """The function returns tokens with higher probability given context."""
    # This algorithem returns all the tokens having been preceded
    # by the whole `context` or a part of the `context`. All
    # smoothing algorithms will give these tokens higher probability
    # along with those tokens which have higher unigram probability.
    # This algorithm will not necessarily return the best unigrams.
    # Those are explicitely appended with `top_unigrams` list.
    context = context.split()
    children = {}
    while len(context) >= 1:
        active = count_graph
        # Get to the end of trie for this context
        for c in context:
            active = active.children[c]
        # Add all the children for this context
        for k, v in active.children.iteritems():
            if k not in children:
                children[k] = 1
        # Move to a smaller context and fetch its children next time
        context = context[1:]
    return children.keys()


parser = argparse.ArgumentParser()

parser.add_argument('--data_dir', type=str, default='data',
                    help='Path to data directory to be worked upon.')
parser.add_argument('--srilm', type=str, default='../../srilm/bin/i686-m64',
                    help='Path to binaries of SRILM.')
parser.add_argument('--filename', type=str, default='input.txt',
                    help='Filename having textual data in data_dir.')
parser.add_argument('--order', type=int, default=3,
                    help='N-gram order of the model to be generated')
parser.add_argument('--unigram_k', type=int, default=10,
                    help='Level down to which probabilities to be captured')
parser.add_argument('--vocab_file', type=str, default="vocab.pkl",
                    help='Vocabulary file located in data_dir')
parser.add_argument('--use_vocab', dest='use_vocab', action='store_true',
                    help="Use a precomputed vocab file")
parser.add_argument('--no-use_vocab', dest='use_vocab', action='store_false')
parser.set_defaults(use_vocab=False)
parser.add_argument('--count_file', type=str, default="counts.txt",
                    help='Count file located in data_dir. Generated by SRILM')
parser.add_argument('--ngram_file', type=str, default="ngram-lm",
                    help='N-Gram model file located in data_dir. Generated by SRILM')
parser.add_argument('--constants_file', type=str, default="const.npy",
                    help='N-Gram constants file located in data_dir')
parser.add_argument('--frequencies_file', type=str, default="freq.json",
                    help='N-Gram frequencies file located in data_dir')
parser.add_argument('--map_file', type=str, default="map.pkl",
                    help='N-Gram map file located in data directory')

parser.add_argument('--freq', dest='freq', action='store_true',
                    help='Whether frequencies should be calculated')
parser.add_argument('--no-freq', dest='freq', action='store_false')
parser.set_defaults(freq=True)
parser.add_argument('--num_processes', type=int, default=None,
                    help='Number of processes to run in parallel.')
parser.add_argument('--parallel', dest='parallel', action='store_true',
                    help='Whether parallelization is needed.')
parser.add_argument('--no-parallel', dest='parallel', action='store_false')
parser.set_defaults(parallel=True)
parser.add_argument('--full-distro', dest='full_distro', action='store_true',
                    help='Whether or not full distribution should be calculated.')
parser.add_argument('--no-full-distro', dest='full_distro', action='store_false')
parser.set_defaults(full_distro=False)

# Re-using the DataLoader code
# Not running the `build_extra()` method
args = parser.parse_args()
data_loader = DataLoader(args)

# Saving some memory, as this is only used during training
data_loader.data = None
vocab = data_loader.vocab
text = data_loader.text

# Getting all the file paths
ngram_count = os.path.join(args.srilm, "ngram-count")
count_file = os.path.join(args.data_dir, args.count_file)
lm_file = os.path.join(args.data_dir, args.ngram_file)
constants_file = os.path.join(args.data_dir, args.constants_file)
frequencies_file = os.path.join(args.data_dir, args.frequencies_file)
map_file = os.path.join(args.data_dir, args.map_file)

# Start off by building an N-Gram language model for this dataset
# Here we also store counts to build interpolation constants
# Witten-Bell discounting has been used for the distributions
# Input file is assumed to have <unk> tokens
command = \
    ngram_count + " " \
    "-unk " + \
    "-order " + str(args.order) + " " + \
    "-text " + os.path.join(args.data_dir, args.filename) + " " + \
    "-kndiscount1 -kndiscount2 -kndiscount3 " + \
    "-write " + count_file + " " + \
    "-lm " + lm_file

print subprocess.check_output(command, stderr=subprocess.STDOUT, shell=True)

# Now that model is built, get the interpolation constants and frequencies
# Description of function given in definition above
count_graph, count_dict = load_counts(count_file, args.order)
tokens = text.split()
constants = []
context_map = []
# This is a dictionary of all possible context encountered
# This is used in frequency distro generation
contexts = {'': 1}

for i, token in enumerate(tokens):
    if i == 0:
        # This token is never predicted
        continue
    if i % 100000 == 0:
        print(str(i) + " / " + str(len(tokens)) + " tokens calculated")
    # In this gram, we are prediction gram[-1] using context gram[:-1]
    gram = tokens[max(0, i - args.order + 1):i + 1]
    # Accounting for N-Grams across sentences
    while " ".join(gram) not in count_dict:
        gram = gram[1:]
    if len(gram) == 1 and gram[0] == "<s>":
        # Since we are sure of <s>
        constants.append(1)
        context_map.append("")
    elif len(gram) == 1:
        # Should never happen unless it's a 1gram order
        # The code assumes we aren't going to pass arg.order == 1
        print("Error!")
        sys.exit()
    else:
        # Returns the interpolation constant for prediction i'th
        # token. Note that it is stored in (i-1)'th location.
        constants.append(get_constants(count_graph, gram))
        context = " ".join(gram[:-1])
        # This maps i to the corresponding `context`.
        # Used in BatchLoader
        context_map.append(context)
        if context not in contexts:
            contexts[context] = 1

# To make it same size as data
# Since we skipped evaluation of i == 0
constants.append(1)
context_map.append('')
constants = np.array(constants)
np.save(constants_file, constants)
with open(map_file, 'wb') as f:
        cPickle.dump(context_map, f)

# Saving some memory
del constants
del context_map

# --------Frequency distribution generation begins here--------

# This binary is used to generate probability distributions
ngram = os.path.join(args.srilm, "ngram")
command = \
    ngram + " " + \
    "-unk " + \
    "-order " + str(args.order) + " " + \
    "-lm " + lm_file + " " + \
    "-debug 2 "

# Get the most frequent unigram-tokens.
# These will have a high smoothed probability
top_unigrams = []
for k, v in vocab.iteritems():
    if v < args.unigram_k:
        top_unigrams.append(k)


def calc_frequency(context):
    """The function returns a dictionary having approx probability distro."""
    global count_file
    global command
    global vocab
    global top_unigrams
    global count_graph
    global args
    if context == '':
        # Return a distribution with all probability on <s>
        # A context == '' is only possible with 1-gram order or <s>
        special = {}
        special[1.0] = ['<s>']
        return {"distro": special, "total": 1.0}

    # This is a temporary file stored in data_dir
    # Hashing is a work-around to produce a different file in parallel processes
    count_file += str(hash(context))
    command += "-counts " + count_file
    output = ""
    # Generate the probabilites for all words
    if args.full_distro is True:
        keys = vocab.keys()
    else:
        # Use only more likely contexts
        keys = get_children(count_graph, context)
        # Add the best k unigrams to this, if not already present
        keys.extend(x for x in top_unigrams if x not in keys)
    # This file will act as an input to SRILM
    # This is an idea referred to by SRILM's owner in
    # https://mailman.speech.sri.com/pipermail/srilm-user/2017q1/001748.html
    for word in keys:
        output += context + " " + word + " 1\n"
    with codecs.open(count_file, "w", 'utf-8') as f:
        f.write(output)
    result = subprocess.check_output(command,
                                     stderr=subprocess.STDOUT,
                                     shell=True)
    # result now stores this probability distribution
    result = result.split("\n")

    distro = {}
    total_prob = 0.0
    # Each line of result is parsed using `regex`
    for word in result:
        dataset = regex.search(word)
        if not dataset:
            continue
        # In the `regex`, dataset.group(2) stores probability
        # dataset.group(2) stores the word
        prob = float(dataset.group(2))
        total_prob += prob
        # Note that distro is an "inverse" mapping
        # Here keys are probabilities and values are lists of words
        # This was done to save space by ~2 times
        if prob not in distro:
            distro[prob] = []
        distro[prob].append(dataset.group(1))
    # Removing temporary file
    os.remove(count_file)
    return {'distro': distro, 'total': total_prob}


# Multiprocessing algorithm for generating frequencies
if args.freq is True:
    # We don't need the frequencies for all tokens!
    # Just the unique contexts. For PTB, ~0.33 contexts were unique
    contexts = contexts.keys()

    # This is done to have 1 process per processor
    if args.parallel is True and args.num_processes is None:
        processes = multiprocessing.cpu_count()
    elif args.parallel is True:
        processes = args.num_processes
    else:
        processes = 1

    for i in range(0, len(contexts), processes):
        print str(i) + " / " + str(len(contexts))
        start = time.time()
        frequencies = {}
        if args.parallel is True:
            # Cutting up `contexts` list to divide it among processes
            upper_limit = min(i + processes, len(contexts))
            pool = multiprocessing.Pool(processes=processes)
            outputs = pool.map(calc_frequency, contexts[i:upper_limit])
            pool.terminate()
        else:
            # No need to split up processes if we are going 1 context
            # at a time.
            outputs = [calc_frequency(contexts[i])]

        for j, distro in enumerate(outputs):
            frequencies[contexts[i + j]] = distro

        # Append JSONs to a hard disk file. Done to spare the RAM
        with open(frequencies_file, 'a') as f:
            f.write(json.dumps(frequencies) + '\n')
        end = time.time()
        print "Time taken - " + str(end - start) + " seconds"

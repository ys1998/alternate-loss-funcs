"""This file builds the character trie and generates frequencies."""
import argparse
import numpy as np
import os
import sys

from helpers import Node
from processor import DataLoader
from six.moves import cPickle


def build_trie(tokens):
    """Build a character trie with token list."""
    base = Node('')
    # First convert tokens into a list of words
    # <s> is considered as a start of a word
    # </s> and _ are considered as ends of word
    words = []
    curr_word = []
    for token in tokens:
        curr_word.append(token)
        if token == "</s>" or token == "_":
            words.append(curr_word)
            curr_word = []
    for word in words:
        curr_base = base
        for char in word:
            # Create a leaf node
            if char not in curr_base.children:
                curr_base.children[char] = Node(char)
            curr_base.frequency[0] += 1
            # Move deeper into the trie
            curr_base = curr_base.children[char]
        curr_base.frequency[0] += 1
    return base


def get_probability(char_trie, curr_word, vocab):
    """Get the character probability from trie."""
    context = curr_word
    base = char_trie
    for char in context:
        if char not in base.children:
            return (1.0 / len(vocab)) * np.ones(len(vocab))
        else:
            base = base.children[char]

    distro = np.zeros(len(vocab))
    for char in vocab.keys():
        if char not in base.children:
            # The probability is zero
            distro[vocab[char]] = 0
        else:
            distro[vocab[char]] = \
                base.children[char].frequency[0] / float(base.frequency[0])
    return distro


parser = argparse.ArgumentParser()
parser.add_argument('--data_dir', type=str, default='data',
                    help='Path to data directory to be worked upon.')
parser.add_argument('--filename', type=str, default='input.txt',
                    help='Filename having textual data in data_dir')
parser.add_argument('--word_file', type=str, default='input.txt',
                    help='Filename having corresponding word tokens')
parser.add_argument('--vocab_file', type=str, default="char_vocab.pkl",
                    help='Vocabulary file located in data directory')
parser.add_argument('--word_vocab_file', type=str, default="vocab.pkl",
                    help='Word vocab file located in data directory')
parser.add_argument('--use_vocab', dest='use_vocab', action='store_true')
parser.add_argument('--no-use_vocab', dest='use_vocab', action='store_false')
parser.set_defaults(use_vocab=False)
parser.add_argument('--word_constants', type=str, default="const.npy",
                    help='These are read and used to generate character constants')
parser.add_argument('--constants_file', type=str, default="char_const.npy",
                    help='Output filename for character constants')
parser.add_argument('--frequencies_file', type=str, default="char_freq.pkl",
                    help='Output filename for frequency distro')
parser.add_argument('--map_file', type=str, default="char_map.pkl",
                    help='Output file mapping index to contexts')

# Re-using the DataLoader code
# Not running the `build_extra()` method
args = parser.parse_args()
data_loader = DataLoader(args)

# Saving some memory, as this is only used during training
data_loader.data = None
vocab = data_loader.vocab
text = data_loader.text
tokens = text.split()
# Building a character trie. It's described in the paper as well
char_trie = build_trie(tokens)

# Reading word files
args.filename = args.word_file
args.vocab_file = args.word_vocab_file
data_loader = DataLoader(args)
word_tokens = data_loader.text.split()
# This pointer refers to the active word in loop below
word_pointer = 0
word_constants_path = os.path.join(args.data_dir, args.word_constants)
# Load the word level interpolation constants
word_constants = np.load(word_constants_path)

curr_word = []
debug = False
char_constants = []
char_probability = {}
context_map = []

for i, token in enumerate(tokens):
    # Adding a check to ensure word_pointer is not getting astray
    test = curr_word
    if len(test) > 0 and test[0] == "<s>":
        test = test[1:]
    if not word_tokens[word_pointer].startswith("".join(test)):
        print "Word - Character file mismatch"
        sys.exit()

    if i % 100000 == 0:
        print str(i) + " / " + str(len(tokens)) + " completed"
    if i == 0:
        # This token is never predicted
        # This is a <s> token in both, so move one step forward
        word_pointer += 1
        curr_word.append(token)
        continue
    if token == "<s>":
        # No need to append word pointer as we've accounted for it later
        # in this loop
        char_constants.append(1)
        context_map.append('')
        # Assign all mass to <s>
        if "" not in char_probability:
            char_probability[""] = np.zeros(len(vocab))
            char_probability[""][vocab["<s>"]] = 1
        curr_word.append(token)
        continue
    # The -1 is necessary since `word_constants` is offset by 1 as they
    # are predicting the "next" token.
    constant = word_constants[word_pointer - 1]
    char_constants.append(constant)
    # char_probability only stores `context` to `distro` maps
    if "".join(curr_word) not in char_probability:
        probability = get_probability(char_trie, curr_word, vocab)
        char_probability["".join(curr_word)] = probability
    # This will be used in `BatchLoader`
    context_map.append("".join(curr_word))

    # Build next context
    curr_word.append(token)
    if token == "_":
        curr_word = []
        word_pointer += 1
    elif token == "</s>":
        # Skipping </s> and <s>
        word_pointer += 3
        curr_word = []

# To make the length correct
char_constants.append(1)
context_map.append('')

constants_file = os.path.join(args.data_dir, args.constants_file)
np.save(constants_file, np.array(char_constants))
frequencies_file = os.path.join(args.data_dir, args.frequencies_file)
map_file = os.path.join(args.data_dir, args.map_file)

with open(frequencies_file, "wb") as f:
    cPickle.dump(char_probability, f)
with open(map_file, "wb") as f:
    cPickle.dump(context_map, f)

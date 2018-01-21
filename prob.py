"""This script loads an old model and evaluates text."""
from __future__ import print_function

from six.moves import cPickle

from config.arguments import parser
from model.model import Model
from utils.strings import LOGS, FILES
from utils.processor import prob_loader

import codecs
import copy
import numpy as np
import tensorflow as tf

import os

tf.reset_default_graph()
np.random.seed(1)
tf.set_random_seed(1)


def main():
    """The main method of script."""
    args = parser.parse_args()
    evaluate(args)


def evaluate(args):
    """Prepare the data and begins evaluation."""
    # check if all necessary files exist
    ckpt = tf.train.get_checkpoint_state(args.save_dir)

    # open old config and check if models are compatible
    with open(os.path.join(args.save_dir, FILES[2])) as f:
        saved_model_args = cPickle.load(f)

    # open saved vocab/dict and check if vocabs/dicts are compatible
    with open(os.path.join(args.save_dir, FILES[3])) as f:
        saved_vocab = cPickle.load(f)

    # Define the training and eval models in correct scopes
    with tf.name_scope("Train"):
        with tf.variable_scope("Model", reuse=None):
            model = Model(saved_model_args)
    eval_args = copy.deepcopy(saved_model_args)
    eval_args.batch_size = 1
    with tf.name_scope("Eval"):
        with tf.variable_scope("Model", reuse=True):
            eval_model = Model(eval_args, evaluation=True)

    # Preparing evaluation data
    print(LOGS[8])
    eval_x, total_len, tokens = prob_loader(args, saved_vocab)
    print(LOGS[4])

    with tf.Session() as sess:
        tf.initialize_all_variables().run()
        saver = tf.train.Saver(tf.all_variables())
        # restore model
        saver.restore(sess, ckpt.model_checkpoint_path)
        run_eval_epoch(sess=sess,
                       model=eval_model,
                       eval_x=eval_x,
                       total_len=total_len,
                       tokens=tokens,
                       vocab=saved_vocab,
                       args=eval_args)


def run_eval_epoch(sess, model, eval_x, total_len, tokens, vocab, args):
    """Calculate perplexity after every epoch."""
    state = sess.run(model.initial_state)
    output = ""
    for i in range(eval_x.shape[0]):
        print(str(i) + " / " + str(eval_x.shape[0]))
        feed = {
            model.input_data: eval_x[i:i + 1, :],
            model.initial_state: state
        }
        [state, prob] = sess.run([model.last_state,
                                  model.probs], feed)
        for j in range(len(prob)):
            if (i * args.seq_length + j) > total_len:
                continue
            token = tokens[i * args.seq_length + j + 1]
            if token in vocab:
                output += token + " " + \
                    str(prob[j, vocab[token]]) + "\n"
            else:
                output += '<unk>' + " " + \
                    str(prob[j, vocab['<unk>']]) + "\n"
    prob_file = os.path.join(args.save_dir, 'prob_data.txt')
    with codecs.open(prob_file, 'w', 'utf-8') as f:
        f.write(output)

if __name__ == '__main__':
    main()

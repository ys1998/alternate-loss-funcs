"""This script loads an old model and evaluates text."""
from __future__ import print_function

from six.moves import cPickle

from config.arguments import parser
from model.model import Model
from utils.strings import LOGS, FILES
from utils.processor import eval_loader

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
    args.save_dir = os.path.join(args.save_dir,args.job_id)

    ckpt = tf.train.get_checkpoint_state(args.save_dir)

    # open old config and check if models are compatible
    with open(os.path.join(args.save_dir, FILES[2])) as f:
        saved_model_args = cPickle.load(f)

    # open saved vocab/dict and check if vocabs/dicts are compatible
    # with open(os.path.join(args.save_dir, FILES[3])) as f:
    #    saved_vocab = cPickle.load(f)

    # Define the training and eval models in correct scopes
    with tf.name_scope("Train"):
        with tf.variable_scope("Model", reuse=None):
            model = Model(saved_model_args, saved_model_args.config.batch_size, mode='train')
    eval_args = copy.deepcopy(saved_model_args)
    eval_args.batch_size = 1
    with tf.name_scope("Eval"):
        with tf.variable_scope("Model", reuse=True):
            eval_model = Model(eval_args, eval_args.batch_size, mode='eval')

    # Preparing evaluation data
    print(LOGS[8])
    eval_x, eval_y, eval_total_len = eval_loader(args, saved_vocab)
    print(LOGS[4])

    with tf.Session() as sess:
        tf.initialize_all_variables().run()
        saver = tf.train.Saver(tf.all_variables())
        # restore model
        saver.restore(sess, ckpt.model_checkpoint_path)
        run_eval_epoch(sess=sess,
                       model=eval_model,
                       eval_x=eval_x,
                       eval_y=eval_y,
                       eval_total_len=eval_total_len,
                       args=eval_args)


def run_eval_epoch(sess, model, eval_x, eval_y, eval_total_len, args):
    """Calculate perplexity after every epoch."""
    state = sess.run(model.initial_state)
    total_loss = 0.0
    for i in range(eval_x.shape[0]):
        # Need to pass L1 to get evaluation perplexity
        feed = {
            model.input_data: eval_x[i:i + 1, :],
            model.targets: eval_y[i:i + 1, :],
            model.l1: np.ones((args.batch_size, args.seq_length)),
            model.initial_state: state
        }
        [state, loss, cost, prob] = sess.run([model.last_state,
                                              model.loss,
                                              model.cost,
                                              model.probs], feed)
        total_loss += loss.sum()

    # need to subtract off loss from padding tokens
    total_loss -= loss[eval_total_len % args.seq_length - args.seq_length:].sum()
    avg_entropy = total_loss / eval_total_len
    perplexity = np.exp(avg_entropy)
    print(LOGS[9].format(perplexity))
    eval_data = str(perplexity) + "\n"
    with open(os.path.join(args.save_dir, FILES[12]), 'w') as f:
        f.write(eval_data)

if __name__ == '__main__':
    main()

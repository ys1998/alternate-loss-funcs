# Research - Neural Language Modeling
This repository is an extension of the **Language Models** codebase by **Kalpesh Krishna** (@martiansideofthemoon) and was developed as a part of my R&D Project titled **"Alternate Loss Functions for Neural Language Modeling"** under *Prof. Preethi Jyothi* (@pjyothi) during January 2018 - present.

## Requirements
* TensorFlow v1.1
* Python 2.7
* other Python packages as mentioned in `requirements.txt` *(doesn't exist currently)*
## Setting up the SRILM code
* Download the code from [here](http://verispeak.com/projects/srilm/download.html) by filling the form.
* Extract the code and `cd` into the directory.
* Follow the instructions listed in the `INSTALL` file.
* In order to set the `PATH` and `MANPATH` variables, add these lines to `~/.profile` file:
```
export PATH=$PATH:<$SRILM/bin/$MACHINE_TYPE>:<$SRILM/bin>
export MANPATH=$MANPATH:<$SRILM/man>
```
(here `$SRILM` is the path to `srilm` as mentioned in the `Makefile`)
## Steps to run this code on *voxel10*
* Copy the code to the remote machine using `scp`:
```
scp -r <path to local codebase> <username>@voxel10:<folder path relative to remote home directory>
```
(I have assigned `10.130.39.40 voxel10` in my PC's `/etc/hosts` file)
* Set up a proxy in order to access internet via the remote terminal. For this, add these lines to the `~/.profile` file :
```
export HTTP_PROXY = <LDAP ID>:<LDAP password>@netmon.iitb.ac.in:80
export HTTPS_PROXY = $HTTP_PROXY
```
Then execute `source ~/.profile` in order to reflect these changes in the current session.
* Since you don't have permissions to use `sudo`, all the required packages have to be installed locally.
* Set up the `PYTHONPATH` variable by adding this line to `~/.profile` so that Python can recognize and use packages not present in the default install directory:
```
export PYTHONPATH = $PYTHONPATH:<absolute path to home directory>
```
* Install the required packages using `easy_install`:
```
easy_install --install-dir <absolute path to home directory> <PACKAGE NAME>
```
## Running the code
Download the datasets from [here](https://drive.google.com/file/d/0B5Y_SiDYwIObaE52dmZ0YVFXckU/view?usp=sharing). Assuming you have stored the folders `ptb` and `indian` in the same directory as the rest of the code, run the following commands -
* `python utils/gen_frequency.py --data_dir ptb/ --filename ptb.train.txt` : This is done to generate the necessary n-gram files using the SRILM toolkit. `counts.txt`, `ngram-lm` and `vocab` files are generated in the `data_dir`. The `filename` attribute specifies the file for which n-grams are calculated.
* `python main.py --mode train --data_dir ptb/ --save_dir save/ --best_dir save_best --config_file config/sgd.yml --loss_mode l1 [--mixed_constant 0.6]` : Initiates training the model. Different loss modes can be tried by changing the `loss_mode` argument, and the `mixed_constant` can be changed for *mixed* loss (i.e. `--loss_mode mixed`) by changing the corresponding argument. Custom config file can be used using the `config_file` flag.
* `python main.py --mode valid --data_dir ptb/ --save_dir save/ --best_dir save_best --config_file config/sgd.yml` : Loads the best model from `best_dir` and evaluates perplexity on validation set and stores the probability distribution in `probs_valid.txt` in the `save_dir`.
* `python main.py --mode test --data_dir ptb/ --save_dir save/ --best_dir save_best --config_file config/sgd.yml` : Loads the best model from `best_dir` and evaluates perplexity on test set and stores the probability distribution in `probs_test.txt` in the `save_dir`.
* Sample output can also be generated for the trained model using this command : `python main.py --mode generate --data_dir ptb/ --save_dir save/ --best_dir save_best --config_file config/sgd.yml`. It generates sample text file and stores it in `save_dir` as `generate.txt`.
* * *
# Original README

Courtesy : Kalpesh Krishna (@martiansideofthemoon)

## File Description

* `config/arguments.py` - Contains a list of all the arguments used to configure this model while training.
* `model/model.py` - Contains the actual TensorFlow model.
* `utils/adaptive.py` - A list of wrappers to allow an adaptive mode across epochs.
* `utils/fix_raw.py` - A script which fixes raw data and generates standard LM input files.
* `utils/gen_char_frequency.py` - A script which generates character level interpolation constants and probabilities. It needs word level constants as an input. This is not executed in `train.py` and must be run explicitely.
* `utils/gen_frequency.py` - A script which generates word level interpolation constants and probabilities. This is not executed in `train.py` and must be run explicitely.
* `utils/helpers.py` - A useful set of utility functions and classes used by both `utils/gen*_frequency.py` and for training.
* `utils/processor.py` - Contains the `DataLoader` class and `BatchLoader` class.
* `utils/strings.py` - A list of all logs and error outputs.
* `train.py` - The starting point for the neural network training.
* `schedule.py` - A useful script which calls `train.py` on a different `screen`. It also adds logs to a local file.
* `grid_search.py` - A script to automatically carry out grid searching for finding optimal hyperparameters.

## To Run

Download the datasets from [here](https://drive.google.com/file/d/0B5Y_SiDYwIObaE52dmZ0YVFXckU/view?usp=sharing). Assuming you have stored them in the same directory, run the following commands -

* `python utils/gen_frequency.py --data_dir ptb/ --filename ptb.train.txt`. (This will take a lot of time to run. If you just want constants for character level script, add flag `--no-freq`. This runs quickly).
* `python utils/gen_char_frequency.py --data_dir ptb/ --filename ptb.char.train.txt --word_file ptb.train.txt`. (Only for character level models)
* `mkdir save`
* `python train.py --data_dir ptb/ --save_dir save/ --filename ptb.train.txt --eval_text ptb/ptb.valid.txt` (for word level models)
* `python train.py --data_dir ptb/ --save_dir save/ --filename ptb.char.train.txt --eval_text ptb/ptb.char.valid.txt --char` (for character level models, don't forget the `--char` flag)

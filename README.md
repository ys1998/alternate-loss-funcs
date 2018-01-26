# Research - Neural Language Modeling
This repository is an extension of the **Language Models** codebase by Kalpesh Krishna (@martiansideofthemoon) and was developed as a part of my R&D Project titled **"Alternate Loss Functions for Neural Language Modeling"** under Prof. Preethi Jyothi (@pjyothi) during January 2018 - present.

## Requirements
* TensorFlow v1.1
* Python 2.7
* other Python packages as mentioned in `requirements.txt`

* * *
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
* **Currently using this until errors are resolved** : `python train.py --data_dir ptb/ --train_dir save/ --filename ptb.train.txt`
* `python train.py --data_dir ptb/ --save_dir save/ --filename ptb.char.train.txt --eval_text ptb/ptb.char.valid.txt --char` (for character level models, don't forget the `--char` flag)

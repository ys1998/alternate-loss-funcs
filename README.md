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
* This step is done to generate the necessary n-gram files using the SRILM toolkit. `counts.txt`, `ngram-lm` and `vocab` files are generated for the specified text corpus.
```
srilm/bin/i686-m64/ngram-count -unk -order 3 -text ptb/ptb.train.txt -kndiscount1 -kndiscount2 -kndiscount3 -write ptb/counts.txt -lm ptb/ngram-lm -interpolate2 -gt3min 1
```
* Initiate training the model. Different loss modes can be tried by changing the `loss_mode` argument, and the `mixed_constant` can be changed for *mixed* loss (i.e. `--loss_mode mixed`) by changing the corresponding argument. Custom config file can be used using the `config_file` flag.

```
python main.py --mode train --data_dir ptb/ --save_dir save/ --best_dir save_best --config_file config/sgd.yml --lm ngram-lm --loss_mode l1 [--mixed_constant 0.6]
```
* Load the best model from `best_dir` and evaluate perplexity on validation set and stores the probability distribution in `probs_valid.txt` in the `save_dir`.
```
python main.py --mode valid --data_dir ptb/ --save_dir save/ --best_dir save_best --config_file config/sgd.yml --lm ngram-lm
```
* Load the best model from `best_dir` and evaluate perplexity on test set and stores the probability distribution in `probs_test.txt` in the `save_dir`.
```
python main.py --mode test --data_dir ptb/ --save_dir save/ --best_dir save_best --config_file config/sgd.yml --lm ngram-lm
```
* Sample output can also be generated for the trained model using this command. It generates sample text file and stores it in `save_dir` as `generate.txt`.
```
python main.py --mode generate --data_dir ptb/ --save_dir save/ --best_dir save_best --config_file config/sgd.yml --lm ngram-lm
```
* More data can be easily mined using the `spider.py` script, as shown :
```
python spider.py --lang hi --N 500 --D 15 --filename hi_space.txt
```
This will initiate a *spider* which will crawl across `N` pages upto a maximum depth of `D` in the corresponding BFS tree, collecting data of `lang` language and storing it in the `filename` file. Possible languages are Hindi (`hi`), Tamil (`ta`), Telugu (`te`), Kannada (`ka`) and Malayalam (`ma`).

## Experiments
These were the experiments I performed during my R&D project, and the steps by which they can be replicated :
1. **Alter temperature for softmax layer** : Import the correct model in `main.py` by commenting/uncommenting the required import statements. Add the `--T` argument to the *train* command and obtain results for different value of `T` and `loss_mode`.
```
python main.py --mode train --data_dir ptb/ --save_dir save/ --best_dir save_best --config_file config/sgd.yml --lm ngram-lm --loss_mode l1 --T 5
```

# Linguistic Probing
This directory contains code for the experiments in section `5.3` of the paper. We only provide the code for the synthetic dataset here. For running the experiments on CoNLL-2012, we refer the reader to the original author's [repository](https://github.com/clarkkev/attention-analysis).

This is because the CoNLL-2012 dataset needs permissions to share the code. Please contact asd@cs.princeton.edu for help with setup.

### Installation
1. Create a new environment. Don't use the same environment for this experiment - `conda create -n probing python=3.6`
1. `conda activate probing`
1. Install `transformers` - `pip install transformers==2.4.1`

### Commands
1. Run the experiment on RoBERTa-Base - `python run_coreference.py --train_data_file reflexive.txt --model_type roberta --seed 42 --model_name_or_path roberta-base`

### References
Dataset borrowed from [https://github.com/yongjie-lin/bert-opensesame](https://github.com/yongjie-lin/bert-opensesame)
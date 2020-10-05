# Attention Guidance for Language Modeling

This directory contains code adapted from an older version of [HuggingFace](https://github.com/huggingface/transformers).

## Installation
The best way to install all the necessary packages is to use `conda` or `miniconda`.

1. Follow the instructions [here](https://docs.conda.io/projects/conda/en/latest/user-guide/install/index.html) to install `conda/miniconda`. `miniconda` is recommended because it takes considerably lesser disk space.
1. Run `conda activate base`
1. Install the required packages from the environment file - `conda env create -f requirements/environment.yml`
1. Activate the environment - `conda activate ag_lm`
1. Install the `transformers` package in editable mode so that you can make changes to the code - `pip install -e .`
1. Install packages required to run scripts from the `example` folder - `pip install -r requirements/requirements.txt`

## Running Language Modeling on English
To run Language Modeling on English, you can choose one of the model configurations present in `ag_config`.

Download your training data and move it to the `data` folder. You can run `./scripts/download_wikitext.sh` if you want to run your model on [WikiText-103](https://blog.einstein.ai/the-wikitext-long-term-dependency-language-modeling-dataset/).

Use the following command to run LM using a `12` layer RoBERTa model

```python
python ag_models/run_language_modeling_ag.py --output_dir=model_outputs/roberta_12_wiki --model_type=roberta --config_name=ag_config/roberta_12 --tokenizer_name=ag_config/roberta_12 --do_train --train_data_file=data/wikitext-103/wiki.train.tokens --mlm --ag --num_train_epochs 3 --attn_head_types 0,1,1,4 --scale 100 --per_device_train_batch_size=4 --disable_wandb
```

**Important arguments:**
1. `--model_type` - We currently support only RoBERTa.
1. `--config_name` - Configuration file for your model. You can change the number of layers and other hyperparameters here.
1. `--tokenizer_name` - Tokenizer to use for tokenizing the data. Defaults like `roberta-base` can be used too.
1. `--train_data_file` - File to use for training.
1. `--mlm, --ag` flags - Omitting the `--ag` flag will run a vanilla MLM model without attention-guidance.
1. `--attn_head_types 0,1,1,4` - What are the types of attention heads to use. The numbers correspond to 1) pay attention to the same token, 2) [PREV], 3) [NEXT], and 4) [FIRST]. Refer to the paper for more details.
1. `--scale 100` - This is `alpha` in equation `4` in the paper, which is the relative weight of the `AG` loss as compared to the `MLM` loss. The default value of 100 should work fine in most cases. Smaller values work for models with more layers.
1. `--per_device_train_batch_size=4` - Use the largest batch size that can fit on your GPUs. Code works with multi-GPU settings.

Other parameters can be seen in `src/transformers/training_args.py`

## Running Language Modeling on other Languages
You can use the same commands used above for running LM on other languages. But before that, you will have to generate tokenizer files for the new language.

```python
python scripts/create_tokenizer.py --file data/wikitext-103/wiki.train.tokens --store_files ag_config/other_language --vocab_size 52000
```

## Evaluating English Models on GLUE
Follow instructions given in [examples/text-classification/README.md][examples/text-classification/README.md].

`Important:` Make sure you move your `pytorch` model file from `checkpoint-XXX/` folder to some other folder, because the code automatically assumes `XXX` in `checkpoint-XXX/` corresponds to the number of steps the model has been run for on GLUE.

## Making modifications to the Code Base
The following are the two directories/files which are important if you want to make modifications to the code base
1. All helper scripts and the main script for the AG models are in `ag_models` folder
1. The file which handles the training and evaluation is `src/transformers/trainer_ag.py`
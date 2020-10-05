# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors and The HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Initializes/Modifies the attention heads to conform to predecided types
"""

"""
Example command
python run_coreference.py --train_data_file small_reflexive.txt --model_type roberta --seed 42 --cache_dir ../../../global_data/transformer_models/ --model_name_or_path roberta-base
"""


import argparse
import glob
import logging
import os
import pickle
import random
import re
import shutil
from typing import Dict, List, Tuple

import numpy as np
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset, RandomSampler, SequentialSampler
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm, trange

# Added import statements
# import pdb; pdb.set_trace();
from utils_pretrain import modify_config, test_attention_loss, get_attn_tokens, substitute_embeddings, linear_schedule_for_scale

from transformers import (
    WEIGHTS_NAME,
    AdamW,
    BertConfig,
    BertForMaskedLM,
    BertTokenizer,
    CamembertConfig,
    CamembertForMaskedLM,
    CamembertTokenizer,
    DistilBertConfig,
    DistilBertForMaskedLM,
    DistilBertTokenizer,
    GPT2Config,
    GPT2LMHeadModel,
    GPT2Tokenizer,
    OpenAIGPTConfig,
    OpenAIGPTLMHeadModel,
    OpenAIGPTTokenizer,
    PreTrainedModel,
    PreTrainedTokenizer,
    RobertaConfig,
    RobertaForMaskedLM,
    RobertaTokenizer,
    get_linear_schedule_with_warmup,
)

# # New Models
# from models import RobertaForMaskedLMOnlyAttention

try:
    from torch.utils.tensorboard import SummaryWriter
except ImportError:
    from tensorboardX import SummaryWriter


logger = logging.getLogger(__name__)

# Weights and Biases
import wandb
from wandb_utils import wandb_init_setup

MODEL_CLASSES = {
    "gpt2": (GPT2Config, GPT2LMHeadModel, GPT2Tokenizer),
    "openai-gpt": (OpenAIGPTConfig, OpenAIGPTLMHeadModel, OpenAIGPTTokenizer),
    "bert": (BertConfig, BertForMaskedLM, BertTokenizer),
    "roberta": (RobertaConfig, RobertaForMaskedLM, RobertaTokenizer),
    # "roberta-only-attention": (RobertaConfig, RobertaForMaskedLMOnlyAttention, RobertaTokenizer),
    "distilbert": (DistilBertConfig, DistilBertForMaskedLM, DistilBertTokenizer),
    "camembert": (CamembertConfig, CamembertForMaskedLM, CamembertTokenizer),
}

# distractors = ["and it is like they say .", "and as they say ,", "everyone believes that", "and"]
# distractors = ["and it is like they say .", "and as they say ,", "everyone believes that", "and", "and it is like she says .", "and it is like he says ."]
distractors = ["The people believed .", "and as obama says ,", "the girl believes that .", "and that is what the boy said .", "and it is like they say .", "and as they say ,", "everyone believes that", "and"]

def load_reflexive_dataset(args, tokenizer, distract=True):
    if distract:
        lines = open(args.train_data_file, 'r').readlines()
        dataset, labels = [], []
        no = 0
        for line in lines:
            comp = line.strip().split()
            sentence = comp[:-3]
            encoded = tokenizer.encode(' '.join(sentence))
            if len(encoded) == len(sentence) + 2:
                # rand_sent = distractors[no % len(distractors)]
                rand_sent = random.sample(distractors, k=1)[0]
                rand_sent_encode = tokenizer.encode(rand_sent)
                # dataset.append([rand_sent_encode + encoded])
                # labels.append((int(comp[-3])+1+len(rand_sent_encode), (int(comp[-2])+1+len(rand_sent_encode), int(comp[-1])+1+len(rand_sent_encode))))
                full_sent_encode = tokenizer.encode(rand_sent+ ' ' + ' '.join(sentence))
                dataset.append([full_sent_encode])                
                labels.append((int(comp[-3])+1+len(rand_sent_encode)-2, (int(comp[-2])+1+len(rand_sent_encode)-2, int(comp[-1])+1+len(rand_sent_encode)-2)))
            no += 1
        return (dataset, labels)
    else:
        lines = open(args.train_data_file, 'r').readlines()
        dataset, labels = [], []
        no = 0
        for line in lines:
            comp = line.strip().split()
            sentence = comp[:-3]
            encoded = tokenizer.encode(' '.join(sentence))
            if len(encoded) == len(sentence) + 2:
                dataset.append([encoded])
                labels.append((int(comp[-3])+1, (int(comp[-2])+1, int(comp[-1])+1)))
            no += 1
        return (dataset, labels)

def train(args, train_dataset, model: PreTrainedModel, tokenizer: PreTrainedTokenizer):
    layers = model.config.num_hidden_layers
    heads = model.config.num_attention_heads
    scores = np.zeros((layers, heads), dtype=np.float)
    dataset, labels = train_dataset
    for point, label in tqdm(zip(dataset, labels)):
        point = torch.LongTensor(point)
        point = point.to(args.device)
        attention_map = model.forward(point)[-1]
        for i in range(layers):
            for j in range(heads):
                if torch.argmax(attention_map[i][0][j][label[0]]).item() in label[1]:
                    scores[i, j] += 1

    # print("Aggregated Scores are: {}".format(scores))
    # print("Average Accuracy are: {}".format(scores/(len(dataset))))
    print("Maximum accuracy is: {}".format(np.max(scores/(len(dataset)))))
    print("Mean accuracy is: {}".format(np.mean(scores/(len(dataset)))))
    print("Total number of samples in the dataset is {}".format(len(dataset)))

def main():
    parser = argparse.ArgumentParser()

    # Required parameters
    parser.add_argument(
        "--train_data_file", default=None, type=str, required=True, help="The input training data file (a text file)."
    )
    parser.add_argument(
        "--model_type", type=str, required=True, help="The model architecture to be trained or fine-tuned.",
    )
    parser.add_argument(
        "--model_name_or_path",
        default=None,
        type=str,
        help="The model checkpoint for weights initialization. Leave None if you want to train a model from scratch.",
    )

    parser.add_argument(
        "--config_name",
        default=None,
        type=str,
        help="Optional pretrained config name or path if not the same as model_name_or_path. If both are None, initialize a new config.",
    )
    parser.add_argument(
        "--tokenizer_name",
        default=None,
        type=str,
        help="Optional pretrained tokenizer name or path if not the same as model_name_or_path. If both are None, initialize a new tokenizer.",
    )
    parser.add_argument(
        "--cache_dir",
        default=None,
        type=str,
        help="Optional directory to store the pre-trained models downloaded from s3 (instead of the default one)",
    )
    parser.add_argument("--no_cuda", action="store_true", help="Avoid using CUDA when available")
    parser.add_argument("--seed", type=int, default=42, help="random seed for initialization")

    
    args = parser.parse_args()

    # Seed
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    # Setup CUDA, GPU & distributed training
    device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
    args.n_gpu = torch.cuda.device_count()
    args.device = device

    config_class, model_class, tokenizer_class = MODEL_CLASSES[args.model_type]

    if args.config_name:
        config = config_class.from_pretrained(args.config_name, cache_dir=args.cache_dir)
    elif args.model_name_or_path:
        config = config_class.from_pretrained(args.model_name_or_path, cache_dir=args.cache_dir)
    else:
        config = config_class()

    if args.tokenizer_name:
        tokenizer = tokenizer_class.from_pretrained(args.tokenizer_name, cache_dir=args.cache_dir)
    elif args.model_name_or_path:
        tokenizer = tokenizer_class.from_pretrained(args.model_name_or_path, cache_dir=args.cache_dir)
    else:
        raise ValueError(
            "You are instantiating a new {} tokenizer. This is not supported, but you can do it from another script, save it,"
            "and load it from here, using --tokenizer_name".format(tokenizer_class.__name__)
        )

    # Modify the configuration to return attention and hidden states
    if args.model_name_or_path:
        model = model_class.from_pretrained(
            args.model_name_or_path,
            from_tf=bool(".ckpt" in args.model_name_or_path),
            config=config,
            cache_dir=args.cache_dir,
        )
        model = modify_config(model, model_type=args.model_type, pretrained='pretrained')
    else:
        logger.info("Training new model from scratch")
        # Modify the config to return attention and hidden states
        config = modify_config(config, model_type=args.model_type)
        model = model_class(config=config)

    model.to(args.device)

    # Training

    # train_dataset = load_and_cache_examples(args, tokenizer, evaluate=False)
    train_dataset = load_reflexive_dataset(args, tokenizer)

    train(args, train_dataset, model, tokenizer)
    # logger.info(" global_step = %s, average loss = %s", global_step, tr_loss)


if __name__ == "__main__":
    main()
'''
Utils for experiments on pretraining along with attention initialization
'''

import numpy as np
import torch
from transformers import BertModel, BertConfig
# import pdb; pdb.set_trace()

from transformers import (
    WEIGHTS_NAME,
    AdamW,
    AlbertConfig,
    AlbertForSequenceClassification,
    AlbertTokenizer,
    BertConfig,
    BertForSequenceClassification,
    BertTokenizer,
    DistilBertConfig,
    DistilBertForSequenceClassification,
    DistilBertTokenizer,
    FlaubertConfig,
    FlaubertForSequenceClassification,
    FlaubertTokenizer,
    RobertaConfig,
    RobertaForSequenceClassification,
    RobertaTokenizer,
    XLMConfig,
    XLMForSequenceClassification,
    XLMRobertaConfig,
    XLMRobertaForSequenceClassification,
    XLMRobertaTokenizer,
    XLMTokenizer,
    XLNetConfig,
    XLNetForSequenceClassification,
    XLNetTokenizer,
    get_linear_schedule_with_warmup,
)

MODEL_CLASSES = {
    "bert": (BertConfig, BertForSequenceClassification, BertTokenizer),
    "xlnet": (XLNetConfig, XLNetForSequenceClassification, XLNetTokenizer),
    "xlm": (XLMConfig, XLMForSequenceClassification, XLMTokenizer),
    "roberta": (RobertaConfig, RobertaForSequenceClassification, RobertaTokenizer),
    "distilbert": (DistilBertConfig, DistilBertForSequenceClassification, DistilBertTokenizer),
    "albert": (AlbertConfig, AlbertForSequenceClassification, AlbertTokenizer),
    "xlmroberta": (XLMRobertaConfig, XLMRobertaForSequenceClassification, XLMRobertaTokenizer),
    "flaubert": (FlaubertConfig, FlaubertForSequenceClassification, FlaubertTokenizer),
}

def modify_config(config_or_model, model_type='roberta', pretrained=None):
    '''
    Modifies model's config to ensure attention and hidden states are returned
    '''
    assert pretrained == 'pretrained' or pretrained is None, "Illegal option in modify_config"
    assert model_type in ['bert', 'roberta', 'roberta-only-attention'], "Currently supporting only bert and roberta"
    if model_type == 'bert':
        if pretrained == 'pretrained':
            # Overloaded the variable config here
            config_or_model.bert.encoder.output_attentions = True
            config_or_model.bert.encoder.output_hidden_states = True
            # Can't change the variable only in encoder, have to change it in all attention heads
            for layer in config_or_model.bert.encoder.layer:
                layer.attention.self.output_attentions = True
        else:
            config_or_model.output_attentions = True
            config_or_model.output_hidden_states = True
    elif model_type in ['roberta', 'roberta-only-attention']:
        if pretrained == 'pretrained':
            # Overloaded the variable config here
            config_or_model.roberta.encoder.output_attentions = True
            config_or_model.roberta.encoder.output_hidden_states = True
            # Can't change the variable only in encoder, have to change it in all attention heads
            for layer in config_or_model.roberta.encoder.layer:
                layer.attention.self.output_attentions = True
        else:
            config_or_model.output_attentions = True
            config_or_model.output_hidden_states = True        
        
    return config_or_model

def get_attn_tokens(tokenizer, model_type='roberta'):
    '''
    Returns the tokens to which attention needs to be paid
    '''
    assert model_type in ['bert', 'roberta', 'roberta-only-attention'], "Currently supporting only bert and roberta"

    if model_type == 'bert':
        tokens = ['[SEP]', '.']
        # tokens = ['<s>', '</s>']
        ids = []
        for token in tokens:
            ids.append(tokenizer.vocab[token])
    elif model_type in ['roberta', 'roberta-only-attention']:
        # tokens = ['[SEP]', '.', '[CLS]']
        tokens = ['<s>', '</s>']
        ids = []
        for token in tokens:
            ids.append(tokenizer.convert_tokens_to_ids(token))
    
    return ids

def create_pos_attn_patterns(attentions):
    '''
    Creates attention patterns related to positional encoding for attention initialization
    one_to_one - pays attention to the corresponding token
    next_token - pays attention to the next token
    prev_token - pays attention to the previous token
    cls_token  - pays attention to the first index ([CLS])
    '''

    one_to_one = target = torch.eye(attentions[0].shape[-1])
    next_token = torch.cat((torch.cat((torch.zeros(attentions[0].shape[-1]-1, 1), torch.eye(attentions[0].shape[-1]-1)), dim=1),\
         torch.zeros(1, attentions[0].shape[-1])), dim=0)
    prev_token = torch.cat((torch.zeros(1, attentions[0].shape[-1]), \
        torch.cat((torch.eye(attentions[0].shape[-1]-1), torch.zeros(attentions[0].shape[-1]-1, 1)), dim=1)), dim=0)
    cls_token = torch.zeros(attentions[0].shape[-1], attentions[0].shape[-1])
    cls_token[:,0] = 1.

    return [one_to_one, next_token, prev_token, cls_token]

def create_token_attn_patterns(inputs, attentions, tokens_to_attend):
    '''
    Creates attention patterns to pay attention to specific tokens like period, [SEP] and [CLS]

    params:
    tokens_to_attend - The tokens to which attention needs to be paid. [SEP] and '.' are the most common
    '''
    targets = []
    for token in tokens_to_attend:
        temp = inputs.clone()
        ones = (temp == token)
        zeros = (temp != token)
        temp[ones] = 1.
        temp[zeros] = 0.
        targets.append(temp)

    return targets

def test_attention_loss(inputs, outputs, attn_tokens, device, attn_head_split=[0,1,1,0,3,3], start_sep_from=0):
    '''
    Adds a random loss based on attention values
    To test gradients
    outputs[-1] contains the attention values (tuple of size num_layers)
    and each elements is of the shape
    [batch_size X num_heads X max_sequence_len X max_sequence_len]

    params:
    attn_tokens - The tokens to which attention needs to be paid
    '''
    # Get the attention values
    attentions = outputs[-1]

    # The number attention heads of each type. one-to-one, next, previous, [CLS]
    # numbers = [0,1,1,0]
    numbers = attn_head_split[:4]
    cum_sum = np.cumsum(numbers)
    # [SEP] and . in BERT
    # <s> and </s> in RoBERTa
    # token_numbers = [3,3]
    token_numbers = attn_head_split[4:]
    token_cum_sum = np.cumsum(token_numbers)+cum_sum[-1]

    # Matrices containing the attention patterns
    # First position based attention and then token based attention
    targets = create_pos_attn_patterns(attentions)
    token_targets = create_token_attn_patterns(inputs, attentions, attn_tokens)

    # Loss for positional attention (Next token, Previous token)
    expanded_targets = []

    # Change the tensor's dimension
    for (num, target) in zip(numbers, targets):
        if num == 0:
            expanded_targets.append(None)
        else:
            # Add dimensions so that the tensor can be repeated
            target = torch.unsqueeze(target, 0)
            target = torch.unsqueeze(target, 0)

            # Change the target tensor's dimension so that it matches batch_size X num_heads[chosen]
            target = target.repeat(attentions[0].shape[0], num, 1, 1)
            target = target.to(device)
            expanded_targets.append(target)

    loss = torch.nn.MSELoss()
    total_loss = 0.

    # Go over all the layers
    for i in range(len(attentions)):
        for j in range(len(numbers)):
            if expanded_targets[j] is not None:
                if j == 0:
                    total_loss += loss(expanded_targets[j], attentions[i][:,0:cum_sum[j]])
                else:
                    total_loss += loss(expanded_targets[j], attentions[i][:,cum_sum[j-1]:cum_sum[j]])

    
    # Loss for token attention ([SEP], [CLS], etc)
    expanded_targets = []

    # Change the tensor's dimension
    for (num, target) in zip(token_numbers, token_targets):
        if num == 0:
            expanded_targets.append(None)
        else:
            # Add dimensions so that the tensor can be repeated
            target = torch.unsqueeze(target, 1)
            target = torch.unsqueeze(target, 2)


            # Change the target tensor's dimension so that it matches batch_size X num_heads[chosen]
            target = target.repeat(1, num, attentions[0].shape[-1], 1)
            target = target.to(device)
            expanded_targets.append(target)


    # Go over all the layers
    for i in range(start_sep_from, len(attentions)):
        for j in range(len(token_numbers)):
            if expanded_targets[j] is not None:
                if j == 0:
                    total_loss += loss(expanded_targets[j], attentions[i][:,cum_sum[-1]:token_cum_sum[j]])
                else:
                    total_loss += loss(expanded_targets[j], attentions[i][:,token_cum_sum[j-1]:token_cum_sum[j]])    
    return total_loss


def substitute_embeddings(model, args):
    '''
    Substitute the embeddings 
    '''
    config_class, model_class, tokenizer_class = MODEL_CLASSES["bert"]
    config = config_class.from_pretrained("bert-base-uncased", cache_dir=args.cache_dir)
    aux_model = model_class.from_pretrained(
        "bert-base-uncased",
        from_tf=False,
        config=config,
        cache_dir=args.cache_dir if args.cache_dir else None,
    )
    model.bert.embeddings = aux_model.bert.embeddings
    return model

def linear_schedule_for_scale(num_training_steps, current_step, num_stagnant_steps=20000):
    if current_step < num_stagnant_steps:
        return 1.0
    else:
        return max(0.0, float(num_training_steps - current_step) / float(max(1, num_training_steps - num_stagnant_steps)))
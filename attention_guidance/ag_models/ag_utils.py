'''
Utils for attention-guided models
'''

from transformers import PretrainedConfig, PreTrainedModel
# import pdb; pdb.set_trace()

# def modify_config(config_or_model, model_type='roberta', pretrained=None):
#     '''
#     Modifies model's config to ensure attention probabilities of each head is returned

#     Parameters
#     ----------
#     config_or_model : obj
#         An overloaded variable which contains either the config variable, or the model variable
#     model_type : str
#         The type of model being used (bert/roberta)
#     pretrained : str, optional
#         If the model is pretrained or not. Options are 'pretrained'/None. If the former, 
#         config_or_model should be an instance of the model
    
#     Returns
#     -------
#     config_or_model : obj
#         Transformed config or model which returns attention probabilities of each head
#     '''

#     # Check if all the options are legal
#     assert pretrained == 'pretrained' or pretrained is None, "Illegal option in modify_config"
#     assert model_type in ['bert', 'roberta'], "Currently supporting only bert and roberta"

#     # If BERT Model
#     if model_type in ['bert']:
#         # If the model is a pretrained model
#         if pretrained == 'pretrained':
#             config_or_model.bert.encoder.output_attentions = True

#             # Can't change the variable only in encoder, have to change it in all attention heads
#             for layer in config_or_model.bert.encoder.layer:
#                 layer.attention.self.output_attentions = True
#         else:
#             config_or_model.output_attentions = True
#     # If RoBERTa model
#     elif model_type in ['roberta']:
#         # If the model is a pretrained model
#         if pretrained == 'pretrained':
#             config_or_model.roberta.encoder.output_attentions = True

#             # Can't change the variable only in encoder, have to change it in all attention heads
#             for layer in config_or_model.roberta.encoder.layer:
#                 layer.attention.self.output_attentions = True
#         else:
#             config_or_model.output_attentions = True         
        
#     return config_or_model

def modify_config(config_or_model, model_type='roberta'):
    '''
    Modifies model's config to ensure attention probabilities of each head is returned

    Parameters
    ----------
    config_or_model : obj
        An overloaded variable which contains either the config variable, or the model variable
    model_type : str
        The type of model being used (bert/roberta)
    
    Returns
    -------
    config_or_model : obj
        Transformed config or model which returns attention probabilities of each head
    '''

    # Check if all the options are legal
    assert model_type in ['bert', 'roberta'], "Currently supporting only bert and roberta"

    # If BERT Model
    if model_type in ['bert']:
        # If the model is a pretrained model
        if isinstance(config_or_model, PreTrainedModel):
            # Change output_attention in the encoder and all the attention heads
            config_or_model.bert.encoder.output_attentions = True

            for layer in config_or_model.bert.encoder.layer:
                layer.attention.self.output_attentions = True
        else:
            config_or_model.output_attentions = True
    # If RoBERTa Model
    elif model_type in ['roberta']:
        # If the model is a pretrained model
        if isinstance(config_or_model, PreTrainedModel):
            # Change output_attention in the encoder and all the attention heads
            config_or_model.roberta.encoder.output_attentions = True

            for layer in config_or_model.roberta.encoder.layer:
                layer.attention.self.output_attentions = True
        else:
            config_or_model.output_attentions = True         
        
    return config_or_model    
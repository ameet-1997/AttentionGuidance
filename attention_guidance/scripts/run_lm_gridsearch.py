import os

# Define hyperparameter grid
lrs = [1e-4, 5e-5, 1e-5]
warmups = [0, 100, 1000]
batches = [2,3]
# lrs = [1e-4]
# warmups = [0]
# batches = [1]

# MLM model or AG model
mlm = False

for learning_rate in lrs:
    for warmup in warmups:
        for batch in batches:
            if mlm:
                os.system('python ../ag_models/run_language_modeling_ag.py --output_dir=../../model_outputs/mlm_grid_test --model_type=roberta --config_name=../ag_config/roberta_8 --tokenizer_name=../ag_config/roberta_8  --do_train  --train_data_file=../../../../global_data/wikitext-103-raw/wiki.train.raw  --mlm --save_steps 100000 --overwrite_output_dir --num_train_epochs 1 --per_device_train_batch_size={} --learning_rate={} --warmup_steps={} --wandb_project {} --wandb_name {}_{}_{}_{}'.format(batch, learning_rate, warmup, 'gridsearch', 'mlm8', learning_rate, warmup, batch))
            else:
                os.system('python ../ag_models/run_language_modeling_ag.py --output_dir=../../model_outputs/ag_grid_test --model_type=roberta --config_name=../ag_config/roberta_8 --tokenizer_name=../ag_config/roberta_8  --do_train  --train_data_file=../../../../global_data/wikitext-103-raw/wiki.train.raw  --mlm --ag --attn_head_types 0,2,2,2 --scale 100 --save_steps 100000 --overwrite_output_dir --num_train_epochs 1 --per_device_train_batch_size={} --learning_rate={} --warmup_steps={} --wandb_project {} --wandb_name {}_{}_{}_{}'.format(batch, learning_rate, warmup, 'gridsearch', 'ag8', learning_rate, warmup, batch))                

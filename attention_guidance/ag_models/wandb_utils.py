import wandb
import os

def wandb_init_setup(args):
    '''
    Uses API key and sets initial config and hyperparameters
    '''
    # Ameet's wandb key
    os.environ["WANDB_API_KEY"] = "a8d4de02e5bbee944cdfa143d1dba8f1a7b63fb4"
    os.environ["WANDB_WATCH"] = "false"
    os.environ["WANDB_PROJECT"] = args.wandb_project
    os.environ["WANDB_NAME"] = args.wandb_name

    if args.disable_wandb:
       os.environ["WANDB_DISABLED"] = 'true'

    # # Initialize with hyperparameters and project name
    # wandb.init(config=args, name=args.wandb_name, project=args.wandb_project)
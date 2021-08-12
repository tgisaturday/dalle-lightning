import argparse, os, sys, datetime, glob, importlib, yaml
import numpy as np
import random
from PIL import Image
import torch


# vision imports
from torchvision import transforms as T
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder

from pl_dalle.models.vqgan import VQGAN, EMAVQGAN, GumbelVQGAN
from pl_dalle.models.vqvae import VQVAE, EMAVQVAE, GumbelVQVAE
from pl_dalle.models.vqvae2 import VQVAE2
from pl_dalle.loader import ImageDataModule
from pl_dalle.callbacks import ReconstructedImageLogger

import pytorch_lightning as pl
from pytorch_lightning import seed_everything
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import XLAStatsMonitor
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint


if __name__ == "__main__":

    now = datetime.datetime.now().strftime("%Y-%m-%dT%H-%M-%S")

    config_parser = parser = argparse.ArgumentParser(description='Training Config', add_help=False)
    parser.add_argument('-c', '--config', default='', type=str, metavar='FILE',
                    help='YAML config file specifying default arguments')


    parser = argparse.ArgumentParser(description='VQVAE Training for Pytorch TPU')

    parser.add_argument('--config', default='', type=str, metavar='FILE',
                    help='YAML config file specifying default arguments')

    #path configuration
    parser.add_argument('--train_dir', type=str, default='dataset/train/',
                    help='path to train dataset')
    parser.add_argument('--val_dir', type=str, default='dataset/val/',
                    help='path to val dataset')                    
    parser.add_argument('--log_dir', type=str, default='results/',
                    help='path to save logs')
    parser.add_argument('--backup_dir', type=str, default='backups/',
                    help='path to save backups for sudden crash')                    
    parser.add_argument('--ckpt_path', type=str,default='results/checkpoints/last.ckpt',
                    help='path to previous checkpoint') 
 

    #training configuration
    parser.add_argument('--resume', action='store_true', default=False,
                    help='whether to resume from checkpoint')      
    parser.add_argument('--backup', action='store_true', default=False,
                    help='save backup and load from backup if restart happens')      
    parser.add_argument('--backup_steps', type =int, default = 1000,
                    help='saves backup every n training steps') 
    parser.add_argument('--log_images', action='store_true', default=False,
                    help='log image outputs. not recommended for tpus')   
    parser.add_argument('--image_log_steps', type=int, default=1000,
                    help='log image outputs for every n step. not recommended for tpus')   
    parser.add_argument('--refresh_rate', type=int, default=1,
                    help='progress bar refresh rate')    
    parser.add_argument('--precision', type=int, default=32,
                    help='precision for training')         

    parser.add_argument('--fake_data', action='store_true', default=False,
                    help='using fake_data for debugging') 
    parser.add_argument('--use_tpus', action='store_true', default=False,
                    help='using tpu')
                                                                                        
             
    parser.add_argument('--seed', type=int, default=42,
                    help='random seed')  
    parser.add_argument('--gpus', type=int, default=16,
                    help='number of gpus')     
    parser.add_argument('--gpu_dist', action='store_true', default=False,
                    help='distributed training with gpus') 

    parser.add_argument('--tpus', type=int, default=8,
                    help='number of tpus')                                 
    parser.add_argument('--num_sanity_val_steps', type=int, default=0,
                    help='num_sanity_val_steps')  

    parser.add_argument('--learning_rate', default=4.5e-6, type=float,
                    help='base learning rate')
    parser.add_argument('--lr_decay', action='store_true', default=False,
                    help = 'use learning rate decay')

    parser.add_argument('--starting_temp', type = float, default = 1., 
                    help = 'starting temperature')
    parser.add_argument('--temp_min', type = float, default = 0.5, 
                    help = 'minimum temperature to anneal to')
    parser.add_argument('--anneal_rate', type = float, default = 1e-6, 
                    help = 'temperature annealing rate')    

    parser.add_argument('--batch_size', type=int, default=8,
                    help='training settings')  
    parser.add_argument('--epochs', type=int, default=100,
                    help='training settings')                                    
    parser.add_argument('--num_workers', type=int, default=16,
                    help='training settings')   
    parser.add_argument('--img_size', type=int, default=256,
                    help='training settings')
    parser.add_argument('--resize_ratio', type=float, default=0.75,
                    help='Random resized crop lower ratio')

    parser.add_argument('--test', action='store_true', default=False,
                    help='test run')                     
    parser.add_argument('--debug', action='store_true', default=False,
                    help='debug run') 
    parser.add_argument('--xla_stat', action='store_true', default=False,
                    help='print out tpu related stat')  
    parser.add_argument('--web_dataset',action='store_true', default=False,
                    help='enable web_dataset')  
    parser.add_argument('--dataset_size', type=int, default=1e9,
                    help='training settings')
    #model configuration
    parser.add_argument('--model', type=str, default='vqvae')
    parser.add_argument('--codebook_dim', type=int, default=256,
                    help='number of embedding dimension for codebook')       
    parser.add_argument('--num_tokens', type=int, default=1024,
                    help='codebook size')        
    parser.add_argument('--double_z', type=bool, default=False,
                    help='double z for encoder')
    parser.add_argument('--z_channels', type=int, default=256,
                    help='image latent feature dimension')
    parser.add_argument('--resolution', type=int, default=256,
                    help='image resolution')
    parser.add_argument('--in_channels', type=int, default=3,
                    help='input image channel')
    parser.add_argument('--out_channels', type=int, default=3,
                    help='output image channel')    
    parser.add_argument('--hidden_dim', type=int, default=128,
                    help='hidden dimension init size')  
    parser.add_argument('--ch_mult', nargs='+', type=int, default=[1,1,2,2,4],
                    help='resnet channel multiplier')  
    parser.add_argument('--num_res_blocks', type=int, default=2,
                    help='number of resnet blocks')                     
    parser.add_argument('--attn_resolutions', nargs='+', type=int, default=[16],
                    help='model settings')  
    parser.add_argument('--dropout', type=float, default=0.0,
                    help='model settings') 
    parser.add_argument('--quant_beta', type=float, default=0.25,
                    help='quantizer beta')                     
    parser.add_argument('--quant_ema_decay', type=float, default=0.99,
                    help='quantizer ema decay')
    parser.add_argument('--quant_ema_eps', type=float, default=1e-5,
                    help='quantizer ema epsilon')  
                              
    #vqvae2 specialized options
    parser.add_argument('--num_res_ch', type=int, default=32,
                    help='model settings')                                        

    #loss configuration
    parser.add_argument('--smooth_l1_loss', dest = 'smooth_l1_loss', action = 'store_true')
    parser.add_argument('--kl_loss_weight', type = float, default=1e-8,
                    help = 'KL loss weight')
    parser.add_argument('--disc_conditional', type=bool, default=False,
                    help='lossconfig')      
    parser.add_argument('--disc_in_channels', type=int, default=3,
                    help='lossconfig') 
    parser.add_argument('--disc_start', type=int, default=10001,
                    help='lossconfig') 
    parser.add_argument('--disc_weight', type=float, default=0.8,
                    help='lossconfig') 
    parser.add_argument('--codebook_weight', type=float, default=1.0,
                    help='lossconfig') 

    parser.add_argument('--wandb', action='store_true', default=False, help='use wandb for logging')
    #misc configuration
    
    args_config, remaining = config_parser.parse_known_args()
    if args_config.config:
        with open(args_config.config, 'r') as f:
            cfg = yaml.safe_load(f)
            parser.set_defaults(**cfg)

    # The main arg parser parses the rest of the args, the usual
    # defaults will have been overridden if config file specified.
    args = parser.parse_args(remaining)

    #random seed fix
    seed_everything(args.seed)   

    if args.use_tpus:
        tpus = args.tpus
        gpus = None
        args.num_cores = args.tpus        
        import torch_xla.core.xla_model as xm
        args.world_size = xm.xrt_world_size()
    else:
        tpus = None
        gpus = args.gpus
        args.num_cores = args.gpus        
        if args.gpu_dist:
            torch.distributed.init_process_group(backend='nccl') 
            args.world_size = torch.distributed.get_world_size()
        else:
            args.world_size = 1

    args.learning_rate = args.learning_rate * args.world_size * args.batch_size * args.num_cores
    
    datamodule = ImageDataModule(args.train_dir, args.val_dir, 
                                args.batch_size, args.num_workers, 
                                args.img_size, args.resize_ratio, 
                                args.fake_data, args.web_dataset,
                                world_size = args.world_size,
                                dataset_size = args.dataset_size)
  
    # model
    #let logger callback know if the model uses multiple optimizers
    args.multi_optim = False
    if args.model == 'vqgan':
        model = VQGAN(args, args.batch_size, args.learning_rate)
        args.multi_optim = True
    elif args.model == 'evqgan':
        model = EMAVQGAN(args, args.batch_size, args.learning_rate)  
        args.multi_optim = True                
    elif args.model == 'gvqgan':
        model = GumbelVQGAN(args, args.batch_size, args.learning_rate) 
        args.multi_optim = True               
    elif args.model == 'vqvae':
        model = VQVAE(args, args.batch_size, args.learning_rate)
    elif args.model == 'evqvae':
        model = EMAVQVAE(args, args.batch_size, args.learning_rate)        
    elif args.model == 'gvqvae':
        model = GumbelVQVAE(args, args.batch_size, args.learning_rate) 
    elif args.model == 'vqvae2':
        model = VQVAE2(args, args.batch_size, args.learning_rate) 

    default_root_dir = args.log_dir

    if args.debug:
        limit_train_batches = 100
        limit_test_batches = 100
        args.backup_steps = 10
        args.image_log_steps = 10
    else:
        limit_train_batches = 1.0
        limit_test_batches = 1.0   

    if args.resume:
        ckpt_path = args.ckpt_path
    else:
        ckpt_path = None

    if args.backup:
        args.backup_dir = os.path.join(args.backup_dir, f'vae/{args.model}')
        backup_callback = ModelCheckpoint(
                                    dirpath=args.backup_dir,
                                    every_n_train_steps = args.backup_steps,
                                    filename='{epoch}_{step}'
                                    )
        
        if len(glob.glob(os.path.join(args.backup_dir,'*.ckpt'))) != 0 :
            ckpt_path = sorted(glob.glob(os.path.join(args.backup_dir,'*.ckpt')))[-1]
            if args.resume:
                print("Setting default ckpt to {}. If this is unexpected behavior, remove {}".format(ckpt_path, ckpt_path))

    if args.wandb:
        logger = pl.loggers.wandb.WandbLogger(project='vqvae', log_model='all')
        logger.watch(model)
    else:
        logger = pl.loggers.tensorboard.TensorBoardLogger(args.log_dir, name='vqvae')                
    if args.use_tpus:
        trainer = Trainer(tpu_cores=tpus, gpus= gpus, default_root_dir=default_root_dir,
                          max_epochs=args.epochs, progress_bar_refresh_rate=args.refresh_rate,precision=args.precision,
                          num_sanity_val_steps=args.num_sanity_val_steps,
                          limit_train_batches=limit_train_batches,limit_test_batches=limit_test_batches,
                          resume_from_checkpoint = ckpt_path,
                          logger = logger)
        if args.xla_stat:
            trainer.callbacks.append(XLAStatsMonitor())
        
    else:
        trainer = Trainer(tpu_cores=tpus, gpus= gpus, default_root_dir=default_root_dir,
                          max_epochs=args.epochs, progress_bar_refresh_rate=args.refresh_rate,precision=args.precision,
                          accelerator='ddp', benchmark=True,
                          num_sanity_val_steps=args.num_sanity_val_steps,
                          limit_train_batches=limit_train_batches,limit_test_batches=limit_test_batches,                          
                          resume_from_checkpoint = ckpt_path,
                          logger = logger)

    if args.backup:
        trainer.callbacks.append(backup_callback)                                 
    if args.log_images:
        trainer.callbacks.append(ReconstructedImageLogger(every_n_steps=args.image_log_steps, use_wandb=args.wandb, multi_optim=args.multi_optim))  
        
    print("Setting batch size: {} learning rate: {:.2e}".format(model.hparams.batch_size, model.hparams.learning_rate))
    
    if not args.test:    
        trainer.fit(model, datamodule=datamodule)
    else:
        trainer.test(model, datamodule=datamodule)



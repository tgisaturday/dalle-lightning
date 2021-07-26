import argparse
from pathlib import Path
import time
from glob import glob
import os
import shutil
import datetime

import torch
from torch.utils.data import DataLoader

from pl_dalle.models.vqgan import VQGAN, EMAVQGAN, GumbelVQGAN
from pl_dalle.models.vqvae import VQVAE, EMAVQVAE, GumbelVQVAE
from pl_dalle.models.vqvae2 import VQVAE2
from pl_dalle.models.dalle import DiscreteVAE, DALLE

from pl_dalle.loader import TextImageDataModule
from pl_dalle.modules.dalle.tokenizer import tokenizer, HugTokenizer, YttmTokenizer
from pl_dalle.callbacks import DalleImageSampler

from torchvision import transforms as T
from PIL import Image
from io import BytesIO

import pytorch_lightning as pl
from pytorch_lightning import seed_everything
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import XLAStatsMonitor
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint

def exists(val):
    return val is not None

def get_trainable_params(model):
    return [params for params in model.parameters() if params.requires_grad]


if __name__ == "__main__":

    # argument parsing
    now = datetime.datetime.now().strftime("%Y-%m-%dT%H-%M-%S")

    parser = argparse.ArgumentParser()

    parser = argparse.ArgumentParser(description='DALL-E Training for Pytorch TPU')

    #path configuration
    parser.add_argument('--train_dir', type=str, default='dataset/train/',
                    help='path to train dataset')
    parser.add_argument('--val_dir', type=str, default='dataset/val/',
                    help='path to val dataset')                    
    parser.add_argument('--log_dir', type=str, default='results/',
                    help='path to save logs')

    parser.add_argument('--vae_path', type=str,
                   help='path to your trained VAE')

    parser.add_argument('--bpe_path', type=str, 
                    help='path to your BPE json file')

    parser.add_argument('--backup_dir', type=str, default='backups/',
                    help='path to save backups for sudden crash') 

    parser.add_argument('--ckpt_path', type=str,default='results/checkpoints/last.ckpt',
                    help='path to previous checkpoint') 
 

    #training configuration
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
    parser.add_argument('--resume', action='store_true', default=False,
                    help='whether to resume from checkpoint')                   
    parser.add_argument('--seed', type=int, default=42,
                    help='random seed')  
    parser.add_argument('--gpus', type=int, default=16,
                    help='number of gpus')    
    parser.add_argument('--tpus', type=int, default=8,
                    help='number of tpus')                                    
    parser.add_argument('--num_sanity_val_steps', type=int, default=0,
                    help='num_sanity_val_steps') 

    parser.add_argument('--batch_size', type=int, default=1,
                    help='training settings')  
    parser.add_argument('--epochs', type=int, default=20,
                    help='training settings')  
    parser.add_argument('--learning_rate', default=3e-4, type=float,
                    help='base learning rate')
    parser.add_argument('--lr_decay', action = 'store_true')                                   
    parser.add_argument('--num_workers', type=int, default=8,
                    help='training settings')   
    parser.add_argument('--img_size', type=int, default=256,
                    help='training settings')  
    parser.add_argument('--clip_grad_norm', default = 0.5, type = float, help = 'Clip gradient norm')
    parser.add_argument('--hug', dest='hug', action='store_true')
    parser.add_argument('--resize_ratio', type=float, default=0.75,
                    help='Random resized crop lower ratio')
    parser.add_argument('--ga_steps', default = 1, type = int, 
                    help = 'Number of steps to accumulate gradients across per each iteration.')                
    parser.add_argument('--truncate_captions', dest='truncate_captions', action='store_true',
                    help='Captions passed in which exceed the max token length will be truncated if this is set.')
    parser.add_argument('--stable_softmax', dest='stable_softmax', action='store_true', default=False,
                    help='Prevent values from becoming too large during softmax. Helps with stability in fp16 and Mixture of Quantization training.')
    parser.add_argument('--sparse_attn', dest='sparse_attn', action='store_true', default=False,
                    help='Use sparse attention')

    parser.add_argument('--test', action='store_true', default=False,
                    help='test run')    
    parser.add_argument('--debug', action='store_true', default=False,
                    help='debug run') 
    parser.add_argument('--xla_stat', action='store_true', default=False,
                    help='print out tpu related stat')     
    parser.add_argument('--web_dataset',action='store_true', default=False,
                    help='enable web_dataset')   
    #VAE configuration
    parser.add_argument('--vae', type=str, default='dvae')

    #Transformer configuration
    parser.add_argument('--attn_types', default = 'full', type = str, 
                    help = 'comma separated list of attention types. attention type can be: full or sparse or axial_row or axial_col or conv_like.')
    parser.add_argument('--hidden_dim', default = 512, type = int, 
                    help = 'Model dimension')
    parser.add_argument('--text_seq_len', default = 128, type = int, 
                    help = 'Text sequence length')
    parser.add_argument('--num_text_tokens', default = 10000, type = int, 
                    help = 'Number of text tokens')                    
    parser.add_argument('--depth', default = 64, type = int, 
                    help = 'Model depth')
    parser.add_argument('--heads', default = 8, type = int, 
                    help = 'Model number of heads')
    parser.add_argument('--dim_head', default = 64, type = int, 
                    help = 'Model head dimension')
    parser.add_argument('--reversible', action='store_true', default=False)                    
    parser.add_argument('--ff_dropout', default = 0.0, type = float, 
                    help = 'Feed forward dropout.')
    parser.add_argument('--attn_dropout', default = 0.0, type = float, 
                    help = 'Feed forward dropout.')
    parser.add_argument('--loss_img_weight', default = 7, type = int, 
                    help = 'Image loss weight')

    args = parser.parse_args()

    #random seed fix
    seed_everything(args.seed)   

    # tokenizer
    if exists(args.bpe_path):
        klass = HugTokenizer if args.hug else YttmTokenizer
        tokenizer = klass(args.bpe_path)  

    default_root_dir = args.log_dir

    if args.resume:
        ckpt_path = args.ckpt_path
    else:
        ckpt_path = None

    if args.backup:
        args.backup_dir = os.path.join(args.backup_dir, f'dalle/{args.vae}')
        backup_callback = ModelCheckpoint(
                                    dirpath=args.backup_dir,
                                    every_n_train_steps = args.backup_steps,
                                    filename='{epoch}_{step}'
                                    )
        
        if len(glob.glob(os.path.join(args.backup_dir,'*.ckpt'))) != 0 :
            ckpt_path = sorted(glob.glob(os.path.join(args.backup_dir,'*.ckpt')))[-1]
            if args.resume:
                print("Setting default ckpt to {}. If this is unexpected behavior, remove {}".format(ckpt_path, ckpt_path))
                


    if args.use_tpus:
        tpus = args.tpus
        gpus = None
    else:
        tpus = None
        gpus = args.gpus

    if args.debug:
        limit_train_batches = 100
        limit_test_batches = 100
        args.backup_steps = 10
        args.image_log_steps = 10
    else:
        limit_train_batches = 1.0
        limit_test_batches = 1.0   

    # model
    if args.vae == 'vqgan':
        vae = VQGAN.load_from_checkpoint(args.vae_path)
    elif args.vae == 'evqgan':
        vae = EMAVQGAN.load_from_checkpoint(args.vae_path)         
    elif args.vae == 'gvqgan':
        vae = GumbelVQGAN.load_from_checkpoint(args.vae_path)       
    elif args.vae == 'vqvae':
        vae = VQVAE.load_from_checkpoint(args.vae_path)
    elif args.vae == 'evqvae':
        vae = EMAVQVAE.load_from_checkpoint(args.vae_path)       
    elif args.vae == 'gvqvae':
        vae = GumbelVQVAE.load_from_checkpoint(args.vae_path) 
    elif args.vae == 'vqvae2':
        vae = VQVAE2.load_from_checkpoint(args.vae_path) 

    elif args.vae == 'dvae':
        loaded_obj = torch.load(str(args.vae_path),map_location=torch.device('cpu'))
        vae_params, weights = loaded_obj['hparams'], loaded_obj['weights']
        vae = DiscreteVAE(**vae_params)
    
    args.image_seq_len = vae.num_tokens
    
    model = DALLE(args, args.batch_size, args.learning_rate, vae=vae)


    datamodule = TextImageDataModule(args.train_dir, args.val_dir, 
                                args.batch_size, args.num_workers, 
                                args.img_size, args.text_seq_len, 
                                args.resize_ratio,args.truncate_captions, 
                                tokenizer,
                                args.fake_data, args.web_dataset)


    if args.use_tpus:
        trainer = Trainer(tpu_cores=tpus, gpus= gpus, default_root_dir=default_root_dir,
                          max_epochs=args.epochs, progress_bar_refresh_rate=args.refresh_rate,precision=args.precision,
                          gradient_clip_val=args.clip_grad_norm, accumulate_grad_batches=args.ga_steps,
                          num_sanity_val_steps=args.num_sanity_val_steps,
                          limit_train_batches=limit_train_batches,limit_test_batches=limit_test_batches,                          
                          resume_from_checkpoint = ckpt_path)
        if args.xla_stat:
            trainer.callbacks.append(XLAStatsMonitor())                         
    else:
        trainer = Trainer(tpu_cores=tpus, gpus= gpus, default_root_dir=default_root_dir,
                          max_epochs=args.epochs, progress_bar_refresh_rate=args.refresh_rate,precision=args.precision,
                          accelerator='ddp', accumulate_grad_batches=args.ga_steps,
                          gradient_clip_val=args.clip_grad_norm,
                          num_sanity_val_steps=args.num_sanity_val_steps,
                          limit_train_batches=limit_train_batches,limit_test_batches=limit_test_batches,                          
                          resume_from_checkpoint = ckpt_path)

    if args.backup:
        trainer.callbacks.append(backup_callback)      
    if args.log_images:
        trainer.callbacks.append(DalleImageSampler(every_n_steps=args.image_log_steps, text_seq_len=args.text_seq_len, image_seq_len=args.image_seq_len))  
        
    print("Setting batch size: {} learning rate: {:.2e}".format(model.hparams.batch_size, model.hparams.learning_rate))
    
    if not args.test:    
        trainer.fit(model, datamodule=datamodule)
    else:
        trainer.test(model, datamodule=datamodule)

import argparse, os, sys, datetime, glob, importlib
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

import pytorch_lightning as pl
from pytorch_lightning import seed_everything
from pytorch_lightning import Trainer




if __name__ == "__main__":

    now = datetime.datetime.now().strftime("%Y-%m-%dT%H-%M-%S")


    parser = argparse.ArgumentParser(description='VQVAE Training for Pytorch TPU')

    #path configuration
    parser.add_argument('--train_dir', type=str, default='dataset/train/',
                    help='path to train dataset')
    parser.add_argument('--val_dir', type=str, default='dataset/val/',
                    help='path to val dataset')                    
    parser.add_argument('--log_dir', type=str, default='results/',
                    help='path to save logs')
    parser.add_argument('--ckpt_path', type=str,default='results/checkpoints/last.ckpt',
                    help='path to previous checkpoint')  

    #training configuration
    parser.add_argument('--refresh_rate', type=int, default=1,
                    help='progress bar refresh rate')    
    parser.add_argument('--precision', type=int, default=16,
                    help='precision for training')                     
    parser.add_argument('--fake_data', action='store_true', default=False,
                    help='using fake_data for debugging') 
    parser.add_argument('--use_tpus', action='store_true', default=False,
                    help='using tpu')
    parser.add_argument('--log_images', action='store_true', default=False,
                    help='log image outputs. not recommended for tpus')                                                                         
    parser.add_argument('--resume', action='store_true', default=False,
                    help='whether to resume from checkpoint')                   
    parser.add_argument('--seed', type=int, default=42,
                    help='random seed')  
    parser.add_argument('--gpus', type=int, default=16,
                    help='number of gpus')                   
    parser.add_argument('--num_sanity_val_steps', type=int, default=0,
                    help='num_sanity_val_steps')                     
    parser.add_argument('--learning_rate', default=4.5e-3, type=float,
                    help='base learning rate')
    parser.add_argument('--lr_decay_rate', type = float, default = 0.98, 
                    help = 'learning rate decay')
    parser.add_argument('--starting_temp', type = float, default = 1., 
                    help = 'starting temperature')
    parser.add_argument('--temp_min', type = float, default = 0.5, 
                    help = 'minimum temperature to anneal to')
    parser.add_argument('--anneal_rate', type = float, default = 1e-6, 
                    help = 'temperature annealing rate')          
    parser.add_argument('--batch_size', type=int, default=8,
                    help='training settings')  
    parser.add_argument('--epochs', type=int, default=30,
                    help='training settings')                                    
    parser.add_argument('--num_workers', type=int, default=8,
                    help='training settings')   
    parser.add_argument('--img_size', type=int, default=256,
                    help='training settings')
    parser.add_argument('--resize_ratio', type=float, default=0.75,
                    help='Random resized crop lower ratio')

    parser.add_argument('--test', action='store_true', default=False,
                    help='test run')                     

    #model configuration
    parser.add_argument('--model', type=str, default='vqgan')
    parser.add_argument('--embed_dim', type=int, default=256,
                    help='number of embedding dimension for codebook')       
    parser.add_argument('--codebook_dim', type=int, default=1024,
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
    parser.add_argument('--ch_mult', type=list, default=[1,1,2,2,4],
                    help='resnet channel multiplier')  
    parser.add_argument('--num_res_blocks', type=int, default=2,
                    help='number of resnet blocks')                     
    parser.add_argument('--attn_resolutions', type=list, default=[16],
                    help='model settings')  
    parser.add_argument('--dropout', type=float, default=0.0,
                    help='model settings') 
    parser.add_argument('--quant_beta', type=float, default=0.5,
                    help='quantizer beta')                     
    parser.add_argument('--quant_ema_decay', type=float, default=0.99,
                    help='quantizer ema decay')
    parser.add_argument('--quant_ema_eps', type=float, default=1e-5,
                    help='quantizer ema epsilon')  
                              
    #vqvae2 specialized options
    parser.add_argument('--num_res_ch', type=int, default=32,
                    help='model settings')                                        
    parser.add_argument('--latent_weight', type=float, default=0.25,
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

    #misc configuration
 
    args = parser.parse_args()

    #random seed fix
    seed_everything(args.seed)   

    #data = ImageDataModule(args.train_dir, args.val_dir, args.batch_size, args.num_workers, args.img_size, args.fake_data)
    
    transform_train = T.Compose([
                            T.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
                            T.RandomResizedCrop(args.img_size,
                                    scale=(args.resize_ratio, 1.),ratio=(1., 1.)),
                            T.ToTensor(),
                            T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                            ])
    transform_val = T.Compose([
                                    T.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
                                    T.Resize(args.img_size),
                                    T.CenterCrop(args.img_size),
                                    T.ToTensor(),
                                    T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                                    ])
 
    if args.fake_data:
        import torch_xla.utils.utils as xu
        import torch_xla.core.xla_model as xm        
        train_loader = xu.SampleGenerator(
                        data=(torch.zeros(args.batch_size, 3, args.img_size , args.img_size ),
                        torch.zeros(args.batch_size, dtype=torch.int64)),
                        sample_count=1200000 // args.batch_size // xm.xrt_world_size())
        val_loader = xu.SampleGenerator(
                        data=(torch.zeros(args.batch_size, 3, args.img_size , args.img_size ),
                        torch.zeros(args.batch_size, dtype=torch.int64)),
                        sample_count=50000 // args.batch_size // xm.xrt_world_size())                           
    else:
        train_dataset = ImageFolder(args.train_dir, transform_train)
        val_dataset = ImageFolder(args.val_dir, transform_val)          
        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, num_workers=args.num_workers,shuffle=True)      
        val_loader = DataLoader(val_dataset, batch_size=args.batch_size, num_workers=args.num_workers)  

    # model
    if args.model == 'vqgan':
        model = VQGAN(args, args.batch_size, args.learning_rate, args.log_images)
    elif args.model == 'evqgan':
        model = EMAVQGAN(args, args.batch_size, args.learning_rate, args.log_images)          
    elif args.model == 'gvqgan':
        model = GumbelVQGAN(args, args.batch_size, args.learning_rate, args.log_images)        
    elif args.model == 'vqvae':
        model = VQVAE(args, args.batch_size, args.learning_rate, args.log_images)
    elif args.model == 'evqvae':
        model = EMAVQVAE(args, args.batch_size, args.learning_rate, args.log_images)        
    elif args.model == 'gvqvae':
        model = GumbelVQVAE(args, args.batch_size, args.learning_rate, args.log_images) 
    elif args.model == 'vqvae2':
        model = VQVAE2(args, args.batch_size, args.learning_rate, args.log_images) 

    default_root_dir = args.log_dir
    if args.resume:
        ckpt_path = args.ckpt_path
    else:
        ckpt_path = None

    if args.use_tpus:
        tpus = 8
        gpus = None
    else:
        tpus = None
        gpus = args.gpus

    if args.use_tpus:
        trainer = Trainer(tpu_cores=tpus, gpus= gpus, default_root_dir=default_root_dir,
                          max_epochs=args.epochs, progress_bar_refresh_rate=args.refresh_rate,precision=args.precision,
                          num_sanity_val_steps=args.num_sanity_val_steps,
                          resume_from_checkpoint = ckpt_path)
    else:
        trainer = Trainer(tpu_cores=tpus, gpus= gpus, default_root_dir=default_root_dir,
                          max_epochs=args.epochs, progress_bar_refresh_rate=args.refresh_rate,precision=args.precision,
                          accelerator='ddp',
                          num_sanity_val_steps=args.num_sanity_val_steps,
                          resume_from_checkpoint = ckpt_path)
    
    print("Setting batch size: {} learning rate: {:.2e}".format(model.hparams.batch_size, model.hparams.learning_rate))
    
    if not args.test:    
        trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader)
    else:
        trainer.test(model, dataloaders=val_loader)



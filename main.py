import argparse, os, sys, datetime, glob, importlib
import numpy as np
import random
from PIL import Image
import torch

# vision imports

from torchvision import transforms as T
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from taming.models.vqgan import VQModel

import pytorch_lightning as pl
from pytorch_lightning import seed_everything
from pytorch_lightning import LightningDataModule, LightningModule, Trainer
from pytorch_lightning.utilities.distributed import rank_zero_only

class ImageDataModule(LightningDataModule):

    def __init__(self, train_dir, val_dir, batch_size, num_workers):
        super().__init__()
        self.train_dir = train_dir
        self.val_dir = val_dir
        self.batch_size = batch_size
        self.num_workers = num_workers

        self.transform = T.Compose([
                                    T.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
                                    T.Resize(args.img_size),
                                    T.CenterCrop(args.img_size),
                                    T.ToTensor()
                                    ])
                                    
    def setup(self, stage=None):
        self.train_dataset = ImageFolder(self.train_dir, self.transform)
        self.val_dataset = ImageFolder(self.val_dir, self.transform)
  

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, num_workers=self.num_workers,shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, num_workers=self.num_workers)

    def test_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, num_workers=self.num_workers)

if __name__ == "__main__":

    now = datetime.datetime.now().strftime("%Y-%m-%dT%H-%M-%S")

    cwd = sys.path.append(os.getcwd())

    parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')

    #path configuration
    parser.add_argument('train_dir', type=str, default=f'{cwd}/data/',
                    help='path to train dataset')
    parser.add_argument('val_dir', type=str, default=f'{cwd}/data/',
                    help='path to val dataset')                    
    parser.add_argument('log_dir', type=str,default=f'{cwd}/results/',
                    help='path to save logs')
    parser.add_argument('ckpt_path', type=str,default=f'{cwd}/results/checkpoints/last.ckpt',
                    help='path to previous checkpoint')

    #training configuration
    parser.add_argument('--use_tpus', action='store_true', default=False,
                    help='using tpu') 
    parser.add_argument('--is_pod', action='store_true', default=False,
                    help='using tpu as pod')    
    parser.add_argument('--auto_lr_find', action='store_true', default=False,
                    help='using auto lr find')   
    parser.add_argument('--auto_scale_batch_size', action='store_true', default=False,
                    help='using auto scale batch size')                                                              
    parser.add_argument('--resume', action='store_true', default=False,
                    help='whether to resume from checkpoint')                   
    parser.add_argument('--seed', type=int, default=42,
                    help='random seed')  
    parser.add_argument('--gpus', type=int, default=16,
                    help='number of gpus')                   
                    
    parser.add_argument('--learning_rate', default=4.5e-6, type=float,
                    help='base learning rate')
    parser.add_argument('--batch_size', type=int, default=6,
                    help='dataconfig')  
    parser.add_argument('--epochs', type=int, default=30,
                    help='dataconfig')                                    
    parser.add_argument('--num_workers', type=int, default=8,
                    help='dataconfig')   
    parser.add_argument('--img_size', type=int, default=256,
                    help='dataconfig')

    parser.add_argument('--test', action='store_true', default=False,
                    help='test run')                     

    #model configuration
    parser.add_argument('--embed_dim', type=int, default=256,
                    help='number of embedding dimension')       
    parser.add_argument('--n_embed', type=int, default=8192,
                    help='codebook size')        
    parser.add_argument('--double_z', type=bool, default=False,
                    help='ddconfig')
    parser.add_argument('--z_channels', type=int, default=256,
                    help='ddconfig')
    parser.add_argument('--resolution', type=int, default=256,
                    help='ddconfig')
    parser.add_argument('--in_channels', type=int, default=3,
                    help='ddconfig')
    parser.add_argument('--out_ch', type=int, default=3,
                    help='ddconfig')    
    parser.add_argument('--ch', type=int, default=128,
                    help='ddconfig')  
    parser.add_argument('--ch_mult', type=list, default=[1,1,2,2,4],
                    help='ddconfig')  
    parser.add_argument('--num_res_blocks', type=int, default=2,
                    help='ddconfig')                     
    parser.add_argument('--attn_resolutions', type=list, default=[16],
                    help='ddconfig')  
    parser.add_argument('--dropout', type=float, default=0.0,
                    help='ddconfig')  

    #loss configuration
    parser.add_argument('--disc_conditional', type=bool, default=False,
                    help='lossconfig')      
    parser.add_argument('--disc_in_channels', type=int, default=3,
                    help='lossconfig') 
    parser.add_argument('--disc_start', type=int, default=250001,
                    help='lossconfig') 
    parser.add_argument('--disc_weight', type=float, default=0.8,
                    help='lossconfig') 
    parser.add_argument('--codebook_weight', type=float, default=1.0,
                    help='lossconfig') 

    #misc configuration
 
    args = parser.parse_args()

    #random seed fix
    seed_everything(args.seed)   

    data = ImageDataModule()
    #data.setup()

    # model
    model = VQModel(args)
 
    if args.use_pod:
        global_rank = os.environ["CLOUD_TPU_TASK_ID"]
        default_root_dir = os.path.join(args.log_dir,global_rank)
    else:
        default_root_dir = args.log_dir

    if args.auto_scale_batch_size:
        auto_scale_batch_size = 'binsearch'
    else:
        auto_scale_batch_size = None

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

  
    trainer = Trainer(tpu_cores=tpus, gpus= gpus, default_root_dir=default_root_dir,
                          max_epochs=5, progress_bar_refresh_rate=20,precision=16,
                          resume_from_checkpoint = ckpt_path,
                          auto_lr_find=args.auto_lr_find, 
                          auto_scale_batch_size=auto_scale_batch_size)


    if args.auto_lr_find or args.auto_scale_batch_size:
        trainer.tune(model)



    print("Setting batch size: {} learning rate: {:.2e}".format(args.batch_size, args.learning_rate))

    if not args.test:    
        trainer.fit(model, data)
    else:
        trainer.test(model, data)



class ImageDataModule(LightningDataModule):

    def __init__(self, train_dir, val_dir, batch_size, num_workers, img_size, fake_data=False):
        super().__init__()
        self.train_dir = train_dir
        self.val_dir = val_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.fake_data = fake_data
        self.img_size = img_size

        self.transform = T.Compose([
                                    T.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
                                    T.Resize(self.img_size),
                                    T.CenterCrop(self.img_size),
                                    T.ToTensor()
                                    ])
                                    
    def setup(self, stage=None):
        if not self.fake_data:
            self.train_dataset = ImageFolder(self.train_dir, self.transform)
            self.val_dataset = ImageFolder(self.val_dir, self.transform)
  

    def train_dataloader(self):
        if self.fake_data:
            import torch_xla.utils.utils as xu
            import torch_xla.core.xla_model as xm
            train_loader = xu.SampleGenerator(
                            data=(torch.zeros(self.batch_size, 3, self.img_size , self.img_size ),
                            torch.zeros(self.batch_size, dtype=torch.int64)),
                            sample_count=1200000 // self.batch_size // xm.xrt_world_size())
            return train_loader
        else:
            return DataLoader(self.train_dataset, batch_size=self.batch_size, num_workers=self.num_workers,shuffle=True)

    def val_dataloader(self):
        if self.fake_data:
            import torch_xla.utils.utils as xu
            import torch_xla.core.xla_model as xm            
            val_loader = xu.SampleGenerator(
                            data=(torch.zeros(self.batch_size, 3, self.img_size , self.img_size ),
                            torch.zeros(self.batch_size, dtype=torch.int64)),
                            sample_count=50000 // self.batch_size // xm.xrt_world_size())            
            return val_loader
        else:
            return DataLoader(self.val_dataset, batch_size=self.batch_size, num_workers=self.num_workers)

    def test_dataloader(self):
        if self.fake_data:
            import torch_xla.utils.utils as xu
            import torch_xla.core.xla_model as xm
            val_loader = xu.SampleGenerator(
                            data=(torch.zeros(self.batch_size, 3, self.img_size , self.img_size ),
                            torch.zeros(self.batch_size, dtype=torch.int64)),
                            sample_count=50000 // self.batch_size // xm.xrt_world_size())            
            return val_loader            
        else:
            return DataLoader(self.val_dataset, batch_size=self.batch_size, num_workers=self.num_workers)

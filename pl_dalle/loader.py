from pathlib import Path
from random import randint, choice

import PIL

from torch.utils.data import Dataset, DataLoader
from torchvision import transforms as T
from torchvision.datasets import ImageFolder
from pytorch_lightning import LightningDataModule

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
                                    T.ToTensor(),
                                    T.Normalize(((0.5,) * 3, (0.5,) * 3)),           
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


class TextImageDataset(Dataset):
    def __init__(self,
                 folder,
                 text_len=256,
                 image_size=128,
                 truncate_captions=False,
                 resize_ratio=0.75,
                 tokenizer=None,
                 transform=None,
                 shuffle=False
                 ):
        """
        @param folder: Folder containing images and text files matched by their paths' respective "stem"
        @param truncate_captions: Rather than throw an exception, captions which are too long will be truncated.
        """
        super().__init__()
        self.shuffle = shuffle
        path = Path(folder)

        text_files = [*path.glob('**/*.txt')]
        image_files = [
            *path.glob('**/*.png'), *path.glob('**/*.jpg'),
            *path.glob('**/*.jpeg'), *path.glob('**/*.bmp')
        ]

        text_files = {text_file.stem: text_file for text_file in text_files}
        image_files = {image_file.stem: image_file for image_file in image_files}

        keys = (image_files.keys() & text_files.keys())

        self.keys = list(keys)
        self.text_files = {k: v for k, v in text_files.items() if k in keys}
        self.image_files = {k: v for k, v in image_files.items() if k in keys}
        self.text_len = text_len
        self.truncate_captions = truncate_captions
        self.resize_ratio = resize_ratio
        self.tokenizer = tokenizer
        self.image_transform = transform

    def __len__(self):
        return len(self.keys)

    def random_sample(self):
        return self.__getitem__(randint(0, self.__len__() - 1))

    def sequential_sample(self, ind):
        if ind >= self.__len__() - 1:
            return self.__getitem__(0)
        return self.__getitem__(ind + 1)

    def skip_sample(self, ind):
        if self.shuffle:
            return self.random_sample()
        return self.sequential_sample(ind=ind)

    def __getitem__(self, ind):
        key = self.keys[ind]

        text_file = self.text_files[key]
        image_file = self.image_files[key]

        descriptions = text_file.read_text().split('\n')
        descriptions = list(filter(lambda t: len(t) > 0, descriptions))
        try:
            description = choice(descriptions)
        except IndexError as zero_captions_in_file_ex:
            print(f"An exception occurred trying to load file {text_file}.")
            print(f"Skipping index {ind}")
            return self.skip_sample(ind)

        tokenized_text = self.tokenizer.tokenize(
            description,
            self.text_len,
            truncate_text=self.truncate_captions
        ).squeeze(0)
        try:
            image_tensor = self.image_transform(PIL.Image.open(image_file))
        except (PIL.UnidentifiedImageError, OSError) as corrupt_image_exceptions:
            print(f"An exception occurred trying to load file {image_file}.")
            print(f"Skipping index {ind}")
            return self.skip_sample(ind)

        # Success
        return tokenized_text, image_tensor

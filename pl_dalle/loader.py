from pathlib import Path
from random import randint, choice

import PIL

from torch.utils.data import Dataset, DataLoader
from torchvision import transforms as T
from torchvision.datasets import ImageFolder, FakeData, VisionDataset
from pytorch_lightning import LightningDataModule
import torch
from typing import Any, Callable, Optional, Tuple
from torchvision import transforms

class ImageDataModule(LightningDataModule):

    def __init__(self, train_dir, val_dir, batch_size, num_workers, img_size, resize_ratio=0.75, fake_data=False):
        super().__init__()
        self.train_dir = train_dir
        self.val_dir = val_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.fake_data = fake_data
        self.img_size = img_size

        self.transform_train = T.Compose([
                            T.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
                            T.RandomResizedCrop(img_size,
                                    scale=(resize_ratio, 1.),ratio=(1., 1.)),
                            T.ToTensor(),
                            T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                            ])
        self.transform_val = T.Compose([
                                    T.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
                                    T.Resize(img_size),
                                    T.CenterCrop(img_size),
                                    T.ToTensor(),
                                    T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                                    ])
                                    
    def setup(self, stage=None):
        if self.fake_data:
            self.train_dataset = FakeData(1200000, (3, self.img_size, self.img_size), 1000, self.transform_train)
            self.val_dataset = FakeData(50000, (3, self.img_size, self.img_size), 1000, self.transform_train)
        else:
            self.train_dataset = ImageFolder(self.train_dir, self.transform)
            self.val_dataset = ImageFolder(self.val_dir, self.transform)
  

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, num_workers=self.num_workers,shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, num_workers=self.num_workers)

    def test_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, num_workers=self.num_workers)


class TextImageDataModule(LightningDataModule):

    def __init__(self, train_dir, val_dir, batch_size, num_workers, img_size, text_seq_len,
                resize_ratio=0.75, truncate_captions=False, tokenizer=None, fake_data=False):
        super().__init__()
        self.train_dir = train_dir
        self.val_dir = val_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.img_size = img_size
        self.text_seq_len = text_seq_len
        self.resize_ratio = resize_ratio
        self.truncate_captions = truncate_captions
        self.tokenizer = tokenizer
        self.fake_data = fake_data

        self.transform_train = T.Compose([
                            T.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
                            T.RandomResizedCrop(img_size,
                                    scale=(resize_ratio, 1.),ratio=(1., 1.)),
                            T.ToTensor(),
                            T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                            ])
        self.transform_val = T.Compose([
                                    T.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
                                    T.Resize(img_size),
                                    T.CenterCrop(img_size),
                                    T.ToTensor(),
                                    T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                                    ])
                                    
    def setup(self, stage=None):
        if self.fake_data:
            self.train_dataset = FakeTextImageData(1200000, (3, self.img_size, self.img_size), self.text_seq_len, self.transform_train)
            self.val_dataset = FakeTextImageData(50000, (3, self.img_size, self.img_size), self.text_seq_len, self.transform_train)
        else:
            self.train_dataset = TextImageDataset(
                                    self.train_dir,
                                    text_len=self.text_seq_len,
                                    image_size=self.image_size,
                                    resize_ratio=self.resize_ratio,
                                    truncate_captions=self.truncate_captions,
                                    tokenizer=self.tokenizer,
                                    transform=self.transform_train,
                                    shuffle=True,
                                    )
            self.val_dataset = TextImageDataset(
                                    self.val_dir,
                                    text_len=self.text_seq_len,
                                    image_size=self.image_size,
                                    resize_ratio=self.resize_ratio,
                                    truncate_captions=self.truncate_captions,
                                    tokenizer=self.tokenizer,
                                    transform=self.transform_val,
                                    shuffle=False,
                                    )

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, num_workers=self.num_workers,shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, num_workers=self.num_workers)

    def test_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, num_workers=self.num_workers)


class FakeTextImageData(VisionDataset):
    def __init__(
            self,
            size: int = 1000,
            image_size: Tuple[int, int, int] = (3, 224, 224),
            text_len: int = 10,
            transform: Optional[Callable] = None,
            target_transform: Optional[Callable] = None,
            random_offset: int = 0,
    ) -> None:
        super(FakeTextImageData, self).__init__(None, transform=transform,  # type: ignore[arg-type]
                                       target_transform=target_transform)
        self.size = size
        self.text_len = text_len
        self.image_size = image_size
        self.random_offset = random_offset

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is class_index of the target class.
        """
        # create random image that is consistent with the index id
        if index >= len(self):
            raise IndexError("{} index out of range".format(self.__class__.__name__))
        rng_state = torch.get_rng_state()
        torch.manual_seed(index + self.random_offset)
        img = torch.randn(*self.image_size)
        target = torch.zeros(self.text_len)
        torch.set_rng_state(rng_state)

        # convert to PIL Image
        img = transforms.ToPILImage()(img)
        if self.transform is not None:
            img = self.transform(img)

        return img, target

    def __len__(self) -> int:
        return self.size

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

from pytorch_lightning.utilities.cli import LightningCLI

from pl_dalle.loader import ImageDataModule
from pl_dalle.models.vdvqvae import VDVQVAE

cli = LightningCLI(VDVQVAE, ImageDataModule)
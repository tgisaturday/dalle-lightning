#borrowed from https://github.com/PyTorchLightning/Lightning-Bolts/blob/master/pl_bolts/callbacks/vision/image_generation.py#L15-L97
from typing import Any, Dict, List, Optional, Tuple

import torch
import pytorch_lightning as pl
from pytorch_lightning import Callback, LightningModule, Trainer

from pl_bolts.utils import _TORCHVISION_AVAILABLE
from pl_bolts.utils.warnings import warn_missing_pkg
from pytorch_lightning.utilities.types import STEP_OUTPUT
from pytorch_lightning.utilities.distributed import rank_zero_only

if _TORCHVISION_AVAILABLE:
    import torchvision
else:  # pragma: no cover
    warn_missing_pkg("torchvision")


class VAEImageSampler(Callback):
    
    def __init__(
        self,
        num_samples: int = 3,
        every_n_steps: int = 1000,
        nrow: int = 8,
        padding: int = 2,
        normalize: bool = True,
        norm_range: Optional[Tuple[int, int]] = None,
        scale_each: bool = False,
        pad_value: int = 0,
    ) -> None:
        """
        Args:
            num_samples: Number of images displayed in the grid. Default: ``3``.
            nrow: Number of images displayed in each row of the grid.
                The final grid size is ``(B / nrow, nrow)``. Default: ``8``.
            padding: Amount of padding. Default: ``2``.
            normalize: If ``True``, shift the image to the range (0, 1),
                by the min and max values specified by :attr:`range`. Default: ``True``.
            norm_range: Tuple (min, max) where min and max are numbers,
                then these numbers are used to normalize the image. By default, min and max
                are computed from the tensor.
            scale_each: If ``True``, scale each image in the batch of
                images separately rather than the (min, max) over all images. Default: ``False``.
            pad_value: Value for the padded pixels. Default: ``0``.
        """
        if not _TORCHVISION_AVAILABLE:  # pragma: no cover
            raise ModuleNotFoundError("You want to use `torchvision` which is not installed yet.")

        super().__init__()
        self.num_samples = num_samples
        self.every_n_steps = every_n_steps
        self.nrow = nrow
        self.padding = padding
        self.normalize = normalize
        self.norm_range = norm_range
        self.scale_each = scale_each
        self.pad_value = pad_value

    @rank_zero_only
    def on_train_batch_end(
        self,
        trainer: 'pl.Trainer',
        pl_module: 'pl.LightningModule',
        outputs: Optional[STEP_OUTPUT],
        batch: Any,
        batch_idx: int,
        dataloader_idx: int,
    ) -> None:
        """Called when the train batch ends."""
        if trainer.global_step % self.every_n_steps == 0:
            
            x, _ = batch
            x = x.to(pl_module.device)
            with torch.no_grad():
                pl_module.eval()
                xrec, _ = pl_module(x)
                pl_module.train()   

            x_grid = torchvision.utils.make_grid(
                tensor=x,
                nrow=self.nrow,
                padding=self.padding,
                normalize=self.normalize,
                value_range=self.norm_range,
                scale_each=self.scale_each,
                pad_value=self.pad_value,
            )           
            xrec_grid = torchvision.utils.make_grid(
                tensor=xrec,
                nrow=self.nrow,
                padding=self.padding,
                normalize=self.normalize,
                value_range=self.norm_range,
                scale_each=self.scale_each,
                pad_value=self.pad_value,
            )    
            x_title = "train/input"
            trainer.logger.experiment.add_image(x_title, x_grid, global_step=trainer.global_step)
            xrec_title = "train/reconstruction"
            trainer.logger.experiment.add_image(xrec_title, xrec_grid, global_step=trainer.global_step)

    @rank_zero_only
    def on_validation_batch_end(
        self,
        trainer: 'pl.Trainer',
        pl_module: 'pl.LightningModule',
        outputs: Optional[STEP_OUTPUT],
        batch: Any,
        batch_idx: int,
        dataloader_idx: int,
    ) -> None:
        """Called when the validation batch ends."""
        if trainer.global_step % self.every_n_steps == 0:
            x, _ = batch
            x = x.to(pl_module.device)
            with torch.no_grad():
                pl_module.eval()
                xrec, _ = pl_module(x)
                pl_module.train()   

            x_grid = torchvision.utils.make_grid(
                tensor=x,
                nrow=self.nrow,
                padding=self.padding,
                normalize=self.normalize,
                value_range=self.norm_range,
                scale_each=self.scale_each,
                pad_value=self.pad_value,
            )           
            xrec_grid = torchvision.utils.make_grid(
                tensor=xrec,
                nrow=self.nrow,
                padding=self.padding,
                normalize=self.normalize,
                value_range=self.norm_range,
                scale_each=self.scale_each,
                pad_value=self.pad_value,
            )    
            x_title = "val/input"
            trainer.logger.experiment.add_image(x_title, x_grid, global_step=trainer.global_step)
            xrec_title = "val/reconstruction"
            trainer.logger.experiment.add_image(xrec_title, xrec_grid, global_step=trainer.global_step)


class DalleImageSampler(Callback):
    
    def __init__(
        self,
        num_samples: int = 3,
        every_n_steps: int = 1000,
        nrow: int = 8,
        padding: int = 2,
        normalize: bool = True,
        norm_range: Optional[Tuple[int, int]] = None,
        scale_each: bool = False,
        pad_value: int = 0,
    ) -> None:
        """
        Args:
            num_samples: Number of images displayed in the grid. Default: ``3``.
            nrow: Number of images displayed in each row of the grid.
                The final grid size is ``(B / nrow, nrow)``. Default: ``8``.
            padding: Amount of padding. Default: ``2``.
            normalize: If ``True``, shift the image to the range (0, 1),
                by the min and max values specified by :attr:`range`. Default: ``True``.
            norm_range: Tuple (min, max) where min and max are numbers,
                then these numbers are used to normalize the image. By default, min and max
                are computed from the tensor.
            scale_each: If ``True``, scale each image in the batch of
                images separately rather than the (min, max) over all images. Default: ``False``.
            pad_value: Value for the padded pixels. Default: ``0``.
        """
        if not _TORCHVISION_AVAILABLE:  # pragma: no cover
            raise ModuleNotFoundError("You want to use `torchvision` which is not installed yet.")

        super().__init__()
        self.num_samples = num_samples
        self.every_n_steps = every_n_steps
        self.nrow = nrow
        self.padding = padding
        self.normalize = normalize
        self.norm_range = norm_range
        self.scale_each = scale_each
        self.pad_value = pad_value

    @rank_zero_only
    def on_train_batch_end(
        self,
        trainer: 'pl.Trainer',
        pl_module: 'pl.LightningModule',
        outputs: Optional[STEP_OUTPUT],
        batch: Any,
        batch_idx: int,
        dataloader_idx: int,
    ) -> None:
        """Called when the train batch ends."""
        if trainer.global_step % self.every_n_steps == 0:          
            text, x = batch
            text = text.to(pl_module.device)
            x = x.to(pl_module.device)       
            with torch.no_grad():
                pl_module.eval()
                logits = pl_module(text, x, return_loss=False)
                xrec = pl_module.vae.decode(logits)
                pl_module.train()   

            x_grid = torchvision.utils.make_grid(
                tensor=x,
                nrow=self.nrow,
                padding=self.padding,
                normalize=self.normalize,
                value_range=self.norm_range,
                scale_each=self.scale_each,
                pad_value=self.pad_value,
            )           
            xrec_grid = torchvision.utils.make_grid(
                tensor=xrec,
                nrow=self.nrow,
                padding=self.padding,
                normalize=self.normalize,
                value_range=self.norm_range,
                scale_each=self.scale_each,
                pad_value=self.pad_value,
            )    
            x_title = "train/input"
            trainer.logger.experiment.add_image(x_title, x_grid, global_step=trainer.global_step)
            xrec_title = "train/reconstruction"
            trainer.logger.experiment.add_image(xrec_title, xrec_grid, global_step=trainer.global_step)

    @rank_zero_only
    def on_validation_batch_end(
        self,
        trainer: 'pl.Trainer',
        pl_module: 'pl.LightningModule',
        outputs: Optional[STEP_OUTPUT],
        batch: Any,
        batch_idx: int,
        dataloader_idx: int,
    ) -> None:
        """Called when the validation batch ends."""
        if trainer.global_step % self.every_n_steps == 0:  
            text, x = batch
            text = text.to(pl_module.device)
            x = x.to(pl_module.device)       
            with torch.no_grad():
                pl_module.eval()
                logits = pl_module(text, x, return_loss=False)
                xrec = pl_module.vae.decode(logits)
                pl_module.train() 
            x_grid = torchvision.utils.make_grid(
                tensor=x,
                nrow=self.nrow,
                padding=self.padding,
                normalize=self.normalize,
                value_range=self.norm_range,
                scale_each=self.scale_each,
                pad_value=self.pad_value,
            )           
            xrec_grid = torchvision.utils.make_grid(
                tensor=xrec,
                nrow=self.nrow,
                padding=self.padding,
                normalize=self.normalize,
                value_range=self.norm_range,
                scale_each=self.scale_each,
                pad_value=self.pad_value,
            )    
            x_title = "val/input"
            trainer.logger.experiment.add_image(x_title, x_grid, global_step=trainer.global_step)
            xrec_title = "val/reconstruction"
            trainer.logger.experiment.add_image(xrec_title, xrec_grid, global_step=trainer.global_step)

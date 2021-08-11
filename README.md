# Text-to-Image Translation (DALL-E) for TPU in Pytorch

Refactoring 
[Taming Transformers](https://github.com/CompVis/taming-transformers) and [DALLE-pytorch](https://github.com/lucidrains/DALLE-pytorch)
for TPU VM with [Pytorch Lightning](https://github.com/PyTorchLightning/pytorch-lightning)

## Requirements

```
pip install -r requirements.txt
```

## Data Preparation

Place any image dataset with ImageNet-style directory structure (at least 1 subfolder) to fit the dataset into pytorch ImageFolder.

## Training VQVAEs
You can easily test main.py with randomly generated fake data.
```
python train_vae.py --use_tpus --fake_data
```

For actual training provide specific directory for train_dir, val_dir, log_dir:

```
python train_vae.py --use_tpus --train_dir [training_set] --val_dir [val_set] --log_dir [where to save results]
```

## Training DALL-E
```
python train_dalle.py --use_tpus --train_dir [training_set] --val_dir [val_set] --log_dir [where to save results] --vae_path [pretrained vae] --bpe_path [pretrained bpe(optional)]
```

## TODO
- [ ] Refactor Encoder and Decoder modules for better readability
- [ ] Refactor VQVAE2
- [ ] Add Net2Net Conditional Transformer for conditional image generation
- [ ] Refactor, optimize, and merge DALL-E with Net2Net Conditional Transformer
- [ ] Add Guided Diffusion + CLIP for image refinement
- [ ] Add VAE converter for JAX to support [dalle-mini](https://github.com/borisdayma/dalle-mini)
- [ ] Add DALL-E colab notebook
- [ ] Add [RBGumbelQuantizer](https://arxiv.org/abs/2010.04838)
- [ ] Add [HiT](https://arxiv.org/abs/2106.07631)

## ON-GOING
- [ ] Test large dataset loading on TPU Pods
- [ ] Change current DALL-E code to fully support latest updates from [DALLE-pytorch](https://github.com/lucidrains/DALLE-pytorch) 

## DONE
- [x] Add VQVAE, VQGAN, and Gumbel VQVAE(Discrete VAE), Gumbel VQGAN
- [x] Add [VQVAE2](https://arxiv.org/abs/1906.00446)
- [x] Add EMA update for Vector Quantization
- [x] Debug VAEs (Single TPU Node, TPU Pods, GPUs)
- [x] Resolve SIGSEGV issue with large TPU Pods [pytorch-xla #3028](https://github.com/pytorch/xla/issues/3028)
- [x] Add DALL-E
- [x] Debug DALL-E (Single TPU Node, TPU Pods, GPUs)
- [x] Add WebDataset support
- [x] Add VAE Image Logger by modifying pl_bolts TensorboardGenerativeModelImageSampler()
- [x] Add DALLE Image Logger by modifying pl_bolts TensorboardGenerativeModelImageSampler()
- [x] Add automatic checkpoint saver and resume for sudden (which happens a lot) TPU restart
- [x] Reimplement EMA VectorQuantizer with nn.Embedding
- [x] Add DALL-E colab notebook by [afiaka87](https://github.com/afiaka87)
- [x] Add Normed Vector Quantizer by [GallagherCommaJack](https://github.com/GallagherCommaJack)
- [x] Resolve SIGSEGV issue with large TPU Pods [pytorch-xla #3068](https://github.com/pytorch/xla/issues/3068)
- [x] Debug WebDataset functionality
## BibTeX
```
@misc{oord2018neural,
      title={Neural Discrete Representation Learning}, 
      author={Aaron van den Oord and Oriol Vinyals and Koray Kavukcuoglu},
      year={2018},
      eprint={1711.00937},
      archivePrefix={arXiv},
      primaryClass={cs.LG}
}
```
```
@misc{razavi2019generating,
      title={Generating Diverse High-Fidelity Images with VQ-VAE-2}, 
      author={Ali Razavi and Aaron van den Oord and Oriol Vinyals},
      year={2019},
      eprint={1906.00446},
      archivePrefix={arXiv},
      primaryClass={cs.LG}
}
```
```
@misc{esser2020taming,
      title={Taming Transformers for High-Resolution Image Synthesis}, 
      author={Patrick Esser and Robin Rombach and Björn Ommer},
      year={2020},
      eprint={2012.09841},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```
```
@misc{ramesh2021zeroshot,
    title   = {Zero-Shot Text-to-Image Generation}, 
    author  = {Aditya Ramesh and Mikhail Pavlov and Gabriel Goh and Scott Gray and Chelsea Voss and Alec Radford and Mark Chen and Ilya Sutskever},
    year    = {2021},
    eprint  = {2102.12092},
    archivePrefix = {arXiv},
    primaryClass = {cs.CV}
}
```



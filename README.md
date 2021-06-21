# Taming Transformers for High-Resolution Image Synthesis, CVPR 2021 (Oral)

Refactoring [Taming Transformers](https://github.com/CompVis/taming-transformers) for TPU VM. 

![teaser](assets/mountain.jpeg)

[**Taming Transformers for High-Resolution Image Synthesis**](https://compvis.github.io/taming-transformers/)<br/>
[Patrick Esser](https://github.com/pesser)\*,
[Robin Rombach](https://github.com/rromb)\*,
[Björn Ommer](https://hci.iwr.uni-heidelberg.de/Staff/bommer)<br/>
\* equal contribution

**tl;dr** We combine the efficiancy of convolutional approaches with the expressivity of transformers by introducing a convolutional VQGAN, which learns a codebook of context-rich visual parts, whose composition is modeled with an autoregressive transformer.

![teaser](assets/teaser.png)
[arXiv](https://arxiv.org/abs/2012.09841) | [BibTeX](#bibtex) | [Project Page](https://compvis.github.io/taming-transformers/)


## Requirements

```
pip install -r requirements.txt
```

## Data Preparation

Place any image dataset with ImageNet-style directory structure (at least 1 subfolder) to fit the dataset into pytorch ImageFolder.

## Training models
You can easily test main.py with random sampled fake data.
```
python main.py --use_tpus --fake_data
```

For actual training provide specific directory for train_dir, val_dir, result_dir
```
python main.py --use_tpus --train_dir [training_set] --val_dir [val_set] --result_dir [where to save results]
```

## BibTeX

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

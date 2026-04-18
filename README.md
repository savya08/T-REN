# T-REN: Learning Text-Aligned Region Tokens Improves Dense Vision-Language Alignment and Scalability

**Authors**: [Savya Khosla](https://savya08.github.io/), [Sethuraman TV](https://github.com/sethuramanio), [Aryan Chadha](https://www.linkedin.com/in/aryan-chadha/), [Alex Schwing](https://www.alexander-schwing.de/), [Derek Hoiem](https://dhoiem.cs.illinois.edu/)

[![ArXiv](https://img.shields.io/badge/arXiv-Paper-b31b1b.svg)]()

T-REN (**T**ext-aligned **R**egion **E**ncoder **N**etwork) is an image encoder that produces region-level tokens aligned with text, built on top of [DINOv3](https://github.com/facebookresearch/dinov3) ViT-L/16 backbone. Compared to its patch-based backbone, T-REN delivers +5.9 mIoU on ADE20K open-vocabulary segmentation, +18.4% recall on COCO object-level text-image retrieval, +15.6% recall on Ego4D video object localization, and +17.6% mIoU on VSPW video scene parsing, all while reducing token counts by more than 24× for images and 187× for videos.

This repository contains training code, a small inference demo ([`tren.py`](tren.py)), and evaluation scripts for several benchmarks: semantic segmentation, video query search, video scene parsing, and Visual Haystacks.

![Python](https://img.shields.io/badge/Python-3.10-blue.svg)
![CUDA](https://img.shields.io/badge/CUDA-12.4-green.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-2.5-orange.svg)


## Getting started

#### 1. Clone the repo and set up the environment.

```
git clone https://github.com/savya08/T-REN.git
cd T-REN
conda env create -f setup.yaml
conda activate tren
```

#### 2. Download T-REN region encoder

Pretrained T-REN `RegionEncoder` weights are hosted on [Hugging Face — savyak2/T-REN](https://huggingface.co/savyak2/T-REN). To download, run:

```bash
./download.sh
```

This creates `logs/tren-ckpts/` and downloads `tren_region_encoder.pth`. To use a different directory:

```bash
./download.sh /path/to/my-ckpts
```

If you use a custom path, set `logging.save_dir` and `logging.exp_name` in your configs so that `save_dir/exp_name/` matches that folder.

#### 3. Download DINOv3 backbone + text head (separate from HF repo)

`model.py` expects the following files next to `tren_region_encoder.pth` (same directory), with exact names:

- `dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth`
- `dinov3_vitl16_dinotxt_vision_head_and_text_encoder-a442d8f5.pth`

They are not on the `savyak2/T-REN` model card. Obtain them from the [DINOv3](https://github.com/facebookresearch/dinov3) release / `torch.hub` workflow and copy them into `logs/tren-ckpts/` (or your chosen checkpoint directory).

#### 4. Update config file(s)

Set dataset paths in the configs before training or evaluation. These paths are currently set to `/path/to/...` placeholders.


| Config File                         | Purpose                                          |
| ----------------------------------- | ------------------------------------------------ |
| `configs/train_dinov3_vitl16.yaml`  | Multi-dataset training paths and hyperparameters |
| `semantic_segmentation/config.yaml` | ADE20K / Cityscapes roots                        |
| `video_query_search/config.yaml`    | VQ2D validation records                          |
| `video_scene_parsing/config.yaml`   | VSPW root                                        |
| `visual_haystacks/config.yaml`      | COCO 2017 and Visual Haystacks roots             |


The checkpoint directory in task configs is set to `../logs/tren-ckpts/`. Update it to point to the custom checkpoint path, if needed.


## Training

```bash
# optional: Weights & Biases for logging (off by default)
export USE_WANDB=1

python train.py
```

Training reads `configs/train_dinov3_vitl16.yaml`, uses `aux_files/cat_to_idx.json` for category indexing, and writes checkpoints under the configured `logging.save_dir` / `logging.exp_name`.


## Inference demo

```bash
python tren.py
```

This downloads a sample image from a public URL, runs `TREN`, and writes visualizations under `region_vis/`.


## Evaluation

Each task lives in its own directory with a `config.yaml` and `eval.py`. After pointing dataset paths, run the corresponding `eval.py` from that directory.


## License

This project is released under the MIT License. See `LICENSE` for details.


## Citing T-REN

```
@misc{khosla2026tren,
      title={T-REN: Learning Text-Aligned Region Tokens Improves Dense Vision-Language Alignment and Scalability}, 
      author={Savya Khosla and Sethuraman T V and Aryan Chadha and Alexander Schwing and Derek Hoiem},
      year={2026},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
}
```


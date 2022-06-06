# Visual Prompting
This is the official implementation of the paper [Exploring Visual Prompts for Adapting Large-Scale Models](https://arxiv.org/abs/2203.17274). 

![](./figures/clip_vs_vision.png)


## Installation
Clone this repo:
```bash
git clone https://github.com/hjbahng/visual_prompting.git
cd visual_prompting
```

This code requires python 3+. Install dependencies by:
```bash
pip install -r requirements.txt
```

Prepare the pre-trained models:
```bash
bash models/download_models.sh
```

## Training
* Training for CLIP:
```bash
python main_clip.py --dataset cifar100 --root [path_to_cifar100] 
```

* Training for vision models:
```bash
python main_vision.py --model bit_m --dataset cifar100 --root [path_to_cifar100]
```
## Testing
* Testing for CLIP:
```bash
python main_clip.py --evaluate --resume /path/to/checkpoints/model_best.pth.tar --dataset cifar100 --root [path_to_cifar100]
```

* Testing for vision models:
```bash
python main_vision.py --evaluate --resume /path/to/checkpoints/model_best.pth.tar --model bit_m --dataset cifar100 --root [path_to_cifar100]
```


## Citation
If you use this code for your research, please cite our paper.
```
@article{bahng2022visual,
         title={Exploring Visual Prompts for Adapting Large-Scale Models}, 
         author={Hyojin Bahng and Ali Jahanian and Swami Sankaranarayanan and Phillip Isola},
         journal={arXiv preprint arXiv:2203.17274},
         year={2022}
}
```

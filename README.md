# Adversarial EfficientSegmentation
## Introduction
- Adversarial EfficientSegmentation is a segmentation method for Medical segmentation, based on [EfficientSegmentation](https://github.com/Shanghai-Aitrox-Technology/EfficientSegmentation) 
- For more information about Adersarial EfficientSegmentation, please read the following paper:
[Efficient Context-Aware Network for Abdominal Multi-organ Segmentation](https://arxiv.org/abs/2109.10601). Please also cite this paper if you are using the method for your research!

## Benchmark
| Task | Architecture | Parameters(MB) | Flops(GB) | DSC | NSC |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
|[FLARE22](https://flare22.grand-challenge.org)| EfficientSegNet | 9 | 333 | 0.800 | 0.837 |
|[FLARE22](https://flare22.grand-challenge.org)| Adversarial EfficientSegNet | 9 | 333 | 0.840 | 0.872 |

## Installation

#### Enviroment
- Ubuntu 20.04.1 LTS
- Python 3.7+
- Pytorch 1.11.0+
- CUDA 11.4

1.Git clone
```
git clone https://github.com/Shanghai-Aitrox-Technology/EfficientSegmentation.git
```

2.Install Nvidia Apex
- Perform the following command:
```
git clone https://github.com/NVIDIA/apex
cd apex
pip install -v --disable-pip-version-check --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" ./
```

3.Install dependencies
```
pip install -r requirements.txt
```

## Get Start
### Preprocesssing
1.  Put your labeled data into Flare/dataset/train_images and Flare/dataset/train_mask and unlabeled data into Flare/dataset/train_unlabel_images
2. Put your data uids list into Flare/dataset/file_list named train_series_uids.txt and train_unlabel_series.txt.
3. (Optional)In case of the dataset is too big to fit in project directory, you can modify FlareSeg/data_prepare/config.yaml to your own data location.
4. After everything set up, you are ready to go
```bash
cd FlareSeg/data_prepare
python run.py
```

### Training
1. coarse model training
```bash
cd FlareSeg/coarse_base_seg
sh run.sh
```
2.  Put you coarsed model's weight into FlareSeg/model_weight/base_coarse_model and run the command below.
```bash
cd FlareSeg/fine_efficient_seg
sh run.sh
```

### Inference
1. Put your coarse model's weight and fine model's weight into FlareSeg/model_weight/base_coarse_model and FlareSeg/model_weight/fine_efficient_seg
2. Run the following code:
```bash
sh predict.sh
```

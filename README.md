# ViRTA Training and Evaluation Commands
## 0. Set Up the Environment

The reference training environment is Linux with an NVIDIA GPU. Use the following versions:

| Component | Version |
|---|---|
| Python | 3.12.3 |
| PyTorch | 2.8.0 |
| torchvision | 0.23.0 |
| PyTorch CUDA build | 12.8 |


Then install the remaining dependencies and check their compatibility:

```bash
python -m pip install -r requirements.txt
python -m pip check
```


## Download the CLIP Weights

Run these commands from the source-code root directory. Download the original `.pt` checkpoint for each language you use:

```bash
mkdir -p pretrained
cd pretrained

# CLIP 
curl -L -C - -O \
"https://openaipublic.azureedge.net/clip/models/5806e77cd80f8b59890b7e101eabd078d9fb84e6937f9e85e4ecb61988df416f/ViT-B-16.pt"

# Chinese-CLIP 
curl -L -C - -O \
"https://clip-cn-beijing.oss-cn-beijing.aliyuncs.com/checkpoints/clip_cn_vit-b-16.pt"

# 下载完成后返回源码根目录，再执行后续命令
cd ..
```

The code loads these files from `pretrained/` automatically. Text positional-embedding interpolation is handled during model loading; no separate interpolation command is required. These are the CLIP backbone checkpoints; train the VPRM separately using the commands below.

## Preprocess the News Data

Run the preprocessing command for your dataset from the source-code root directory.

Weibo:

```bash
python preprocess/weibo_preprocess.py \
  --data-root data/weibo/weibo \
  --output-dir data/weibo/splits --write
```

This writes `train.csv` and `test.csv` to `data/weibo/splits/`.

GossipCop:

```bash
python preprocess/gossip_preprocess.py \
  --data-root data/gossip \
  --output-dir data/gossip/splits --write
```

## 1. Train the VPRM

Put the clean images into `clean-image/`. On the first run, generate the image-split manifest first; Chinese and English share this single manifest:

```bash
python train_vprm.py prepare --language zh \
  --root clean-image \
  --manifest vprm_runs/split.json
```

Train the Chinese VPRM:

```bash
python -u train_vprm.py train --language zh \
  --manifest vprm_runs/split.json --device cuda \
  --output vprm_runs/zh/pretrain
```

Train the English VPRM:

```bash
python -u train_vprm.py train --language en \
  --manifest vprm_runs/split.json --device cuda \
  --output vprm_runs/en/pretrain
```


## 2. Train the Full ViRTA Model
### weibo
```bash
Chinese data (Weibo interface):
python -u train_virta.py \
  --dataset weibo --data-root data/weibo/splits \
  --vprm-checkpoint vprm_runs/zh/pretrain/best.pt \
  --atm-keep-ratio 0.7 --seed 666 \
  --batch-size 4 --accumulation-steps 4 --workers 4 \
  --epochs 50 --patience 3 --freeze-text \
  --output virta_runs/weibo_full/best.pth
```


### GossipCop
```bash
python -u train_virta.py \
  --dataset gossipcop --data-root data/gossip/splits \
  --vprm-checkpoint vprm_runs/en/pretrain/best.pt \
  --atm-keep-ratio 0.1 --seed 666 \
  --batch-size 16 --accumulation-steps 1 --workers 4 \
  --epochs 50 --patience 2 --classification-loss bce --freeze-text \
  --output virta_runs/gossipcop_full/best.pth
```

## 3. Evaluate the Model

### View the Final Ten-Fold Results

Training already performs the per-fold testing automatically; view the summary directly:

```bash
cat virta_runs/weibo_full/best.pth.cv/summary.md
cat virta_runs/gossipcop_full/best.pth.cv/summary.md
```

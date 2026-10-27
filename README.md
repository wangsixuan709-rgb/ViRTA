# ViRTA Training and Evaluation Commands

All commands below are run from the source-code root directory. First set up the runtime environment and prepare the news image-text data, clean images, and the official CLIP weights for the matching language. Place the Chinese weights at `pretrained/clip_cn_vit-b-16.pt` and the English weights at `pretrained/ViT-B-16.pt`; positional-embedding interpolation is handled automatically by the code.

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

After training you get `vprm_runs/zh/pretrain/best.pt` and `vprm_runs/en/pretrain/best.pt`, respectively. If you only run a dataset for one language, train only the VPRM for that language.

## 2. Train the Full ViRTA Model

Chinese data (Weibo interface):

```bash
python -u train_virta.py \
  --dataset weibo --data-root data/weibo/splits \
  --vprm-checkpoint vprm_runs/zh/pretrain/best.pt \
  --atm-keep-ratio 0.7 --seed 666 \
  --batch-size 4 --accumulation-steps 4 --workers 4 \
  --epochs 50 --patience 3 --freeze-text \
  --output virta_runs/weibo_full/best.pth
```

English data (GossipCop interface):

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

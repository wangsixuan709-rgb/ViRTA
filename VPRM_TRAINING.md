# Visual Patch Restoration Module (VPRM) 三阶段训练

2026-10-19 数据采样更新：保留纯干净样本和原整图退化，同时加入平滑局部退化。现有三阶段特征 MSE、评价指标、早停和最佳权重选择规则均保持不变。

本项目实现对应论文第 3.2 节和图 3。恢复模块统一命名为 VPRM，每个模块包含一个前置 NAFBlock、候选恢复分支和 Group-Spatial Gate；组内均值/最大值描述对应 RMX Block。构建接口为 `make_vprm`、`VPRMViTEncoder`，新 checkpoint 使用 `vprm` 字段。

## 接入位置与维度

```text
image                                      [B,3,224,224]
  ↓ patch conv + flatten + CLS + position + ln_pre
ViT tokens                                 [B,197,768]
  ↓ blocks 1–4
VPRM-1：CLS旁路；patch恢复                    [B,197,768]
  ↓ blocks 5–8（输入包含VPRM-1恢复后的patch）
VPRM-2：CLS旁路；patch恢复                    [B,197,768]
  ↓ blocks 9–12（输入包含VPRM-2恢复后的patch）
VPRM-3：CLS旁路；patch恢复                    [B,197,768]
  ↓ ln_post                                [B,197,768]
  ↓ CLIP joint projection 768→512           [B,197,512]
  ↓ ATM（39聚合patch + CLS）                [B,40,512]
  ↓ 与文本refined tokens进入Fine-Grained Adaptive Fusion Module
S、a、b、r_v、r_t → z_align → 融合 → logits   [B,2]
```

三个 VPRM 均执行：

```text
[B,197,768]
   ├─ CLS [B,1,768] ─────────────────────────────┐
   └─ patch [B,196,768]                          │
          ↓ transpose + reshape                 │
      [B,768,14,14]                             │
          ↓ VPRM(768, groups=16)        │
      [B,768,14,14]                             │
          ↓ flatten + transpose                 │
      [B,196,768] ── concat原CLS ────────────────┘
                       ↓
                 [B,197,768]
```

ViT 内部 block 使用 `[197,B,768]`；只在 VPRM 边界转成 batch-first 处理。CLS 不经过卷积，不做伪二维 reshape。第4、8层恢复会通过后续Transformer影响后续CLS；第12层之后没有下一层Transformer，所以 VPRM-3 不直接修改最终CLS，只修改进入ATM的patch。

## VPRM 模块内部结构

恢复分支：X → 前置 NAFBlock → A → 1×1 通道扩展 → GroupNorm → 3×3 分组卷积 → GELU → 组内通道注意力 → 1×1 输出投影 → 与 A 相加得到 R。末尾没有 NAFBlock。

门控分支直接读取 X：分为16组，每组48通道；沿组内通道计算均值和最大值，保留14×14空间位置；拼接32张描述图，经普通1×1跨组卷积、16组3×3深度卷积和Sigmoid，得到 M [B,16,14,14]。

输出：Y = X + broadcast(M) ⊙ (R − X)。权重广播给组内48通道，无需显式 repeat。内部残差使用 A，完整修正量减去原输入 X。三阶段分别拥有一个前置 NAFBlock，全网恢复模块共三个 NAFBlock。

新检查点格式为 `virta-vprm-clip-vit-b16-stages-v1`。加载器同时接受改名前同结构的 `restored-group-spatial-gate-clip-vit-b16-stages-v1`，并转换旧字段。其他结构或语言的权重仍会被拒绝；不使用 `strict=False` 忽略不匹配。

## 现有数据与训练逻辑

干净图像目录默认为 `clean-image/`。项目不再提供服务器固定划分，首次运行必须从当前图片生成清单：

```bash
python train_vprm.py prepare --language zh --root clean-image --manifest vprm_runs/split.json
```

程序检查图片、按解码后的像素哈希去重，使用 seed=42 进行约 80%/10%/10% 划分。实际数量由使用者的图片集合决定，不限制为原服务器数量。中英文共用生成的图片清单，但分别训练恢复参数。

每次训练采样：

1. 读取一张clean RGB图像，必要时放大到短边至少224，随机裁剪224×224，以50%概率水平翻转。这是HQ。
2. 在14种退化和clean共15种情况中均匀采样，clean概率为1/15。
3. 若为clean：LQ直接复制HQ，用于学习不要破坏干净特征。
4. 非干净样本中，默认约50%沿用原整图退化：先将HQ缩小到随机56～223像素，再施加一种退化，最后恢复224×224。严重度1～5的概率依次为5%、25%、40%、25%、5%。
5. 其余约50%生成平滑局部退化：同一HQ分别生成同类型的轻度版本（严重度1～2，缩小至168～223）和重度版本（严重度4～5，缩小至56～111），再与HQ按随机方向、带平滑弯曲的空间权重混合。每张局部样本均有连续的纯干净区、纯轻度区、纯重度区和柔和过渡带；纯干净区的像素与HQ完全一致。
6. HQ与LQ使用同一套CLIP均值和标准差归一化。轻重区域是两种退化结果的空间混合，不是物理退化参数的逐像素精确估计。局部样本原有 `severity` 字段表示最高严重度（4或5），数据元信息 `degradation_mode` 区分clean/global/local；没有新增评价指标。

比例由 `--local-degradation-probability` 控制，默认0.5；该比例仅作用于非干净样本，所以纯干净图像仍占1/15，整图与局部退化各约7/15。训练采样由图像、epoch及seed决定；验证/测试在各退化类型内确定性分配局部样本，保持原清晰参考及类型配额。

比例保存在现有训练配置中；改变比例需要使用新训练输出目录。独立测试恢复checkpoint中的比例，旧checkpoint缺少此字段时按0处理，以复现旧整图退化。manifest中的相对图像根目录按manifest所在目录解析。

14种退化：高斯噪声、散粒噪声、脉冲噪声、离焦模糊、玻璃模糊、运动模糊、缩放模糊、雪、霜、雾、亮度、对比度、像素化、JPEG压缩。

这里并非只施加单一噪声：非clean样本同时包含先缩小再放大的分辨率损失。HQ和LQ来自同一个裁剪，保持空间对应。验证/测试使用固定几何预处理和受seed控制的退化配额；同一round可重复，round改变后退化样本会改变。

## 监督目标与梯度

```text
HQ → 同语言冻结CLIP、不启用VPRM → 第4/8/12层clean patch目标
                                              ↓ detach / no_grad
LQ → 同语言冻结CLIP、启用三个VPRM → 第4/8/12层恢复patch
                                              ↓
                         三阶段特征MSE的平均 → 只更新VPRM
```

每阶段监督特征都是 `[B,768,14,14]`，取在 `ln_post` 和 joint projection 之前：

`loss = (MSE(restored_4, clean_4) + MSE(restored_8, clean_8) + MSE(restored_12, clean_12)) / 3`

这次把原五阶段ResNet损失改为三阶段ViT损失，禁止形状不一致时隐式广播。保留原始特征MSE，不额外做特征归一化，因此各阶段数值幅度会影响其实际损失贡献；当前没有新增CLS损失、图像重建损失或真假分类损失。

CLIP骨干参数被冻结并保持eval；student分支不能整体放在no_grad中，后续冻结的Transformer仍需把梯度传回前面的VPRM。代码使用梯度检查点控制激活内存，已验证最后阶段损失单独就能传回三个VPRM。

## 优化、验证与模型选择

保持原调度与训练策略：

- AdamW，只更新VPRM；weight_decay=0.01，梯度裁剪=1。
- 默认batch_size=1，accumulation=32；不足32个样本的最后累积窗口按真实样本数归一化。
- 默认最多20000次更新；前1000次学习率从1e-6预热到1e-4，然后余弦下降到1e-6，超出计划进度不会回升。
- 每轮在验证集比较：`LQ无VPRM vs HQ`、`LQ有VPRM vs HQ` 的最后阶段patch余弦相似度，记录baseline/restored/gain。
- 选择best时，候选与原best用同一组3个验证round做配对比较；默认平均restored提升超过0.001才替换。
- EarlyStop从12000步后按5轮滑动平均监控，4000步没有达到0.001的进展则停止。
- 测试集只由单独test命令使用；默认5个测试round。这里测的是特征恢复效果，虚假新闻分类效果仍需后续整体模型训练/测评。

## 分别训练与测试

以下命令已经准备好，**本次没有启动正式20000步训练**。

```bash
# 英文：OpenAI CLIP ViT-B/16
python -u train_vprm.py train --language en \
  --output vprm_runs/en/pretrain

# 中文：Chinese-CLIP ViT-B/16
python -u train_vprm.py train --language zh \
  --output vprm_runs/zh/pretrain

# VPRM独立测试
python -u train_vprm.py test --language en \
  --checkpoint vprm_runs/en/pretrain/best.pt --output vprm_runs/en/test
python -u train_vprm.py test --language zh \
  --checkpoint vprm_runs/zh/pretrain/best.pt --output vprm_runs/zh/test
```

首次运行执行上述 prepare；已有同一套图片生成的 `vprm_runs/split.json` 时复用该清单。中断后可以在原输出目录使用 `--resume .../latest.pt`，恢复优化器、学习率调度、随机状态和训练进度。

## 接入整体分类模型

训练得到对应语言的VPRM后：

```bash
python train_virta.py --dataset gossipcop \
  --vprm-checkpoint vprm_runs/en/pretrain/best.pt \
  --batch-size 4 --accumulation-steps 4 --output virta_runs/gossipcop_vprm/best.pth

python train_virta.py --dataset weibo \
  --vprm-checkpoint vprm_runs/zh/pretrain/best.pt \
  --batch-size 4 --accumulation-steps 4 --output virta_runs/weibo_vprm/best.pth
```

独立VPRM预训练入口默认就带第4/8/12层三个模块。完整分类模型必须通过 `--vprm-checkpoint` 加载匹配的VPRM，默认冻结视觉骨干和已训练VPRM；文本与视觉编码器默认均冻结。此模式禁止同时指定 `--train-backbones`。

整体分类checkpoint包含VPRM；`evaluate_virta_errors.py` 使用完整模型结构，不需要再单独提供原VPRM路径。独立checkpoint包含格式版本、language、stages=(4,8,12)、骨干权重、VPRM权重及划分哈希；中英文错配和不匹配的旧结构/ResNet权重会被拒绝。

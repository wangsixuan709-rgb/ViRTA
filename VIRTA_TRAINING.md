# ViRTA 完整模型训练流程

完整模型始终包含第 4/8/12 层视觉 VPRM、文本和图像 ATM、Fine-Grained Adaptive Fusion Module 以及最终分类器。分类训练必须提供匹配语言的预训练 `--vprm-checkpoint`。

图像经 CLIP ViT-B/16 和 VPRM 后得到视觉 token；中文采用 Chinese-CLIP 的 256 token 输入，英文按 FineLIP 的 248 token 重叠片段覆盖全文。图文投影至 512 维，经 ATM 聚合后计算 Fine-Grained Cross-Modal Matching 相似度矩阵。相似度矩阵池化为 1600 维交互特征，与各 48 维的文本、图像分支特征分别乘以学习到的 Softmax 权重，拼接为 1696 维送入最终分类器。

训练、验证、训练结束后的测试及独立测试均使用同一模型前向函数。默认 token 保留率为 0.5，英文内部片段批次为 4；保留率可通过分类训练的 `--atm-keep-ratio` 指定，保存权重后由独立测试自动读取。

Weibo 默认 CE，训练标签平滑 0.05，辅助分支系数 0.2、融合均衡系数 0.02。Gossip 默认仅 BCE，标签平滑和两项附加损失系数均为 0。编码器和 VPRM 默认冻结。

分类训练默认最多 50 轮，Weibo patience=3，Gossip patience=2；按独立验证集 Macro F1 与主分类损失选优，训练结束后恢复选定权重，再评价完整测试集。VPRM 预训练保持独立的数据准备、训练、早停和测试入口。

默认数据目录为 `data/weibo/splits`、`data/gossip/splits`；本地旧数据目录也可通过 `--data-root` 或对应环境变量显式指定。标签为 0=Fake、1=Real。

环境准备、数据处理及训练测试命令见 [README](README.md)，恢复模块训练细节见 [VPRM_TRAINING](VPRM_TRAINING.md)。

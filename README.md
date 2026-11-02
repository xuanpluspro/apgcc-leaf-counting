# APGCC Leaf Counting

面向 RiceW3999（水稻/小麦混合来源）的叶尖点定位与计数。
基于 [APGCC, ECCV 2024](https://github.com/AaronCIH/APGCC) 公开代码和论文补全训练流程。
这是第三方复现与植物数据适配版，不是作者官方训练实现，也未声称复现论文精度。
出处、源码修复及未公开细节的实现选择见 [REPRODUCTION.md](REPRODUCTION.md)。

## 已提供

- VGG16-BN + ASPP + 隐式特征插值 IFI + 坐标偏移/分类头，直接输出点坐标，不生成高斯密度图。
- 匈牙利一对一匹配，以及辅助正/负点监督（默认开启，非仅基础分支）。
- 512/1024 混合原图；训练每次随机生成一个 512×512 patch，验证原图、batch=1。
- 支持文件名中的空格、中文、UTF-8 BOM，以及零点图片；训练/验证 ID 交叉检查。
- 训练/验证进度条，逐轮日志、MAE/RMSE、P/R/F1@4/8/16px、逐图和分来源统计。
- 每个完整 epoch 自动原子保存；同 TAG 默认完整续训，最佳模型按验证 MAE 选择。
- 整图或重叠分块推理、点 TXT/JSON/可视化、独立参数量/延迟/FPS/显存/可选计算量评测。

## GPUFree 快速开始

镜像应已有 PyTorch >=2.3；以下依赖安装不会重新安装 torch/torchvision。

```bash
cd /root/gpufree-data
git clone https://github.com/xuanpluspro/apgcc-leaf-counting.git
cd apgcc-leaf-counting

uv pip install --python "$(which python)" -r requirements.txt \
  --index-url https://mirror.sjtu.edu.cn/pypi/web/simple

cp /root/gpufree-share/vgg16_bn-6c64b313.pth pretrained/
```

VGG 权重就是 PET/P2RLoss 共用的 ImageNet `vgg16_bn-6c64b313.pth`；完整训练权重不能互换。
不联网自动下载；首次训练缺少本地权重会报错。续训、评估、推理无需骨干初始化文件。

可直接复用 P2RLoss 已解压的数据，不用移动或复制数据集：

```bash
python check_data.py --data-path /root/gpufree-data/p2rloss/data/RiceW3999

DATA_PATH=/root/gpufree-data/p2rloss/data/RiceW3999 \
BATCH_SIZE=32 EPOCHS=50 WORKERS=4 \
bash train_ricew3999.sh ricew3999_ep50_bs32_run1
```

若已将压缩包放入本仓库 data/ 并解压为 data/RiceW3999，省略 DATA_PATH 即可。
启动脚本的最后一个参数是实验名；名称中的 ep50/bs8 只是文字，实际参数由环境变量决定。
默认 batch=32，作为48GB显存下的起始尝试；尚未在GPU实测，不能保证峰值显存。若OOM，改用 BATCH_SIZE=16。大batch不保证更高精度，学习率保持不变。
可设置 `AMP=1` 开启 FP16 混合精度；默认 FP32，便于与现有实验比较。

建议在 tmux 中运行上述训练，另开终端执行 nvitop。tmux 不能防止关机，恢复依赖检查点。

## 数据格式

```text
data/RiceW3999/
  JPEGImages/<完整图片ID>.jpg 或 .png
  PointAnnotations/<完整图片ID>.txt
  ImageSets/train.txt
  ImageSets/val.txt
  ImageSets/test.txt   # 可选
```

清单一行一个完整 ID（不带扩展名），不按空格拆分。
标注 TXT 一行两个原图像素坐标 `x y`，不使用 YOLO 归一化坐标；空文件表示零点。
图片尺寸、越界点、重复 ID、缺失配对会明确报错。
彻底检查跨划分的相同图像可加 `check_data.py --check-overlap`，它读取全部像素，较慢。
若不同 patch 来自同一原图/植株，应在划分时按原图/植株分组；像素去重不能检测所有重叠裁剪泄漏。

## 训练参数

| 项目 | 默认值 |
|---|---|
| batch / 总轮数 / workers | 32 / 50 / 4 |
| 训练 patch / 验证 | 512×512 / 原图 batch=1 |
| 优化器 / 权重衰减 | Adam / 1e-4 |
| 主体 LR / VGG LR | 1e-4 / 1e-5 |
| 学习率衰减 | StepLR，第300轮后乘0.1；默认50轮内不衰减 |
| 辅助正点、负点数量 | 每个标注点各2个 |
| 辅助采样范围 | 正点坐标扰动[-2,2]，负点各轴扰动±[2,8]像素 |
| 点回归权重 / 辅助总权重 | 0.0002 / 0.2 |
| 置信度阈值 / 验证间隔 | >0.5 / 每个完整epoch（默认1），最后一轮必验证 |
| 精度 / seed | FP32 / 2024 |

这是叶尖适配配置。原论文训练使用128 patch、更长训练，不保证50轮已收敛；应看验证曲线。
`--no-aux` 仅用于消融，默认不要添加。`--accum-steps` 支持梯度累积；它不会降低单个 microbatch 的峰值显存。
查看完整参数：`python train.py --help`。

## 输出和恢复

主要目录：`exp/<实验名>/output/`。

| 文件 | 含义 |
|---|---|
| checkpoint.pth | 每轮最新完整训练状态，用于恢复 |
| best_checkpoint.pth | 验证MAE最低的完整检查点，用于推理/评估 |
| config.json / resume_config.json | 首次启动 / 最近恢复时的配置及环境 |
| train.log | 追加式日志 |
| train_history.csv | 每轮loss分量、LR、训练耗时和峰值已分配显存 |
| metrics_history.csv | 各次验证的计数、定位、参数量、验证耗时 |
| best_metrics.json | 最佳模型对应的整体指标 |
| evaluation/epoch_XXXX/metrics.json | 该轮整体指标和评价协议 |
| evaluation/epoch_XXXX/per_image.csv | 逐图数量、误差、TP/FP/FN |
| evaluation/epoch_XXXX/per_source.csv | 按 ricetip/wheat500/wheat1799 等前缀汇总 |

脚本终端日志另存 `exp/<实验名>/train.log/running.log`，恢复时追加。
没有教师模型；每个 pth 包含一套 model 权重及优化器、调度器、AMP scaler、轮数、最佳指标和随机状态。
复用同一 TAG 执行原命令即可自动恢复，或 `--resume /path/to/checkpoint.pth`。
保持原 EPOCHS（总预算）和模型/学习率/损失配置；因OOM调小 batch 会警告轨迹变化。
原子写入保护旧文件；中断后重跑当前未保存的整轮，不是按 batch 恢复。实例删除/磁盘清空不在保护范围内。
恢复会清理晚于检查点的CSV行，逐轮评估目录在重跑时覆盖。GPU非确定性仍可能造成差异。

## 独立评估旧实验的最佳模型

```bash
python evaluate.py \
  --checkpoint exp/ricew3999_ep50_bs32_run1/output/best_checkpoint.pth \
  --data-path /root/gpufree-data/p2rloss/data/RiceW3999 \
  --output evaluation/best
```

默认评估 val.txt，不冒充独立测试集；已有 test.txt 可加 `--split test`。
P/R/F1 是整套验证集的micro统计：先汇总TP/FP/FN，再计算比值。
4/8/16px只用于评价，不参与训练中的匹配阈值；所有模型对比时固定原图尺度。
APGCC内部使用xy坐标，评价前明确转换为与PET/P2RLoss一致的yx欧氏距离协议。

## 黑麦草推理

```bash
python infer.py \
  --checkpoint exp/ricew3999_ep50_bs32_run1/output/best_checkpoint.pth \
  --input /path/to/ryegrass_images \
  --output predictions/ryegrass
```

默认整图，不缩放；3072等大图显存不足时加 `--tile-size 1024 --overlap 64`。
分块使用重叠区中线划分归属范围，避免同一位置重复输出；不做固定半径NMS，避免合并相邻真实叶尖。
分块与整图结果可能不同，边界预测仍需目视核查。输出 xy TXT、含置信度JSON、叠加图和 counts.csv。

## 独立效率测试

```bash
python benchmark_model.py \
  --checkpoint exp/ricew3999_ep50_bs32_run1/output/best_checkpoint.pth \
  --size 512 --device cuda \
  --output exp/ricew3999_ep50_bs32_run1/output/benchmark.json
```

默认batch1、FP32、20次预热/100次测量；测前向，不包含读图、传输、定位匹配和可视化。
参数量、平均/中位延迟、FPS、峰值已分配/预留显存保存在JSON。
可选安装 `fvcore` 后加 `--flops`；存在不支持的运算时仅报告 counted_gflops，不冒充完整GFLOPs。

## 测试与验证范围

```bash
python -m unittest discover -s tests -v
```

覆盖512/1024前向、默认模型512前后向另行检查、IFI坐标一致性、辅助梯度、空图、数据配对、
一对一评价、补边过滤、切块归属、原子写入失败及实际训练循环的中断恢复一致性。
CPU测试仅说明工程链路；未在真实 RiceW3999 或 CUDA/AMP 上完成训练，暂无叶尖精度/速度承诺。

## 来源与引用

原项目：[AaronCIH/APGCC](https://github.com/AaronCIH/APGCC)，MIT License。
保留原作者版权及相关注释，详细来源见 [THIRD_PARTY.md](THIRD_PARTY.md)。

```bibtex
@inproceedings{chen2024apgcc,
  title={Improving Point-based Crowd Counting and Localization Based on Auxiliary Point Guidance},
  author={Chen, I-Hsiang and Chen, Wei-Ting and Liu, Yu-Wei and Yang, Ming-Hsuan and Kuo, Sy-Yen},
  booktitle={European Conference on Computer Vision},
  year={2024}
}
```


验证默认每轮执行，`VAL_EVERY`／`--eval-every` 可覆盖。旧检查点允许恢复时切换验证间隔，并记录日志；不会补评估历史未验证轮次。切换频率可能改变随机数消耗和后续训练轨迹。

# 验证记录

日期：2026-10-30。环境：Python 3.12，PyTorch 2.15.0+cpu，无CUDA设备。

- `python -m unittest discover -s tests -v`：19项通过。
- 含512/1024输入前向检查，以及来自512/1024原图的512训练patch混合batch。
- 使用默认17,750,480参数网络，512×512输入、100个合成标注点，完成前向、全部损失和反向传播，损失有限。
- 辅助监督单独反向可产生骨干、分类头和回归头的非零梯度；全空标注batch也能反向。
- 任意IFI查询与规则网格IFI路径一致；辅助查询位于常规参考点时预测一致。
- 使用小头配置（保持VGG骨干和训练流程）在合成图片上连续训练2轮，与第1轮保存、中断、恢复后的2轮结果逐参数完全一致；评估/训练CSV无重复行。
- 原子写入模拟磁盘错误，旧检查点字节不变，临时文件清理成功。
- `evaluate.py` 在合成数据上执行，生成整体指标、逐图和分来源CSV。
- `infer.py --tile-size 32 --overlap 8 --device cpu` 在合成图片上执行，生成TXT、JSON、叠加图和counts.csv。
- `benchmark_model.py --size 32 --iterations 2 --warmup 0 --device cpu` 完成测速JSON导出；仅为脚本检查，不能当作真实模型性能报告。
- 所有Python源码语法检查及训练脚本 `bash -n` 通过。

尚未完成：真实RiceW3999训练、ImageNet权重文件实物载入、CUDA/AMP前后向、GPU显存/速度测试、原论文人群数据复现。
VGG载入路径使用标准torchvision features字段严格匹配；本次未下载或分发大权重文件。
因此上述结果证明的是本实现的工程正确性检查，不是叶尖精度或超过PET的证据。

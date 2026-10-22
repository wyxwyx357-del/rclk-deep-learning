# CUDA环境与训练程序试跑

- 日期：2026-10-22
- 状态：通过；仅程序试跑，正式交叉验证训练尚未启动
- 环境：仓库独立`.venv`，Python 3.14，PyTorch 2.12.0+cu130，scikit-learn 1.9.1
- 显卡：NVIDIA GeForce RTX 5060 Laptop GPU；CUDA可用；显存总量约8 GB
- 目的：确认真实训练输入能在GPU上完成一次前向、反向和参数更新
- 数据范围：固定第0次第0外层折的16位训练患者、18个连续段；外层测试病例逐帧数组未读取
- 结果：损失和梯度均有限；一次参数更新成功；GPU显存峰值约154 MiB；脚本内部耗时约6.32秒
- 本机命令：`.venv\Scripts\python.exe scripts\smoke_train_fold.py --device cuda --out outputs\tcn_fold_smoke_cuda_20261022.json`
- 详细输出：`outputs/tcn_fold_smoke_cuda_20261022.json`，被Git忽略
- 限制：这是一批数据的一次更新，不代表完整训练速度或妊娠预测性能。临床来源核查、完整嵌套交叉验证与对照模型仍未完成。

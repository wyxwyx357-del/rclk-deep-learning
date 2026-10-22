# R/C/L/K逐帧预测代码审查说明

本仓库只包含代码、冻结协议快照和不含患者身份的汇总运行记录。正式逐帧数组、临床表、核查状态、逐患者预测和模型权重均未纳入Git。

## 研究问题与口径

在295例患者中比较M3（临床＋R/C/L聚合量＋逐帧小型TCN）相对M1（临床＋R/C/L聚合量）、M2（再加固定简单时空统计）和MQ（再加质量/运动代理量）的妊娠预测增量。K只在有K的同一288例中做配对次要比较。3次重复、5外折、每外折5内折；TCN仅测试宽度8/16与权重衰减0.001/0.01四组。模型选择、早停、固定种子和成对bootstrap以[`protocol/PROTOCOL_中文.md`](protocol/PROTOCOL_中文.md)为准。

## 建议重点审查

1. [`src/rclk_dl/inputs.py`](src/rclk_dl/inputs.py)的正式数组、拓扑风险屏蔽、35 Hz来源帧与空间插值是否保持数据口径；[`batches.py`](src/rclk_dl/batches.py)是否只在R/C/L三家族均有有效点的连续段内卷积。
2. [`training.py`](src/rclk_dl/training.py)与[`baselines.py`](src/rclk_dl/baselines.py)的所有变换、插补、参数选择和早停是否只使用相应训练折；外层测试结局是否只在最终评价时使用。
3. K对照是否在同一288例、同一分折上，两个网络均有F15静态输入，仅RCLK增加K逐帧通道；机制对照是否重新训练，而非只改测试输入。
4. [`features.py`](src/rclk_dl/features.py)的30项M2统计与12项MQ质量变量是否符合协议，尤其是缺失和相关性可计算门槛；内部平移和旋转只能作运动代理，不能直接标为探头抖动。
5. [`evaluation.py`](src/rclk_dl/evaluation.py)是否按患者、按重复计算折外AUC并共享按结局分层的患者配对bootstrap；校准图是否只作描述。
6. [`study.py`](src/rclk_dl/study.py)的35例来源核查门槛、固定分折、14例拓扑掩码来源和输入哈希是否能防止错接或静默改版。

## 已核验与仍待核验

- 295例输入验收、拓扑1171风险帧、K可用288例见[`protocol/INPUT_ACCEPTANCE_中文.md`](protocol/INPUT_ACCEPTANCE_中文.md)。
- 当前10项单元/边界测试通过；主RCL和次要RCLK分别在真实训练折子集上完成一轮GPU拟合，产生有限内层验证概率；输出不作为预测性能报告。
- 临床来源35例尚未完成逐例核查，正式全队列嵌套交叉验证尚未运行；因此没有正式AUC或模型优劣结论。代码审查应明确区分已测试的程序路径与尚未执行的完整实验。

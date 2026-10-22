# 逐帧输入验收

- 日期：2026-10-22
- 状态：通过；仅验收输入读取，未训练模型
- 目的：核对正式R/C/L/K逐帧数据的读取、35 Hz时间映射和拓扑风险帧屏蔽
- 代码提交：`44fd9d9`（初始仓库版本）
- 输入口径：`protocol/model_plan.json`；正式追踪和QC只读
- 295例名单SHA-256：`8C5BBDB56EDF185B47DFECB5C2C8B2DEA2BA4F40118E5A5F1261E7928F183204`
- 正式F汇总SHA-256：`4E506D5729D0F1D75E7D3AE2E9BBAB97A22A3A3247FBDE44DC0FF29DDD014EE9`
- 样本：295例R/C/L均可读；288例K可读
- 汇总结果：14例共1171个拓扑风险帧按正式规则屏蔽；35 Hz目标时间点插值比例中位数85.69%；2项边界单元测试通过
- 验收命令：`python scripts/accept_inputs.py --root <正式追踪输出目录> --cohort <295例名单> --formal <正式F汇总>`；`python -m unittest discover -s tests -v`
- 详细输出：本机`outputs/input_acceptance.csv`（患者级，不提交Git）
- 限制：该结果只证明输入读取符合当前验收规则；临床来源核查、GPU程序验收、TCN训练及预测增益均尚未完成。插值点不构成新增独立观测。

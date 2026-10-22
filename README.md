# R/C/L/K 逐帧深度学习实验仓库

这是独立的本地Git仓库，用于295例患者的R/C/L逐帧妊娠预测探索，以及同一288例中K的次要比较。现有正式追踪和QC为只读输入；所有患者级数据、视频、模型权重与实验输出留在Git之外。

## 当前状态

- 已纳入[冻结建模协议](protocol/PROTOCOL_中文.md)与[原输入验收记录](protocol/INPUT_ACCEPTANCE_中文.md)的快照。
- `src/rclk_dl/inputs.py`：按正式规则读取R/C/L/K、屏蔽拓扑风险帧，保留35 Hz网格的来源帧、插值权重和有效掩码。
- `scripts/accept_inputs.py`：重新核对295例正式输入，结果写入忽略的`outputs/`目录。
- `src/rclk_dl/batches.py`和`src/rclk_dl/tcn.py`：已实现R/C/L连续段装配和预定的小型因果TCN；边界测试通过。
- `scripts/smoke_train_fold.py`：已接入冻结外层训练折、训练折内静态变量处理和一次真实数据参数更新，供手动程序试跑。
- `scripts/run_experiment.py`：已实现冻结分折的M1/M2/MQ/M3嵌套交叉验证、288例K比较、临床来源敏感性分析、触发后的三项机制对照及患者配对评价。正式全队列训练**尚未启动**；代码处于审查阶段。

## 使用

本机仓库的`.venv`已安装Python 3.14、仓库依赖和PyTorch 2.12.0+cu130，并已识别RTX 5060。新机器上可按下列方式建立独立环境，GPU版PyTorch以[官方安装页](https://pytorch.org/get-started/locally/)的当前命令为准：

```powershell
python -m venv .venv
.venv\Scripts\python -m pip install torch==2.12.0 --index-url https://download.pytorch.org/whl/cu130
.venv\Scripts\python -m pip install -e .
```

系统全局Python仍是CPU版PyTorch；运行GPU命令时必须明确使用`.venv\Scripts\python.exe`。本机GPU单批程序试跑已通过，详见[记录](runs/2026-10-22_gpu_program_smoke.md)。完整训练的耗时仍需在正式代码完成后测量。

输入验收命令（路径按本机实际位置填写；输出不提交Git）：

```powershell
python scripts/accept_inputs.py --root "D:\uterus_2.0\最终追踪特征输出" --cohort "<冻结295例输入CSV>" --formal "<最终正式主分析_F01-F20.csv>"
```

运行边界测试：

```powershell
.venv\Scripts\python.exe -m unittest discover -s tests -v
```

手动运行**一次训练程序试跑**（默认读取本机已核对的295例输入，输出只在本机且不会覆盖已有文件）：

```powershell
.venv\Scripts\python.exe scripts/check_device.py
.venv\Scripts\python.exe scripts/smoke_train_fold.py --device cuda --out outputs/tcn_smoke_manual_gpu.json
```

重复运行时换一个新的`--out`文件名；如果CUDA不可用，命令会明确报错。该脚本固定使用第0次第0外层折的前16位训练患者，仅做一次参数更新，不读取外层测试病例，也不报告AUC。输出摘要没有患者身份；不能作为正式训练结果。

## 正式实验入口（待临床核查后运行）

先生成被Git忽略的35例核查表：

```powershell
.venv\Scripts\python.exe scripts/run_experiment.py --analysis prepare-review
```

逐例对照原临床来源，在`outputs/prepare-review/clinical_review_35.csv`填写`status`与`evidence_note`。`status`只能是`confirmed`（同患者同次移植已核实）、`unresolved`（尚不能核实，但未证实错配）或`confirmed_mismatch`（已证实错配）。发现已证实错配时，先按来源证据修正病例或标签并重新冻结名单与分折；程序会拒绝在旧版本上训练。保留未解决疑点进行295例主分析，并用来源已确认子集重新拟合全部比较组做敏感性分析。不要把核查表提交Git。

核查完成后依次运行；每项全队列任务可能包含数百次模型拟合，输出目录中的哈希清单和分折结果允许在代码及输入未变时续跑：

```powershell
.venv\Scripts\python.exe scripts/run_experiment.py --analysis primary --review outputs/prepare-review/clinical_review_35.csv
.venv\Scripts\python.exe scripts/run_experiment.py --analysis sensitivity --review outputs/prepare-review/clinical_review_35.csv
.venv\Scripts\python.exe scripts/run_experiment.py --analysis k --review outputs/prepare-review/clinical_review_35.csv
.venv\Scripts\python.exe scripts/run_experiment.py --analysis mechanism --review outputs/prepare-review/clinical_review_35.csv
```

只有存在`unresolved`记录时才运行`sensitivity`；只有主分析达到协议门槛时才运行`mechanism`。程序会明确拒绝不适用的任务。输出中的逐患者折外预测、逐文件哈希、校准图和训练详情都留在被Git忽略的`outputs/`。冻结协议原文仍保留实施前状态；最新实现与未完成事项见[审查说明](REVIEW_FOR_GPT.md)。

## 实验边界

所有分折以患者ID而非帧划分；只在训练折拟合标准化、临床插补和参数。主比较是临床＋RCL聚合基线与增加RCL逐帧小型TCN；质量、时长和插值比例另作对照。13例旧ER分析窗与拓扑风险帧重叠，因此旧ER结果只作历史探索记录，不作为本仓库的新序列输入。

临床来源核查与正式训练都尚未完成。35例姓名/移植日期来源线索要逐例标记；未证实错配者不按模型表现删病例。

本仓库源自`频率计算/explore_outputs/sequence_model_readiness_20261022`的只读审查和输入读取器。协议快照保存在`protocol/`，后续实验修改须在本仓库单独记录，不回写正式追踪产物。

## 本地测试记录

每次验收或训练后，在[`runs/`](runs/README.md)增加一份仅含汇总结果的Markdown记录，并提交到本地Git仓库。首份[输入验收记录](runs/2026-10-22_input_acceptance.md)已经保存。患者级验收表、逐例预测、模型权重及日志仍放在被Git忽略的`outputs/`等目录。`git commit`只更新本机的版本历史；本仓库目前没有远端，也无需上传云端即可持续记录测试。

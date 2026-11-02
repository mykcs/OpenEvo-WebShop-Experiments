# OpenEvo × WebShop Experiments

本仓库公开 OpenEvo 在 WebShop 购物任务上的实验代码、协议和可复算的脱敏结果。研究问题是：智能体如何从任务经验中持续学习，以及能否在限制历史参数容量的同时保留学习效果。

## WebShop 与 OpenEvo 的关系

**这些代码用于 OpenEvo 的实验，其中有直接调用 OpenEvo Core 的训练和运行时路径。** WebShop 提供任务环境与评分；本仓库实现实验组织、经验转训练数据、OpenEvo 参数更新、模型重载及配对评测。本仓库不包含 OpenEvo Core 源码，也不意味着每一个历史对照分支都经过 Core。

可从下列实际调用核对，而不必依据仓库名称判断：

| 环节 | 代码 | 调用或职责 |
| --- | --- | --- |
| WebShop 适配 | [`vendor/openevo-webshop`](vendor/openevo-webshop) | OpenEvo harness / evaluator 接口与 WebShop 环境适配 |
| 数据转换 | [`parametric_dataset_adapter.py`](scripts/openevo_webshop/parametric_dataset_adapter.py) | 将任务交互转换为参数训练数据 |
| 原生训练 | [`run_parametric_smoke.py`](scripts/openevo_webshop/run_parametric_smoke.py) | 调用 `openevo.evolution.parametric.sd_lora.parametric_memory_sd_lora` 和 `SubprocessSdLoraTrainerService` |
| 长跑更新 | [`ceiling1_stage2_sd_lora.py`](scripts/openevo_webshop/ceiling1_stage2_sd_lora.py) | 使用 OpenEvo 训练接口与累积参数状态 |
| 运行时上下文 | [`ceiling1_runtime.py`](scripts/openevo_webshop/ceiling1_runtime.py) | 调用 `openevo.evolution.runtime_injection.instruction_with_evolution_context` |
| 更新后评测 | [`run_parametric_rollout.py`](scripts/openevo_webshop/run_parametric_rollout.py) | 加载基础模型或演化后的适配器，执行 WebShop rollout |

普通 LoRA、SEED 复现及诊断分支是比较对象，不能一概称为同一套 OpenEvo 运行路径。代码存在也不自动证明某次运行已完成，科学结论须以对应结果为准。

## 实验代码范围

- **H 系列**：任务与动作契约、探索与经验筛选、训练标签、SD-LoRA、普通 LoRA、迁移与多轮实验。协议见 [`configs/experiment`](configs/experiment)，实现见 [`scripts/openevo_webshop`](scripts/openevo_webshop)。
- **固定容量与长跑**：`ceiling1_*`、`bounded_*`、`post_advisor_*` 等训练、状态、容量及方向干预代码；对应设计位于 [`configs/experiment/designs`](configs/experiment/designs)。
- **后续问题与对照**：Q 系列的新题评测、容量、预测、保留与干预分析。见 [`scripts/analysis`](scripts/analysis)、[`configs/experiment/advisor-questions`](configs/experiment/advisor-questions) 和 [`analysis`](analysis)。文件名中的 `advisor` 是历史实验编号，不附带私人沟通材料。
- **测试与环境参考**：[`tests`](tests)、适配包自带测试、`requirements-*.txt` 和 [`schemas`](schemas)。保留历史回归测试用于理解契约；依赖未公开原始证据或基础设施的测试不属于本公开版的可运行验收集合。

完整协议文件清单见 [EXPERIMENT_INDEX.md](EXPERIMENT_INDEX.md)。公开来源和逐文件变换记录见 [PUBLICATION_MANIFEST.json](PUBLICATION_MANIFEST.json)。

## 已公开的数值结果

[`results`](results) 保留经过筛选的数字与分析脚本，不包含任务正文、原始轨迹或模型权重。

- **同起点 R119 与末期 R159 的冻结新题比较**：64 道配对任务，均分约 58.42 与 71.87，差 +13.44 分，配对 95% 区间约 [+4.81, +22.54]。模型对是在看到开发题改善后选定的；这支持所选保存状态间的新题差异，不能证明固定容量方法优于普通方法，也不是独立训练复现。
- **Q10 第 12 轮开发试验**：简化配方约 33.90 分，普通参照约 45.22 分。结果不支持早期优势，不能当作完整长跑结论。
- **Q04 / Q06**：聚合文件保留原始分析边界。容量比较尚无明确非劣结论；原预测假设族未获确认。不得将封存状态当作科学成功。

复算冻结新题结果（仅标准库，不调用模型）：

```bash
python3 results/fresh-pair-results-20261028/analyze_q09_fresh_pair.py \
  results/fresh-pair-results-20261028/q09-fresh-paired.json \
  /tmp/openevo-fresh-paired-recomputed.json
```

## 本地检查与实际实验

Python 3.10 以上；以下是纯 CPU 适配契约检查：

```bash
python3 -m pip install pytest
PYTHONPATH=.:vendor/openevo-webshop/src python3 -m pytest -q \
  vendor/openevo-webshop/tests/test_prompts.py \
  tests/test_webshop_evaluator_contract.py
```

实际训练还需要与具体历史协议兼容的 OpenEvo Core、模型与数据版本，以及对应 CUDA / PyTorch / WebShop 环境。适配包元数据中的 `openevo==0.1.10` 是历史接口声明，不表示当前任意同名安装都能重现全部实验。各协议中的 SHA 用于识别原始实验，不能以最新版本直接替代。

基础设施路径、设备和跟踪账户已替换为示例；历史启动脚本和冻结协议供研究审阅，**不是可直接使用的部署配置**。经过替换的文件不再具有原冻结文件的字节身份；重现实验需在自己的环境中建立新路径和身份记录。公开快照不运行训练、不申请 GPU，也未配置自动 CI。我们已验证的 CPU 检查见 [VALIDATION.md](VALIDATION.md)。

## 来源与发布边界

本仓库是原实验工作区提交 `5c77b92788332253ff969d5a7ea280bf6bbb54c6` 的筛选快照，采用独立 Git 历史。保留源文件哈希和公开文件哈希，明确哪些文件做了基础设施脱敏。私人文档、申请材料、通话、服务器收据、凭据、原始交互记录和完整数据集不在发布范围。发布的历史实验配置中，原本指向私人会议 RAW 的路径与可识别来源称谓改为不可执行的占位符，保留原件 SHA-256 和公开版 SHA-256 的变换记录；这些文件是研究快照，不是启动授权。

第三方声明见 [`third_party_notices`](third_party_notices)。其中的 MIT 声明仅覆盖其对应第三方代码，不自动为整个研究快照授予新许可证。WebShop、SEED、OpenEvo Core、模型和数据仍遵循各自的许可和获取条件。

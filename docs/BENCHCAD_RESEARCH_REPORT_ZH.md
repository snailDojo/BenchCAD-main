# BenchCAD 项目调研报告

> 本报告依据当前仓库实现整理，重点说明项目组成、端到端业务流程、评测数据以及性能指标。这里的“准确率”是广义的模型性能得分；除 QA 的离散题外，BenchCAD 并不使用传统分类 Accuracy，而使用几何相似度或数值答案得分。

## 1. 项目定位与结论摘要

BenchCAD 是一个面向**程序化参数 CAD**的大模型评测框架。被测模型需要理解或生成能够构造真实机械零件的 CadQuery Python 程序。项目以执行后的 CAD 几何或确定的数值答案为真值，不使用另一个大模型充当裁判。

当前数据基础和任务规模为：

- 17,900 个经过执行验证的 CadQuery 程序；
- 106 个工业零件族；
- 47 项 ISO、DIN、EN、ASME、IEC 等工程标准；
- 17,900 条 Vision2Code 数据、748 条 CodeEdit 数据，以及覆盖 200 个零件的 2,400 个 QA 问题。

仓库将能力拆为四项任务，但物理上放在三个任务目录中：

| 任务 | 模型输入 | 模型输出 | 主要能力 | 默认指标 |
|---|---|---|---|---|
| Vision2Code | 零件四视图组合图 + 指令 | CadQuery Python | 从视觉重建参数化 CAD | Voxel IoU |
| CodeEdit | 原始 CadQuery 代码 + 自然语言编辑指令 | 修改后的 CadQuery Python | 精确理解并修改 CAD 程序 | Normalized IoU |
| Vision-QA | 渲染图 + 数值问题 | JSON 数字数组 | CAD 视觉理解 | 类型感知 QA 得分 |
| Code-QA | CadQuery 代码 + 数值问题 | JSON 数字数组 | CAD 程序与几何语义理解 | 类型感知 QA 得分 |
| QA `both`（扩展模式） | 代码 + 渲染图 + 数值问题 | JSON 数字数组 | 多模态融合理解 | 类型感知 QA 得分 |

## 2. 系统组成

### 2.1 顶层统一调度器

根目录的 `benchcad.py` 是统一入口，负责：

1. 解析 `--task`、`--num`、`--model`、`--seed` 和 `--score`；
2. 选择 Vision2Code、CodeEdit、QA，或依次执行全部任务；
3. 检查本地数据量，不足时调用各任务下载脚本从 HuggingFace 拉取；
4. 将统一参数转换为各任务 `main.py` 的参数；
5. 以子进程运行各任务并汇总退出状态。

`--score` 只作用于 Vision2Code；CodeEdit 和 QA 使用各自固定的指标。

### 2.2 三个任务目录

三个任务目录大致遵循相同结构：

```text
<Task>/
├── main.py                 # 配置加载、抽样、模型×记录循环、均值汇总
├── configs/                # test/prod YAML 配置
├── pipeline/
│   ├── runner.py           # 单条记录的端到端执行
│   ├── prompt.py|modes.py  # prompt 和多模态输入构造
│   ├── store.py            # JSONL 结果持久化
│   └── plot.py             # 结果可视化
├── scoring/                # 任务专属评分逻辑
├── models/                 # 模型适配层（兼容性入口）
├── test_data/              # 仓库内小型 smoke 数据
└── tools/                  # 完整数据下载工具
```

实际共享的模型调用和几何基础设施主要位于 `benchcad_core/`：

- `benchcad_core.models.call_model()`：统一 OpenAI、Anthropic、Gemini、OpenRouter 等模型的调用形式；
- `benchcad_core.scoring.exec_cq`：提取和执行模型生成的 CadQuery 代码并导出 STEP；
- `benchcad_core.scoring.iou`：归一化 Voxel IoU 与 Normalized IoU；
- `benchcad_core.scoring.views`：STEP 渲染及四视图组合图；
- `benchcad_core.run_config`：生成参数与配置处理。

### 2.3 数据与结果层

完整数据托管在 HuggingFace `BenchCAD/BenchCAD`，生产运行时下载至各任务被 Git 忽略的 `data/` 目录；每个任务也提交少量 `test_data/` 供离线 smoke test。

每次运行会写入 `results.jsonl`，并按照任务键覆盖旧结果而非重复追加：

- Vision2Code：`(model, record_id)`；
- CodeEdit：`(mode, model, record_id)`；
- QA：`(model, record_id)`。

结果除主得分外，还记录执行状态、延迟、prompt/completion/reasoning/total token、估算费用、错误信息和产物路径。

## 3. 总体业务流程

```text
CLI / YAML 配置
      ↓
选择任务、模型和记录（可按 seed 随机抽样）
      ↓
检查并下载数据
      ↓
任务 main.py 遍历 model × mode × record
      ↓
单记录 runner 构造 system prompt、user text、image paths
      ↓
统一模型适配器调用大模型
      ↓
┌──────────────────────────────┬───────────────────────────┐
│ Vision2Code / CodeEdit       │ QA                        │
│ 提取 Python → 执行 → STEP    │ 解析等长 JSON 数字数组    │
│ → 几何比较                   │ → 按问题类型评分          │
└──────────────────────────────┴───────────────────────────┘
      ↓
保存原始输出、派生产物和 results.jsonl
      ↓
按模型计算样本得分的算术平均，并可生成柱状图
```

模型级汇总分统一采用样本得分的算术平均：

\[
S_{model}=\frac{1}{N}\sum_{i=1}^{N}s_i
\]

失败样本通常得到 0 分，因此不会从均值中排除。

## 4. 各任务的详细流程

### 4.1 Vision2Code：图像到 CadQuery

#### 输入数据

一条记录的关键字段为：

```json
{
  "record_id": "...",
  "family": "washer",
  "code_path": "codes/...py",
  "step_path": "steps/...step"
}
```

runner 根据 `step_path` 对真值 STEP 进行渲染，产生一张 2×2 四视图组合 PNG。模型实际获得：

- 说明相机方向、坐标系和输出约束的 system prompt；
- 固定的重建指令；
- 四视图组合 PNG。

真值 CadQuery 代码和 STEP 不发送给模型；真值代码只可能在 Composite 评分阶段用于提取特征。

#### 模型输出及处理

模型应输出包含 `import cadquery as cq` 且最终实体保存在 `result` 中的 Python 代码。runner 随后：

1. 提取代码；
2. 在隔离子进程中执行；
3. 导出预测 STEP；
4. 将预测 STEP 与真值 STEP 计算 Voxel IoU；
5. 可选计算 Composite score；
6. 尝试渲染预测 PNG 供人工检查；
7. 保存 `.py`、`.step`、`.png` 和结果行。

常见状态包括 `ok`、`api_fail`、`no_code`、`exec_fail` 和 `score_fail`。模型代码无法执行时，几何项按 0 处理。

### 4.2 CodeEdit：指令驱动的 CAD 修改

#### 输入数据

一条记录通常包含：

```json
{
  "record_id": "...",
  "family": "ball_knob",
  "edit_type": "dim",
  "instruction": "Increase ...",
  "iou": 0.8652,
  "orig_code_path": "codes/..._orig.py",
  "gt_code_path": "codes/..._gt.py",
  "orig_step_path": "steps/..._orig.step",
  "gt_step_path": "steps/..._gt.step"
}
```

模型实际获得：

- 要求“只进行最小必要修改”的 system prompt；
- 自然语言 `instruction`；
- 原始 CadQuery 源代码；
- 不提供图片，也不提供目标代码或目标 STEP。

#### 处理流程

模型返回修改后的完整 CadQuery 程序。runner 提取并执行代码得到 STEP，与 `gt_step_path` 计算 `model_iou`，再使用记录中的原始模型基线 `iou` 计算改进后的 Normalized IoU。这样可区分“目标本来就与原件很相似”和“模型确实完成了编辑”。

### 4.3 QA：CAD 数值问答

每条记录包含 `qa_pairs`：

```json
{
  "record_id": "...",
  "family": "threaded_adapter",
  "code_path": "codes/...py",
  "image_path": "images/...png",
  "qa_pairs": [
    {"question": "...", "answer": 13.8, "type": "dim"},
    {"question": "...", "answer": 1, "type": "integer"}
  ]
}
```

`image_path` 是否存在取决于数据分支和运行模式。三种 mode 为：

- `code`：把 CadQuery 源码作为文本发送，形成 Code-QA；
- `img`：发送渲染图片，形成 Vision-QA；
- `both`：同时发送代码与图片。

只把 `question` 发送给模型，`answer` 和 `type` 留在本地评分。模型必须按问题顺序输出等长 JSON 数字数组，例如 `[20, 2.5, 1]`；yes/no 分别编码为 1/0，尺寸单位为毫米。

runner 会去除可选 Markdown fence，查找 JSON 数组，验证数组长度，并把每个元素转换为浮点数。API 失败、无法解析或答案数量不匹配时，整条记录得 0 分。

## 5. 用于测试大模型性能的数据

### 5.1 数据集规模与内容

| 数据配置 | 规模 | 核心文件/字段 | 测试能力 |
|---|---:|---|---|
| `code_gen` | 17,900 条、106 个零件族 | CadQuery 真值代码、真值 STEP、由 STEP 得到的四视图 | 视觉识别、空间推理、参数化建模、可执行代码生成 |
| `edit-bench` | 748 条 | 原始/目标代码与 STEP、编辑指令、baseline IoU | 指令理解、代码定位、局部精确编辑、保持无关特征 |
| `QA` | 2,400 问题 / 200 个零件 | 代码或图片、问题、数值答案、题型 | 尺寸、数量、比例、布尔判断、方向和代码/CAD 语义 |

其中基础程序覆盖 106 个工业零件族和 47 项工程标准，包括齿轮、弹簧、钻头、螺纹适配件等，不只是立方体、圆柱等简单图元。

### 5.2 发送给模型与仅用于评分的数据

| 数据类型 | Vision2Code | CodeEdit | Code-QA | Vision-QA | 用途 |
|---|---:|---:|---:|---:|---|
| 自然语言 prompt | 是 | 是 | 是 | 是 | 定义任务和输出格式 |
| 四视图/渲染 PNG | 是 | 否 | 否 | 是 | 视觉输入 |
| 原始 CadQuery 代码 | 否 | 是 | 是 | 否 | 程序输入 |
| 编辑指令 | 否 | 是 | 否 | 否 | 指定修改目标 |
| QA 问题 | 否 | 否 | 是 | 是 | 数值问答输入 |
| 真值 STEP | 否 | 否 | 否 | 否 | 本地几何评分或输入图渲染 |
| 目标 CadQuery 代码 | 否 | 否 | 否 | 否 | 本地特征分析/数据真值 |
| QA 标准答案和类型 | 否 | 否 | 否 | 否 | 本地确定性评分 |
| baseline IoU | 否 | 否 | 否 | 否 | CodeEdit 归一化评分 |

## 6. 性能指标与计算公式

### 6.1 Voxel IoU（Vision2Code 基础指标）

#### 几何预处理

预测 STEP 和真值 STEP 分别经 CadQuery/OCP 导入并以容差 `0.05` 三角化。对于顶点 \(\mathbf v_i\)，计算包围盒中心：

\[
\mathbf c=\frac{\mathbf b_{min}+\mathbf b_{max}}{2}
\]

以包围盒最长边作为统一尺度：

\[
L=\max(\mathbf b_{max}-\mathbf b_{min})
\]

将顶点归一化为：

\[
\mathbf v'_i=\frac{\mathbf v_i-\mathbf c}{L}+(0.5,0.5,0.5)
\]

这会消除整体平移和等比例缩放，但保留长宽高比例和世界坐标朝向；评分器不进行旋转或镜像配准。

两个归一化网格以：

\[
p=\frac{1}{64}
\]

为 pitch 体素化并填充内部，随后居中放入带边界余量的 \(68^3\) 布尔数组。

#### IoU 公式

令 \(G\) 为真值实体占据的体素集合，\(P\) 为预测实体占据的体素集合：

\[
IoU(G,P)=\frac{|G\cap P|}{|G\cup P|}
\]

也可写为：

\[
IoU=\frac{TP}{TP+FP+FN}
\]

范围为 \([0,1]\)：1 表示体素化后完全一致，0 表示不重合或执行/几何处理失败。由于各实体独立缩放，IoU 主要衡量形状和相对比例，而不是绝对毫米尺寸；64³ 分辨率对极小孔、圆角或倒角也可能不够敏感。

### 6.2 CodeEdit Normalized IoU

设：

- \(I_m=IoU(P_{model},G_{target})\)：模型编辑结果与目标的 IoU；
- \(I_b=IoU(P_{original},G_{target})\)：不执行编辑时原始零件与目标的 baseline IoU。

则：

\[
NormIoU=clip\left(\frac{I_m-I_b}{1-I_b},0,1\right)
\]

解释：

- \(NormIoU=0\)：模型没有比原始零件更接近目标，或执行失败；
- \(NormIoU=1\)：模型结果与目标完全匹配；
- 当 \(I_b\ge 1\) 时，只有 \(I_m\ge 1\) 得 1，否则得 0。

### 6.3 Vision2Code Composite score

可选 `--score composite` 使用：

\[
S_{comp}=0.60I+0.20E+0.10F_1+0.05S_{CD}+0.05S_{HD}
\]

其中：

- \(I\)：固定朝向 Voxel IoU；
- \(E\in\{0,1\}\)：该零件族要求的 essential operations 是否全部满足；规则支持组内 OR、组间 AND；
- \(F_1\)：hole、fillet、chamfer 三类存在性特征的 F1；孔在生成 STEP 可用时优先通过 B-Rep 内圆柱面检测，圆角和倒角主要从代码操作提取；
- \(S_{CD}\)：Chamfer distance 映射后的 \([0,1]\) 分数；
- \(S_{HD}\)：Hausdorff distance 映射后的 \([0,1]\) 分数。

若零件族没有 essential-op 规则，移除 0.20 项，并将其余权重和为 0.80 的部分乘以 1.25 重新映射到 \([0,1]\)。`iou_rot` 参数仅为兼容/诊断保留，不进入最终得分。

#### Feature-F1

在三个特征布尔标签上统计 TP、FP、FN：

\[
Precision=\frac{TP}{TP+FP},\qquad Recall=\frac{TP}{TP+FN}
\]

\[
F_1=\frac{2\cdot Precision\cdot Recall}{Precision+Recall}
\]

该指标判断特征类别是否存在，并非对每个孔、每条圆角或倒角进行实例级位置和尺寸匹配。

#### Chamfer distance

从两个归一化网格表面各采样 \(n=2048\) 个点，记点集为 \(A,B\)：

\[
CD(A,B)=\frac{1}{|A|}\sum_{a\in A}\min_{b\in B}\|a-b\|_2^2
+\frac{1}{|B|}\sum_{b\in B}\min_{a\in A}\|b-a\|_2^2
\]

原始距离越小越好，映射为分数：

\[
S_{CD}=\begin{cases}
1,&CD\le 0.001\\
\frac{0.2-CD}{0.2-0.001},&0.001<CD<0.2\\
0,&CD\ge 0.2
\end{cases}
\]

#### Hausdorff distance

\[
HD(A,B)=\max\left(
\max_{a\in A}\min_{b\in B}\|a-b\|_2,
\max_{b\in B}\min_{a\in A}\|b-a\|_2
\right)
\]

映射为：

\[
S_{HD}=\begin{cases}
1,&HD\le 0.05\\
\frac{0.5-HD}{0.5-0.05},&0.05<HD<0.5\\
0,&HD\ge 0.5
\end{cases}
\]

Chamfer 反映平均表面偏差，Hausdorff 更容易发现局部最坏偏差。

### 6.4 QA 类型感知得分

#### 离散题

对于 `integer`、`count`、`boolean`、`bool`：

\[
s(p,g)=\begin{cases}
1,&p=g\\
0,&p\ne g
\end{cases}
\]

这是项目中最接近传统“准确率”的指标，适用于数量、布尔值和代码行号；差 1 也不得部分分。

#### 连续数值题

对于 `dim`、`ratio` 和其他非离散类型，在预测 \(p\) 与真值 \(g\) 非零且同号时：

\[
s(p,g)=\frac{\min(|p|,|g|)}{\max(|p|,|g|)}
\]

特殊情况：

- 若 \(p=0\) 或 \(g=0\)，仅当 \(p=g\) 时得 1，否则得 0；
- 若二者符号不同，得 0。

该公式对高估和低估对称。例如真值为 10，预测 8 或 12.5 都得 0.8。

一条记录包含 \(K\) 道题时：

\[
QA\_Score=\frac{1}{K}\sum_{j=1}^{K}s(p_j,g_j)
\]

### 6.5 辅助运行指标

各 runner 还记录但不纳入主质量分数的指标：

- `lat_s`：单次 API 调用延迟；
- prompt、completion、reasoning、total tokens；
- `cost_usd`：根据模型价格表和 token 估算的调用费用；
- 执行成功率：可由 `status == "ok"` 的记录占比派生；
- `api_fail`、`no_code`、`exec_fail`、`parse_fail`、`score_fail` 等失败分布。

实际比较模型时，建议同时报告主分数、执行/解析成功率、延迟和费用，而不是只看平均分。

## 7. 指标适用边界与风险

1. **几何正确不等于代码工程质量。** IoU 不评价变量命名、参数化程度、可维护性或建模特征树是否符合制造意图。
2. **归一化弱化绝对尺寸。** 两个同比例但毫米尺寸不同的实体可能获得很高 IoU；绝对尺寸能力应结合 QA 或额外尺寸约束评估。
3. **固定朝向会惩罚坐标轴错误。** 这符合当前任务要求，但如研究纯形状恢复，宜额外报告 24 个轴对齐旋转中的最大 IoU，不能直接替代官方得分。
4. **体素分辨率限制小特征。** 小孔、薄壁、小圆角和倒角对 64³ IoU 的贡献有限；Composite 的 Feature-F1 和表面距离只能部分补足。
5. **Feature-F1 是类别存在性而非实例匹配。** 它不直接度量每个孔的位置、轴向、直径和深度误差。
6. **Composite 含代码策略约束。** 几何等价但未使用规则指定操作的程序可能损失 essential-op 分。
7. **QA 混合了多种能力。** Code-QA 同时涉及 Python 静态分析、算术和 CAD 语义；Vision-QA 的精确尺寸也受渲染可观测性限制。
8. **宏平均需结合切片分析。** 简单和复杂零件等权；建议额外按 family、问题类型、编辑类型和失败状态分组报告。

## 8. 建议的评测结果报告模板

为保证结果可比较，建议每次实验至少披露：

```text
模型及明确版本：
任务 / QA mode：
数据配置与记录数：
抽样 seed：
生成参数（max_tokens、timeout）：
Vision2Code score 类型（iou/composite）：

主指标：mean IoU / mean Composite / mean NormIoU / mean QA score
成功率：ok / total
失败分布：api_fail / no_code / exec_fail / parse_fail / score_fail
效率：平均延迟、总 token、估算费用
切片结果：按 family / edit_type / QA type
原始 results.jsonl 和模型输出：
```

## 9. 总结

BenchCAD 的核心特点是把大模型输出放回真实 CAD 执行链中验证：生成类任务不是比较代码字符串，而是执行 CadQuery、导出 STEP，再和真值几何比较；问答类任务也只比较结构化数值答案。其数据覆盖图像、程序代码、自然语言编辑指令、数值问题、STEP 几何和工程零件族，分别测试视觉理解、程序理解、可执行生成和精确修改能力。

因此，解读 BenchCAD 成绩时应使用“任务得分”而非笼统的“准确率”：Vision2Code 看 IoU 或 Composite，CodeEdit 看相对 baseline 的 Normalized IoU，QA 则按离散精确匹配和连续对称比例得分后取平均。

# 评测与可视化脚本说明

本目录包含用于批量评测模型 Checkpoint、计算指标以及绘制趋势图的自动化脚本。这些脚本经过优化，支持断点续传、数据集划分（Train/Test/All）以及生成与评测解耦。

## 核心脚本概览

### 1. `evaluate_all_checkpoints.sh` (一键生成与评测)
**功能**：遍历指定目录下的所有 `checkpoint-*`，依次启动 vLLM Server，进行推理生成（Generation）并立即进行评测（Evaluation）。

**用法**：
```bash
# 默认运行模式（处理 Train + Test 所有数据）
./scripts/evaluate_all_checkpoints.sh

# 指定只跑测试集
./scripts/evaluate_all_checkpoints.sh test

# 指定只跑训练集
./scripts/evaluate_all_checkpoints.sh train
```

**关键特性**：
*   **断点续传 (Resume)**：
    *   **Checkpoint 级**：如果某个 Checkpoint 目录下已经存在 `eval_results.jsonl`，脚本会自动跳过该 Checkpoint，避免重复跑。
    *   **Sample 级**：生成过程中会实时（逐行）写入 `generated_responses.jsonl`。如果程序意外中断，重新运行脚本时，它会读取已有文件，**只生成剩余未完成的样本**。
*   **确定性路径**：输出目录不再包含随机时间戳，而是使用固定的 `code_<split>` 格式，以便于断点续传和后续画图工具的索引。

### 2. `compute_metrics_only.sh` (仅计算指标)
**功能**：在已经完成生成（即存在 `generated_responses.jsonl`）的情况下，跳过 vLLM 启动和生成步骤，直接根据现有结果计算准确率并生成 `eval_results.jsonl`。

**适用场景**：
*   生成过程已完成，但想重新计算指标。
*   生成过程耗时较长，想在生成结束后单独进行评测。

**用法**：
```bash
# 计算所有数据的指标
./scripts/compute_metrics_only.sh

# 计算测试集指标
./scripts/compute_metrics_only.sh test
```

### 3. `plot_all_metrics.sh` (一键画图)
**功能**：扫描输出目录，提取所有 Checkpoint 的评测结果，自动绘制三张准确率变化曲线图：
1.  **Test Accuracy**: 测试集准确率
2.  **Train Accuracy**: 训练集准确率
3.  **Overall Accuracy**: 整体准确率

**用法**：
```bash
./scripts/plot_all_metrics.sh
```
图片将保存在 `outputs/code_domain/outputs-5e7-bs16-ep3-sgd/plots/` 目录下。

---

## 输出目录结构

脚本运行后的结果保存在 `outputs/code_domain/outputs-5e7-bs16-ep3-sgd/` 下，结构如下：

```text
outputs/code_domain/outputs-5e7-bs16-ep3-sgd/
├── plots/                        # 画图结果
│   ├── accuracy_test.png
│   ├── accuracy_train.png
│   └── accuracy_all.png
│
├── checkpoint-10/
│   └── code_all/                 # 确定性子目录 (无时间戳)
│       ├── experiment_config.json
│       ├── generated_responses.jsonl # 生成结果 (支持增量写入)
│       ├── eval_results.jsonl        # 评测结果
│       └── eval.log
│
├── checkpoint-44/
│   └── code_all/
│       └── ...
└── ...
```

## 高级功能说明

### 数据集划分 (Splits)
系统支持三种划分模式，通过脚本的第一个参数控制：
*   **`test`**: 仅使用测试集（通常为 held-out 验证集）。
*   **`train`**: 仅使用训练集（用于观察模型在训练数据上的拟合程度）。
*   **`all`** (默认): 将训练集和测试集合并。
    *   生成时：一次性生成所有样本的回答。
    *   评测时：会自动识别样本标签（`dataset_split`），分别计算并打印 **Train**、**Test** 和 **Overall** 的准确率。

### 资源配置
默认配置在脚本头部定义，可根据需要修改：
*   **GPU 配置**: `CUDA_VISIBLE_DEVICES="4,5,6,7"`
*   **vLLM 端口**: `24444`
*   **模型路径**: `BASE_DIR` 变量指定 Checkpoint 所在根目录。

## 常见工作流

1.  **启动评测任务**（建议在 tmux 或 nohup 中运行）：
    ```bash
    nohup ./scripts/evaluate_all_checkpoints.sh > eval.log 2>&1 &
    ```
2.  **查看进度**：
    可以随时查看生成的日志，或者直接看输出目录下的文件增长情况。
3.  **任务中断后恢复**：
    直接再次运行上述命令即可。系统会自动检测并跳过已完成的部分。
4.  **生成报表**：
    任务完成后（甚至运行过程中），运行画图脚本查看趋势：
    ```bash
    ./scripts/plot_all_metrics.sh
    ```

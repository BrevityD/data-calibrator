# data-calibrator

Experiments on calibrating the correlation between data domains and model alignment outcomes.

## Usage

### Installation

Install uv before proceeding.

```console
foo@bar:~$ git clone git@github.com:BrevityD/data-calibrator.git
foo@bar:~$ cd data-calibrator
foo@bar:~/data-calibrator$ uv venv .venv
foo@bar:~/data-calibrator$ uv sync
```

Activate your virtual environment:

```console
foo@bar:~/data-calibrator$ source .venv/bin/activate
```

Log in to `WandB` (optional):

Skip this step if not using WandB.

### Reproduce Samples

Samples:
1. [EXP-1](./samples/experiment1/README.md)

### Download Datasets

Configure your `hf-token` in `data-calibrator/datasets/download_data.py`.

To obtain your `hf-token`, visit [https://huggingface.co/settings/tokens](https://huggingface.co/settings/tokens). To access restricted datasets like `livecodebench`, grant permission for "Read access to contents of all public gated repos you can access".

```console
foo@bar:~/data-calibrator$ source .venv/bin/activate
(.venv)foo@bar:~/data-calibrator/datasets$ python download_data.py
```

### 评测系统 (Evaluation)

本项目包含一套专门适配思维链（CoT）模型的自动化评测流水线。

#### 1. 核心逻辑
- **CoT 后处理**: 自动识别并剥离 `<think>` 标签内容，确保评测仅针对最终生成的代码。
- **环境隔离**: 每个测试用例都在独立的临时目录中运行，避免产生垃圾文件或交叉干扰。
- **自动统计**: 统计包括 Pass@1 (Greedy)、格式合规率（Format Compliance）以及长文本截断率。

#### 2. 运行方法
修改 `scripts/run_code_eval.sh` 中的模型路径后运行：
```bash
bash scripts/run_code_eval.sh
```

#### 3. 实验追踪
评测结果保存在 `outputs/` 目录下，每个运行目录包含 `experiment_config.json`，详细记录了本次实验的所有推理超参数（Temperature, Top-p, Max Tokens 等）。

---

## 目录结构与详细逻辑 (Detailed Directory Logic)

本次重构实现了功能逻辑、执行入口与调试工具的彻底分离。

### 1. `evaluation/` —— 评测核心模块
这是整个评测系统的核心，采用 Python 模块化设计，支持通过 `python -m evaluation` 运行。

*   **`core.py` (功能逻辑库)**:
    *   **CoT 剥离逻辑 (`extract_code`)**: 专门针对 DeepSeek-R1 等思维链模型优化。先通过正则表达式彻底删除思考过程，再提取剩余部分中的 Python 代码块。
    *   **沙箱隔离执行 (`evaluate_code_correctness`)**: 为每个测试用例创建一个 `tempfile.TemporaryDirectory`。所有代码运行和文件操作都被限制在临时目录中，运行结束后自动销毁，防止污染项目根目录。
    *   **异步批量推理 (`generate_responses`)**: 基于 `AsyncOpenAI` 客户端实现高并发推理。使用信号量限制并发数，并在生成结束后自动统计 `finish_reason` 以计算模型输出的截断率。
    *   **实验追踪 (`run_eval`)**: 自动导出 `experiment_config.json`，记录所有推理超参数（Temperature, Top-p 等），确保实验可追溯。
*   **`__main__.py` (CLI 入口)**:
    *   将逻辑库封装为命令行工具，负责处理 `--domain`、`--temperature` 等参数解析。

### 2. `scripts/` —— 执行入口目录
该目录仅存放 Shell 脚本（`.sh`），作为用户操作的顶层入口。

*   **`run_code_eval.sh` (自动化流水线)**:
    *   **动态算力分配**: 自动读取 `CUDA_VISIBLE_DEVICES` 计算 GPU 数量，并设置 vLLM 的并行度。
    *   **守护进程管理**: 后台启动 vLLM 服务器，并通过轮询检查服务可用性。
    *   **自动清理**: 使用 `trap` 钩子，确保无论脚本正常结束还是报错，都能自动杀掉后台的服务器进程。

### 3. `tools/` —— 辅助开发工具
存放用于开发调试和数据验证的独立脚本。

*   **`inspect_train_data.py`**: 用于直观打印处理后的数据 JSON 结构，确认 Prompt 模板是否正确。
*   **`verify_code_eval.py`**: 评测引擎的单元测试工具，通过模拟样本验证 Pass/Fail 判定逻辑是否符合预期。
*   **`check_dataset_columns.py`**: 快速自检加载的数据集是否包含评测所需的关键字段。

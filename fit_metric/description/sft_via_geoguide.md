# sft_via_geoguide.py 实现思路

## 目标

在 math/code 两个领域的 SFT 训练中，利用黎曼度规上的测地线方向动态调整数据配比。每隔固定训练步数（而非每个 epoch）重新评估模型状态、计算测地线、调整 math:code 配比，实现更细粒度的在线配比控制。

## 整体流程

```
加载数据 → 预 tokenize → [评估 loss → 归一化 → 测地线方向 → 配比 → 训练 N 步] × segments
```

外层循环以 **segment**（固定步数的训练段）为单位，而非 epoch。每个 segment：

1. 评估当前模型的 math/code loss
2. 归一化到 $[0,1]^2$ 坐标空间
3. 计算从当前位置到目标直线的测地线，取起点切向量
4. 通过 $J^{-1}$ 将切向量映射为 math:code 配比
5. 按新配比混合数据，训练 `rebalance_steps` 步

## 核心设计

### 1. 按步数重新配比（segment-based rebalancing）

传统做法是每个 epoch 结束后调整配比，但 epoch 长度取决于数据集大小，粒度不可控。改为每 `rebalance_steps` 步重新配比一次：

- `max_segments`：外层循环安全上限（默认 1000）
- `rebalance_steps`：每段训练步数（默认 20）
- `max_steps`：全局累计步数预算（硬上限）

每段实际步数 = $\min(\text{rebalance\_steps},\; \text{max\_steps} - \text{global\_step})$。

SFTTrainer 中设置 `num_train_epochs=9999, max_steps=segment_steps`，用 `max_steps` 精确控制段长，`num_train_epochs` 设大值保证 dataloader 循环不会因数据不够提前停止。

### 2. 预 tokenize（`pre_tokenize_pool`）

SFTTrainer 在 `__init__` 时对数据集做 tokenization（apply_chat_template → tokenize → truncate）。每段新建 Trainer 会重复这个开销。

解决方案：在主循环开始前，对整个数据池做一次性 tokenize：

```python
def pre_tokenize_pool(dataset, tokenizer, max_length):
    def _tokenize(example):
        prompt_ids = tokenizer.apply_chat_template(example["prompt"])
        full_ids = tokenizer.apply_chat_template(
            example["prompt"] + example["completion"]
        )
        completion_mask = [0]*len(prompt_ids) + [1]*(len(full_ids)-len(prompt_ids))
        # truncate
        if len(full_ids) > max_length:
            full_ids = full_ids[:max_length]
            completion_mask = completion_mask[:max_length]
        return {"input_ids": full_ids, "completion_mask": completion_mask}
    return dataset.map(_tokenize)
```

之后每段 Trainer 使用 `dataset_kwargs={"skip_prepare_dataset": True}` 跳过内部 tokenize，直接消费预处理好的 `input_ids` + `completion_mask`。

eval 数据集（math_test, code_test）也同样预处理，否则 `skip_prepare_dataset` 会跳过 eval 的 tokenize 导致报错。

### 3. 无状态优化器设计

每段新建 Trainer 意味着优化器状态会丢失。通过以下设置消除影响：

- `adam_beta1=1e-12, adam_beta2=1e-12`：动量近似为零，Adam 退化为 SGD
- `lr_scheduler_type="constant"`：学习率恒定，不依赖 step 计数

这样每段新建 Trainer 不会丢失有意义的优化器状态。

### 4. 归一化与测地线方向

从 `m2c.json` / `c2m.json` 提取全局 min-max 归一化参数，将原始 math/code loss 映射到 $[0,1]^2$。

在归一化坐标空间中，调用 `multi_start_variational_geodesic_to_line` 计算从当前位置到目标直线（如 $x = 0.2$）的最短测地线，取起点处切向量 $\dot{\gamma}(0)$。

### 5. 切向量 → 配比（`tangent_to_ratio`）

切向量在 loss 空间中，需要通过 Jacobian 逆映射回 ratio 空间：

$$\mathbf{a} = J^{-1} \dot{\gamma}(0)$$

其中 $J = [v_{\text{math}} \mid v_{\text{code}}]$ 是两个方向模型在当前点的输出。

配比计算：$p_{\text{math}} = |a_1| / (|a_1| + |a_2|)$。若出现负系数，取绝对值后归一化（Proposal §8.5 truncation）。

### 6. 多设备管理

训练和测地线计算分别在不同 GPU 上进行，避免显存冲突：

- `train_device`：SFT 训练用
- `geo_device`：测地线计算用（geo_common + variational_geodesic）

启动时设置 `CUDA_VISIBLE_DEVICES` 包含两张卡，geo_common 初始化后缩减为仅训练卡 + 设置 `WORLD_SIZE=1`，防止 Trainer 做 DataParallel。

## 停止条件

三个条件任一满足即停止：

1. `global_step >= max_steps`：步数预算耗尽
2. `segment >= max_segments`：段数上限
3. `target_loss > 0` 且目标领域 loss 降到阈值以下：early stopping

## 输出

- `geoguide_log.json`：完整日志，包含每段的配比、loss、归一化坐标、切向量、测地线路径等
- `epoch_summary.json`：每段摘要（loss、坐标、配比、步数）
- `checkpoint_step{N}/`：模型 checkpoint（按 `save_steps` 间隔保存）
- `sft_via_geoguide.log`：stdout/stderr 完整日志

## 可视化

配套 `draw_geoguide_trajectory.py` 读取 `geoguide_log.json`，在 $\log\det G$ 热力图上绘制：

- 模型归一化坐标的 segment 间轨迹（红色箭头）
- 每个 segment 起点处的测地线路径（橙色渐变）
- 目标直线（青色虚线）
- segment 标记点和配比标注

## 默认运行参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `rebalance_steps` | 20 | 每段训练步数 |
| `max_segments` | 1000 | 外层循环上限 |
| `max_steps` | -1 | 全局步数预算（-1=不限） |
| `total_train_size` | 1000 | 每段混合数据集大小 |
| `learning_rate` | 2e-7 | 学习率 |
| `per_device_train_batch_size` | 4 | 批大小 |
| `gradient_accumulation_steps` | 4 | 梯度累积 |
| `geo_target_domain` | math | 优化目标领域 |
| `geo_target_value` | 0.2 | 目标直线坐标 |

## 依赖

- `geo_common`：度规张量计算、热力图、椭圆绘制
- `variational_geodesic`：多起点变分测地线求解
- `datacalibrator.datasets`：math/code 数据集加载
- `trl.SFTTrainer`：SFT 训练

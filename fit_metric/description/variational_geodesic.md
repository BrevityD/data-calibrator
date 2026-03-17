# variational_geodesic.py 实现思路

## 目标

求解从给定起点到竖直线 $x = x_{\text{target}}$ 的最短测地线。终点的 $x$ 坐标固定为 $x_{\text{target}}$，$y$ 坐标由优化自动确定。通过最小化离散能量，自动满足横截性条件（终点切向量与直线在度规意义下正交）。

## 核心思路

### 1. 离散变分法求测地线

将路径离散为 $N+1$ 个节点 $q_0, q_1, \dots, q_N$，其中：
- $q_0$ = 起点（固定）
- $q_N = (x_{\text{target}},\; y_{\text{free}})$，$x$ 固定，$y$ 自由
- $q_1, \dots, q_{N-1}$ 为内部自由节点

定义离散能量：

$$E = \sum_{i=0}^{N-1} \Delta q_i^T \, G(m_i) \, \Delta q_i$$

其中 $\Delta q_i = q_{i+1} - q_i$，$m_i = \frac{1}{2}(q_i + q_{i+1})$ 为中点，$G$ 为该点的度规张量（由神经网络 `metric_tensor_xy_batch` 提供）。

最小化该能量等价于求测地线，同时终点 $y$ 自由使得横截性条件自然满足。

### 2. 对数障碍约束

坐标需限制在 $(0, 1)$ 范围内。采用对数障碍函数：

$$E_{\text{barrier}} = -\mu \sum \left[\log(c - \epsilon) + \log(1 - \epsilon - c)\right]$$

$\mu$ 随迭代逐步退火（乘以 `barrier_anneal`），使障碍项逐渐减弱，最终结果趋近无约束最优解。

### 3. 两阶段多起点搜索

为避免陷入局部最优，采用两阶段策略：

**阶段 1 — 批量粗搜索（`batch_coarse_search`）**
- 在 $y \in [0.05, 0.95]$ 上均匀取 $K$ 个候选终点
- 将 $K$ 条路径的参数打包为 $(K, n_{\text{free}})$ 矩阵
- 用 Adam 优化器并行优化，所有路径共享一次 NN forward pass（$K \times N$ 个中点一次性计算度规），大幅提升效率
- 按弧长排序，筛选出 top-$k$ 候选

**阶段 2 — L-BFGS 精化（`variational_geodesic_to_line`）**
- 将粗搜索的 top-$k$ 路径重采样到更多节点（`refine_N`）
- 用 L-BFGS（strong Wolfe line search）逐条精化
- 关闭障碍项（$\mu = 0$），依赖粗搜索已将路径引导到可行域内部
- 取弧长最短者为最终结果

### 4. 横截性检验（`check_transversality`）

验证终点切向量 $\dot{q}_N$ 与直线切向量 $e_y = (0,1)$ 在度规意义下的正交性：

$$\cos\theta = \frac{\dot{q}_N^T \, G \, e_y}{\|\dot{q}_N\|_G \, \|e_y\|_G}$$

该值越接近 0，说明横截性条件满足越好。

### 5. 弧长计算（`compute_variational_arc_length`）

$$L = \sum_{i=0}^{N-1} \sqrt{\Delta q_i^T \, G(m_i) \, \Delta q_i}$$

批量计算所有中点的度规张量后向量化求和。

## 自由变量编码

总共 $2(N-1) + 1$ 个标量：
- 内部节点 $q_1, \dots, q_{N-1}$ 各有 $x, y$ 两个分量 → $2(N-1)$
- 终点 $y$ 坐标 → $1$

## 输出

- 最优路径的可视化图（叠加 log det G 热力图和度规椭圆）
- JSON 日志：起终点、能量、弧长、横截性指标、收敛信息、所有候选摘要

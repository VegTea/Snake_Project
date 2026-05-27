# 训练不稳定（mean_value_function → inf）问题分析与修复

## 问题现象

在 `snake_velocity_flat_tracking` 训练过程中（实验目录 `2026-05-26_13-35-16_exp1`）：

- **步骤 4–15**：`ang_vel_xy_l2`、`joint_acc_l2`、`track_lin_vel_xy_exp` 三项指标出现极大负值
- **步骤 ~180**：`mean_value_function` 输出为 `inf`，训练崩溃

## 根因分析

### 核心原因：SVD 分解在近直线构型下主轴不稳定

[`virtual_chassis.py`](../source/snake_project/snake_project/tasks/manager_based/velocity_tracking/mdp/virtual_chassis.py) 中 `compute_virtual_chassis_frame` 函数对 15 个身体连杆的 3D 位置矩阵做 SVD 分解，求取虚拟底盘的主轴方向。

当蛇形机器人处于**近似直线构型**时（训练初期频繁出现）：

1. 位置矩阵的秩接近 1，只有沿身体方向的奇异值较大，其余两个奇异值接近零
2. 对应近零奇异值的左奇异向量（主轴）在数学上几乎任意——任何与第一主轴正交的向量都满足条件
3. 连续仿真步之间，微小数值扰动导致这些主轴**旋转 ~90°**，甚至完全翻转
4. 世界坐标系下的速度投影到这些不稳定主轴上，得到的虚拟底盘速度在两个 step 之间剧烈跳动
5. 跟踪误差 `(command - actual_vel_vc)^2` 变得极大，线性惩罚项 `0.5 * sqrt(error)` 可达到 **-50 甚至更低**
6. 乘以权重 5.0 后，该项主导总奖励信号

```
熵增链：
极端负奖励 → 极端负回报 → 价值函数拟合困难 → value network 权重膨胀 → step 180 溢出为 inf
```

### 次要原因：ang_vel_xy_l2 使用 root body frame

原本的 `ang_vel_xy_l2`（Isaac Lab 内置函数）对 `root_ang_vel_b`（base_link/蛇头）的 body-frame 角速度做 L2 惩罚。对于蛇形机器人，蛇头可以独立于身体快速旋转，此惩罚物理意义不当。

## 修复方案

### 修改文件清单

| 文件 | 修改内容 |
|------|----------|
| `mdp/virtual_chassis.py` | SVD → 正则化 Gram 矩阵特征分解 |
| `mdp/rewards.py` | 新增 `VirtualChassisAngVelXYL2` + reward clipping |
| `velocity_env_cfg.py` | 更新奖励项配置 |

---

### 修复 1：SVD → 正则化特征分解

**文件**: `mdp/virtual_chassis.py` — `compute_virtual_chassis_frame()`

**改动**：将 `torch.linalg.svd(data_matrix)` 替换为对 Gram 矩阵 `data_matrix @ data_matrix^T` 做带 Tikhonov 正则化的特征分解：

```python
gram = torch.bmm(data_matrix, data_matrix.transpose(1, 2))
eye = torch.eye(3, device=body_pos_w.device).unsqueeze(0)
gram_reg = gram + reg * eye * torch.amax(gram, dim=(1, 2), keepdim=True)
_, eigvecs = torch.linalg.eigh(gram_reg)
axes_w = torch.flip(eigvecs, dims=[2])  # 降序排列，等效于 SVD 顺序
```

**原理**：添加 `reg * max(eigenvalue) * I`（默认 `reg=1e-6`）确保所有特征值至少为最大特征值的 1e-6 倍。这等效于对奇异值做下限约束，使得近零奇异值对应的主轴不再"任意"，而是由正则化项平滑确定。对于良好构型（奇异值分布正常），1e-6 的正则化项可以忽略不计。

数学上等价于将原始 SVD 问题 `min ||data - U S V^T||` 替换为 `min ||data - U S V^T|| + reg * ||S||^2`，防止 S 中出现近零奇异值。

---

### 修复 2：reward clipping 防止极端负值

**文件**: `mdp/rewards.py` — `VirtualChassisTrackLinVelXYExp.__call__()`

**改动**：新增 `reward_clip_min` 参数，对单步奖励做下界截断：

```python
raw_reward = exp_reward - lin_penalty
if reward_clip_min is not None:
    return torch.clamp(raw_reward, min=reward_clip_min)
return raw_reward
```

**配置**: `reward_clip_min = -20.0`，即单步该项奖励不低于 -20（加权后不低于 -100）。

---

### 修复 3：虚拟底盘角速度惩罚（替代 root body frame）

**文件**: `mdp/rewards.py` — 新增 `VirtualChassisAngVelXYL2`

**改动**：新增奖励项类，计算所有虚拟底盘连杆角速度的均值，投影到虚拟底盘坐标系后对 xy 分量做 L2 惩罚：

```python
vc_ang_vel_w = body_ang_vel_w.mean(dim=1)         # 世界系下均值
ang_vel_vc = axes_w^T @ vc_ang_vel_w               # 投影到虚拟底盘系
raw_penalty = sum(ang_vel_vc[:, :2]^2)             # xy 分量 L2
return clamp(raw_penalty, max=max_penalty)          # 上界截断
```

同时添加 `max_penalty=10.0` 上界截断（加权后最大惩罚 = -0.5）。

---

### 修复 4：配置文件更新

**文件**: `velocity_env_cfg.py`

```python
# track_lin_vel_xy_exp: 新增 reward_clip_min
track_lin_vel_xy_exp = RewTerm(
    func=mdp.VirtualChassisTrackLinVelXYExp,
    weight=5.0,
    params={
        ...
        "reward_clip_min": -20.0,  # ← 新增
    },
)

# ang_vel_xy_l2: 替换为虚拟底盘版本
ang_vel_xy_l2 = RewTerm(
    func=mdp.VirtualChassisAngVelXYL2,  # ← 原为 mdp.ang_vel_xy_l2
    weight=-0.05,
    params={"asset_cfg": virtual_chassis_body_cfg(), "max_penalty": 10.0},  # ← 新增
)
```

## 验证

重新启动训练：

```bash
bash start_train.sh
```

或从 checkpoint 恢复：

```bash
bash resume_train.sh
```

预期效果：
- 训练初期（step 4–15）的 `ang_vel_xy_l2`、`track_lin_vel_xy_exp` 指标不再出现极端负值
- `mean_value_function` 保持在合理数值范围内，训练可稳定运行至 5000+ steps

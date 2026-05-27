# 训练不稳定（value_function_loss → inf）问题分析与修复

## 问题现象

在 `snake_velocity_flat_tracking` 训练过程中，出现两次训练崩溃：

- **第一次崩溃**（实验 `2026-05-26_13-35-16_exp1`）：
  - 步骤 4–15：`ang_vel_xy_l2`、`joint_acc_l2`、`track_lin_vel_xy_exp` 三项指标出现极大负值
  - 步骤 ~180：`mean_value_function` 输出为 `inf`，训练崩溃

- **第二次崩溃**（实验 `2026-05-27_05-25-16_exp1`，已应用第一轮修复）：
  - 训练在 ~1000–2000 步间 `value_function_loss` 趋近 `inf`，训练崩溃

## 根因分析

### 第一层：奖励函数产生极端负值（第一次崩溃的根因）

[`virtual_chassis.py`](../source/snake_project/snake_project/tasks/manager_based/velocity_tracking/mdp/virtual_chassis.py) 中 `compute_virtual_chassis_frame` 函数对 15 个身体连杆的 3D 位置矩阵做 SVD 分解求取虚拟底盘主轴。

当蛇形机器人处于**近似直线构型**时（训练初期频繁出现）：

1. 位置矩阵的秩接近 1，两个近零奇异值对应的主轴在数学上几乎任意
2. 连续步间微小数值扰动导致主轴 ~90° 旋转，投影速度剧烈波动
3. 跟踪误差极大 → 线性惩罚 `0.5 * sqrt(error)` 达到 -50 甚至更低
4. 极端负奖励 → 极端负回报 → value network 权重膨胀 → 溢出

### 第二层：Value Network 发散（第二次崩溃的根因）

第一轮修复（SVD 正则化 + reward clipping）缓解了奖励层面的问题，但 **value function 发散**仍有以下深层原因：

1. **缺失的 NaN 传播路径**：reward 函数只检查了 `body_pos_w` 是否有限，但未检查 `body_lin_vel_w` 和 `body_ang_vel_w`。仿真物理引擎在极端状态下可能产生 NaN 速度，绕过了有限性检查，NaN 通过 reward → return → value loss → gradient → weights 路径传播

2. **critic 无观测归一化**：`critic_obs_normalization=False` 导致 value network 直接处理原始观测值（不同尺度：角速度 ~5 rad/s，关节位置 ±1.57 rad）。ELU 对正值无上界，大值激活在网络中逐层放大，权重逐渐膨胀

3. **value_loss_coef 过高 + 全局 advantage 归一化**：`value_loss_coef=0.01` 和 `normalize_advantage_per_mini_batch=False` 的组合使得 value network 尝试在全 batch 的极端 advantages 上快速拟合，每次更新步幅过大

4. **PPO value clipping 只约束变化量**：`use_clipped_value_loss=True` 裁剪的是 `V_new - V_old` 的变化量，而非 V 的绝对值。Value 可以多步累积增长直至溢出

```
完整因果链：
NaN 速度 → 未检测 → NaN reward → NaN return → NaN value loss → NaN grad → NaN weights → inf forward pass
                                    ↓（或）
大尺度观测 → 未归一化 → ELU 激活放大 → 权重累积增长 → 溢出
```

## 修复方案

### 修改文件清单

| 文件 | 第一轮修复 | 第二轮修复 |
|------|-----------|-----------|
| `mdp/virtual_chassis.py` | SVD → 正则化 Gram 特征分解 | — |
| `mdp/rewards.py` | 新增 `VirtualChassisAngVelXYL2` + reward clipping | 添加 body velocity NaN 检查 |
| `velocity_env_cfg.py` | 更新奖励项配置 | — |
| `agents/rsl_rl_ppo_cfg.py` | — | critic_obs_normalization + advantage 归一化 + value_loss_coef 降低 |

---

### 第一轮修复（已提交）

#### 修复 1a：SVD → 正则化特征分解

**文件**: `mdp/virtual_chassis.py` — `compute_virtual_chassis_frame()`

将 `torch.linalg.svd(data_matrix)` 替换为 Gram 矩阵的 Tikhonov 正则化特征分解：

```python
gram = torch.bmm(data_matrix, data_matrix.transpose(1, 2))
gram_reg = gram + reg * eye * torch.amax(gram, dim=(1, 2), keepdim=True)
_, eigvecs = torch.linalg.eigh(gram_reg)
axes_w = torch.flip(eigvecs, dims=[2])   # 降序 = SVD 顺序
```

`reg=1e-6` 确保最小特征值 ≥ 1e-6 × 最大特征值，消除近零奇异值导致的主轴不稳定性。

#### 修复 1b：reward clipping

**文件**: `mdp/rewards.py` — `VirtualChassisTrackLinVelXYExp`

新增 `reward_clip_min=-20.0` 参数：

```python
raw_reward = exp_reward - lin_penalty
return torch.clamp(raw_reward, min=reward_clip_min)
```

#### 修复 1c：虚拟底盘角速度惩罚

**文件**: `mdp/rewards.py` — 新增 `VirtualChassisAngVelXYL2`

替代 Isaac Lab 内置的 root body frame `ang_vel_xy_l2`，改用所有虚拟底盘连杆角速度均值投影到虚拟底盘系：

```python
vc_ang_vel_w = body_ang_vel_w.mean(dim=1)
ang_vel_vc = axes_w^T @ vc_ang_vel_w
raw_penalty = sum(ang_vel_vc[:, :2]^2)
return clamp(raw_penalty, max=max_penalty)
```

---

### 第二轮修复（本次）

#### 修复 2a：body velocity NaN 防护

**文件**: `mdp/rewards.py` — 所有三个 VirtualChassis reward 类的 `__call__`

**问题**：之前的有限性检查只覆盖 `body_pos_w`，未覆盖 `body_lin_vel_w` 和 `body_ang_vel_w`。物理引擎产生 NaN 速度时会绕过检查。

**修复**：扩展有限性检查到全部 body 数据：

```python
# 之前（不完整）
if not torch.isfinite(body_pos_w).all():
    return torch.zeros(self.num_envs, device=self.device)

# 之后（完整）
if not (torch.isfinite(body_pos_w).all() and
        torch.isfinite(body_lin_vel_w).all() and
        torch.isfinite(body_ang_vel_w).all()):
    return torch.zeros(self.num_envs, device=self.device)
```

影响类：`VirtualChassisTrackLinVelXYExp`、`VirtualChassisTrackAngVelZExp`、`VirtualChassisAngVelXYL2`

#### 修复 2b：启用 critic 观测归一化

**文件**: `agents/rsl_rl_ppo_cfg.py`

```python
# 之前
critic_obs_normalization=False

# 之后
critic_obs_normalization=True
```

Value network 维护观测 running statistics (μ, σ)，对输入做 `(obs - μ) / (σ + ε)` 归一化。消除不同尺度观测值导致的激活放大问题。

#### 修复 2c：advantage per-mini-batch 归一化 + 降低 value_loss_coef

**文件**: `agents/rsl_rl_ppo_cfg.py`

```python
# 之前
value_loss_coef=0.01
# (normalize_advantage_per_mini_batch 默认 False)

# 之后
value_loss_coef=0.005
normalize_advantage_per_mini_batch=True
```

- `value_loss_coef`: 0.01 → 0.005，value 更新幅度减半，降低发散风险
- `normalize_advantage_per_mini_batch=True`: 在每个 mini-batch 内独立归一化 advantage，防止全 batch 中极端 advantage 主导梯度

## 配置变更汇总

| 参数 | 旧值 | 新值 | 原因 |
|------|------|------|------|
| `critic_obs_normalization` | `False` | `True` | 防止未归一化观测导致激活放大 |
| `value_loss_coef` | `0.01` | `0.005` | 更保守的 value function 更新 |
| `normalize_advantage_per_mini_batch` | `False` | `True` | mini-batch 内独立归一化 advantage |

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
- 训练初期不再出现极端负值或 NaN 的 reward 指标
- `value_function_loss` 保持在合理范围（< 100），训练可稳定运行至 5000+ steps
- `mean_value_function` 平滑变化，不发散

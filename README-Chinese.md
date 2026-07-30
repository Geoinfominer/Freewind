# Freewind

> **GPT-style decoder-only weather model on ERA5** — a research prototype for studying long-context, block-wise causal sequence modeling in data-driven NWP. Not a claimed SOTA system.

**Freewind** 是一个独立研究原型：用 **Encoder → GPT Transformer（RoPE / 分块因果 / KV cache）→ Decoder** 在 ERA5 上做多时刻因果预报，重点探索「固定参数量下，更长上下文能否延缓自回归误差累积」，而非复现短上下文 ViT 范式的榜单数字。

作者：郭金雨（独立研究）。本仓库为可训练的模型骨架与数据接口；**不含** 数 TB 级 ERA5 原始数据与预训练权重。

---

## 动机与范式

现有数据驱动全球预报（Pangu / Fuxi / FengWu / GraphCast 等）在短中期已很强，但训练与推理多围绕 **短上下文**（1→1 或 2→1）与 **图像/图范式**（ViT、Cube Embedding、GNN）。多帧常被拼进通道再经空间卷积混合，时间顺序与趋势方向容易被弱化；加长输入序列往往要改输入层结构，**参数量与序列长度难以解耦**。

Freewind 的工作假说是：

1. 把气象场切成时空 token，用 **按时刻分块的因果注意力（block-wise / timestep-block causal attention）：同一时刻内空间 patch 可互相注意，不可看未来时刻** 做 next-step 预测，可在不改主干宽度的前提下调节序列长度；
2. 更长的显式历史可能有助于约束自回归演化（是否成立需对照实验检验）；
3. 在固定算力下，应系统研究 \(N\)（参数）、\(S\)（上下文）、\(D\)（数据）与有效预报时效的关系，而不是「塞满一张卡」。

因此它更接近 **LLM 式序列建模框架用于 LWM（Large Weather Model）**，而不是「再做一个更强的单步 Fuxi」。

---

## 架构概览

![Freewind 模型架构](model_architecture.png)

```
ERA5 场 (B, S, C, H, W)
        │
        ▼
┌───────────────────┐
│  Encoder          │  PeriodicConv 双分支 CompressPatch
│  (局部压缩)        │  → patch embedding + 可选 skip
└─────────┬─────────┘
          │  (B, S, P, E) → flatten (B, S·P, E)
          ▼
┌───────────────────┐
│  GPT Transformer  │  分块（时刻块）因果掩码 · 空间 Embedding
│                   │  时间 RoPE · KV cache · 可选 VAE 噪声
└─────────┬─────────┘
          │  reshape 回 patch 网格
          ▼
┌───────────────────┐
│  Decoder          │  镜像上采样 (PixelShuffle / ConvTranspose)
│  (场重建)          │  + encoder skip → (B, S, C, H, W)
└───────────────────┘
```

| 模块 | 文件 | 要点 |
|------|------|------|
| Encoder | `model_archi_encoder.py` | 经向 circular / 纬向 constant 的 `PeriodicConv2d`；双分支压缩后相加；`PermuteLayerNorm2d` |
| GPT | `model_archi_GPT.py` | 按 `patches_one_step` 时刻块的分块因果可见性；空间可学习 PE；时间 RoPE；推理侧 KV cache / 滑动窗示例 |
| Decoder | `model_archi_decoder.py` | 从 Encoder `configure()` 镜像构造；默认 PixelShuffle + concat skip；末端双线性对齐 |
| 整模 | `model_archi_complete.py` | `ModelComplete`：Encoder → GPT → Decoder |
| 训练封装 | `model_main.py` | LightningModule；AdamW + WarmupCosine；默认 L1（残差目标） |

**训练目标：** Dataset 返回 `target - input` 残差（因果对齐：输入右移一帧并拼上下一时刻真值）。解读 RMSE 时需注意反归一化与残差叠加方式。

默认实验规模（见 `configs.py`）：约 **126M** 参数量级（`emb≈1024`，`layers=6`，`heads=8`，`patch=(8,16)`，`resample_scale=2`，`max_seq_len_per_forward=2`）。

---

## 仓库结构

```
WeatherModel_opensource/
├── imports.py                 # 依赖与 Lightning 等导入
├── configs.py                 # 模型 / 数据 / 训练超参与路径
├── model_archi_encoder.py     # Encoder
├── model_archi_GPT.py         # GPT-style Transformer
├── model_archi_decoder.py     # Decoder
├── model_archi_complete.py    # 三段式组装 + 滑动窗推理示例
├── model_main.py              # LightningModule（优化器、L1/VAE loss）
├── dataset_define.py          # ERA5 Zarr Dataset（归一化、残差目标）
├── dataset_main.py            # Lightning DataModule
├── normalization.py           # 离线计算 mean/std → Zarr
├── trainer.py                 # DDP + 16-mixed 训练入口
└── custom_callbacks.py        # checkpoint 加载辅助
```

---

## 环境与依赖

依赖由 `imports.py` / `normalization.py` 实际导入推断；**仓库未提供固定 `requirements.txt` 版本锁定**（除注释中提到 Lightning ≈ 2.1.3）。建议自行准备与集群兼容的环境，例如：

| 包 | 用途 |
|----|------|
| `torch` | 模型与训练 |
| `pytorch-lightning` | 训练循环（注释：~2.1.3）；另用到 `lightning.pytorch.profilers` |
| `einops` | tensor rearrange |
| `xarray`, `zarr`, `dask` | ERA5 Zarr I/O |
| `numpy`, `pandas` | 时间索引与数组 |
| `tensorboard` | 日志 / Profiler |

安装示例（版本请按本机 CUDA / 集群策略选择）：

```bash
pip install torch pytorch-lightning einops xarray zarr dask numpy pandas tensorboard
```

将本目录加入 `PYTHONPATH`，或在该目录下直接运行脚本。

---

## 数据说明（需自行准备）

本仓库 **不提供** ERA5 原始数据或统一 Zarr 快照（体量可达数十 TB 量级）。使用者需自行从 [CDS](https://cds.climate.copernicus.eu/) 等渠道获取，并整理为与 `dataset_define.py` 兼容的 **多 group Zarr**。

### 期望变量与通道

对齐常见 Fuxi / FengWu 设定，共 **69** 通道：

- **气压层（5×13）**：`u` / `v` / `z` / `t` / `r`（相对湿度），13 层  
- **地面（4）**：`10m_u` / `10m_v` / `2m_t` / `msl`

Zarr group 名称需与 Dataset 一致，例如：

| 逻辑变量 | Zarr group |
|----------|------------|
| u / v / z / t / r | `u_component_of_wind`, `v_component_of_wind`, `geopotential`, `temperature`, `relative_humidity` |
| 地面 | `10m_u_component_of_wind`, `10m_v_component_of_wind`, `2m_temperature`, `mean_sea_level_pressure` |

网格默认对应 ERA5 全球 **721 × 1440**；训练时可经 `resample_scale` 最近邻下采样。时间步长按 **6 小时** 取样。

### 时间切分（代码现状）

| 阶段 | 大致区间 |
|------|----------|
| train | 1979–2009（6h）+ 2010–2015 |
| validation | 2016–2017 |
| testing | 2018–2020 |

注意：历史数据在 2009/2010 交界处分辨率可能不一致；`init_time` 需落在严格 6h 网格上，否则可能触发时间键缺失。

### 归一化

运行 `normalization.py`（先改其中的 Zarr 路径）在训练时段上按变量 / 气压层计算全局 mean、std，写入：

```text
<parent_of_era_zarr>/era5_norm_stats/norm_stats_<group>.zarr
```

`ERA_Dataset` 会按该约定自动加载。

在 `configs.py` 中设置：

```python
dir_ERA = "/path/to/your_era5_unified.zarr"
parent_path = "/path/to/experiment_root/"   # checkpoints / TensorBoard 输出根目录
```

---

## 如何运行

### 1. 冒烟：Dataset / 模型前向

```bash
# 需已准备好 Zarr 与 norm stats
python dataset_define.py

# 随机张量前向（不依赖真实数据；默认尺寸见脚本内）
python model_main.py
# 或
python model_archi_complete.py
```

### 2. 离线归一化

```bash
# 编辑 normalization.py 中的 fn 路径后：
python normalization.py
```

### 3. 多卡训练（主入口）

编辑 `configs.py`（序列长度、GPU 列表、`dir_ERA`、`parent_path`、`epoch` 等），然后：

```bash
python trainer.py
```

`trainer.py` 会：

- 构建 `Model_pl` + `ERA_Dataset_pl`
- 使用 **DDP**、`precision='16-mixed'`（当 `dtype=float16`）
- 按 epoch 保存 `latest_*` / 按 `valid_loss_epoch` 保存 `best_*`
- 可选 PyTorch Profiler（`enable_profiler=True`）

当前 `fit_or_predict == 'predict'` 分支为空；`predict_step` 亦未实现。正式 rollout / RMSE 评测需自行在外部脚本或 notebook 中完成。路径、设备列表、epoch 等均硬编码在 `configs.py`，开源克隆后务必改成本地路径。

---

## 状态与结果（如实说明）

本项目是 **研究原型**，不是可对外宣称的长时效 SOTA，也未与 Pangu / Fuxi 在同一公开协议下完成正式排行榜对照。相对工业级模型，训练 epoch / 数据吞吐明显不足。

已完成的部分：

- Encoder–GPT–Decoder 可训练骨架（RoPE、分块因果掩码、KV cache 接口）
- ERA5 Zarr Dataset、残差 L1、Lightning DDP + AMP + Checkpoint
- 约 126M 参数规模的集群级跑通；`max_seq_len=2` vs `6` 的初步对照

已知限制与负结果（摘要）：

| 问题 | 说明 |
|------|------|
| **训练与对照实验** | 后因博士主线挤占时间，本项目未充分训练；在已有实验中，固定参数量下 `maxstep=6` 相对 `maxstep=2` **未显示稳定优势**。 |
| **方块伪影** | 非重叠卷积压缩在长 rollout 上可见 block artifact |

---

## 引用与许可

独立研究作者：**郭金雨**。暂无正式论文 DOI；若引用本代码，请注明仓库与作者姓名，并说明其为研究原型。

许可建议：在仓库根目录补充 **MIT** 或 **Apache-2.0** `LICENSE` 文件后按所选协议使用。当前目录若尚未包含 LICENSE，请以作者后续声明为准；贡献与二次发布前请确认许可文本已就绪。

---

## 免责声明

本软件仅为 **科研原型**，用于方法探索与对照实验。**不得** 作为业务数值天气预报（NWP）或决策支持系统使用；不对预报准确性、时效性或任何衍生后果作保证。ERA5 等数据版权与使用条款归数据提供方所有，使用者须自行合规获取与引用。

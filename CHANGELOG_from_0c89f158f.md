# Changelog: 0c89f158f → HEAD (v6 分支)

基线提交：`0c89f158f` (support sybn for npu #13983)

总计：32 个文件，+6131 / -158 行。

---

## 一、Backbone（新增 & 修改）

### 1. `ppocr/modeling/backbones/rec_lcnetv4_det.py`（新增，1302 行）

PPLCNetV4 系列检测专用 backbone，基于 MetaFormer 架构（token_mixer + channel_mixer），全面支持重参数化。

| 组件 | 说明 |
|------|------|
| `NET_CONFIG_V4_DET` | 基础版通道配置：48→96→192→384，13 个 block |
| `NET_CONFIG_V4_DET_LARGE` | Large 版通道配置：128→256→512→896，对标 PPHGNetV2_B4（~22M 参数） |
| `Conv2D_BN` | 可融合的 Conv+BN 基础单元，`fuse()` 将 BN 吸收进 Conv |
| `RepDWConv` | 多分支重参数化 DW 卷积：3×3 DW+BN + 1×1 DW + Identity BN，推理时融合为单个 3×3 DW |
| `RepDWConvACN` | ACNet + MobileOne 风格增强版：在 RepDWConv 基础上增加 1×3 / 3×1 异形卷积分支和 K 个并行 3×3，捕获方向性边缘特征 |
| `Rep1x1Conv` | MobileOne 风格可重参数化 1×1 卷积：K 个并行 1×1 Conv+BN，推理时融合 |
| `StemBlock` | 双 3×3 Conv + MaxPool2D stem（移除了无效的 `ceil_mode=True`，stride=1+padding=SAME 下无影响） |
| `SEBlock` | Squeeze-and-Excitation 注意力模块 |
| `LCNetV4Block` | 基础 block：RepDWConv token_mixer + Conv2D_BN channel_mixer + 可选 SE |
| `LCNetV4ACNBlock` | ACN 增强 block：使用 RepDWConvACN + Rep1x1Conv，推理时参数量与 LCNetV4Block 一致 |

**Backbone 变体：**

| 类名 | 通道配置 | Stem | 用途 |
|------|---------|------|------|
| `PPLCNetV4_det` | 48→96→192→384 | mid=24, out=48 | 基础轻量版 |
| `PPLCNetV4_det_tiny` | 24→48→96→192 | mid=12, out=24 | 超轻量版（~405K 参数） |
| `PPLCNetV4_det_acn` | 48→96→192→384 | mid=24, out=48 | ACNet+MobileOne 增强版 |
| `PPLCNetV4_det_acn_tiny` | 24→48→96→192 | mid=12, out=24 | ACN 超轻量版 |
| `PPLCNetV4_det_v7b_tiny` | 24→48→96→192 | mid=12, out=24 | V7B block 结构变体 |
| `PPLCNetV4_det_v7b_tiny_v2` | 48→96→160 | mid=24, out=48 | V7B 通道调整版 |
| `PPLCNetV4_det_large` | 128→256→512→896 | mid=64, out=128 | Large 版，匹配 PPHGNetV2_B4 参数量 |

所有变体共享 `out_indices=[2,5,10,13]`，输出 4 级特征图。所有变体均实现统一的 `rep()` + `is_repped` 重参数化接口。

### 2. `ppocr/modeling/backbones/det_pphgnetv2.py`（新增，683 行）

移植 PPHGNetV2_B4 backbone，用于与 PPLCNetV4_det_large 对比实验。

### 3. `ppocr/modeling/backbones/rec_repvit.py`（修改，+29 行）

为 RepSVTR_det backbone 新增 `rep()` / `is_repped` 接口，统一重参数化调用方式。

### 4. `ppocr/modeling/backbones/rec_lcnetv3.py`（修改，+9 行）

为 PPLCNetV3 backbone 新增 `rep()` / `is_repped` 接口。

### 5. `ppocr/modeling/backbones/__init__.py`（修改，+16 行）

新增 8 个 backbone 的 import 和 support_dict 注册：`PPHGNetV2_B4`、`PPLCNetV4_det`、`PPLCNetV4_det_tiny`、`PPLCNetV4_det_acn`、`PPLCNetV4_det_acn_tiny`、`PPLCNetV4_det_v7b_tiny`、`PPLCNetV4_det_v7b_tiny_v2`、`PPLCNetV4_det_large`。

---

## 二、Neck（新增）

### 1. `ppocr/modeling/necks/db_fpn.py`（修改，+615 行）

新增 2 个 Neck 和 2 个基础模块：

#### 基础模块

| 模块 | 说明 |
|------|------|
| `DilatedReparamBlock` | UniRepLKNet 的多分支膨胀 DW 卷积。训练时并行多个膨胀分支（如 kernel=9 时：9×9 DW + 5×5 dil1 + 5×5 dil2 + 3×3 dil3 + 3×3 dil4），推理时 `rep()` 融合为单个大核 DW Conv |
| `DilatedReparamConv` | DW(DilatedReparam) + PW(1×1) + BN 的组合，作为标准大核 Conv 的 drop-in 替换。参数量减少约 96%，保持相同感受野 |

#### Neck 变体

| Neck | 结构 | 相比原版改动 | 参数变化 |
|------|------|------------|---------|
| `UniRepLKPAN` | 用 DilatedReparamConv 替换 LKPAN 的 8 个 9×9 标准 Conv（4 inp_conv + 4 pan_lat_conv） | 9×9 标准 Conv → DW 9×9 reparam + PW 1×1 | 9×9 部分从 6.6M → 274K（**-96%**） |
| `UniRepLKFPN` | DilatedReparamBlock(DW 7×7) + PW + SE 替换 RSEFPN 的 inp_conv | 3×3 标准 Conv → DW 7×7 reparam + PW | inp_conv 减少 **65%**，感受野 3→7 |

所有新 Neck 均：
- 支持 `intracl=True`（IntraCLBlock 上下文学习模块）
- 训练时返回 `{'fuse': ..., 'aux_p4': ..., 'aux_p3': ..., 'aux_p2': ...}` 支持辅助监督
- 实现 `rep()` + `is_repped` 接口

### 2. `ppocr/modeling/necks/gfpn.py`（新增，534 行）

新增 `GPAN` 和 `GFPN` 两个实验性 Neck。

### 3. `ppocr/modeling/necks/__init__.py`（修改，+9 行）

注册 `UniRepLKPAN`、`UniRepLKFPN`、`GPAN`、`GFPN`。

---

## 三、Head（修改）

### `ppocr/modeling/heads/det_db_head.py`（修改，+284 行）

#### 1. `Head` 类改造

- 新增 `rep()` 方法：融合 Conv2D+BatchNorm 和 Conv2DTranspose+BatchNorm 为带 bias 的单层算子
- 新增 `_fuse_conv_bn()` / `_fuse_convtranspose_bn()` 静态工具方法

#### 2. `DBHead` 改造

| 改动 | 说明 |
|------|------|
| `aux_in_channels` 参数 | 每个尺度（p2/p3/p4）创建独立的 aux binarize + aux thresh Head，实现深监督 |
| `forward()` | 兼容 neck 返回 dict（训练，含 aux 特征）或 tensor（推理）；训练时对 aux_p2/p3/p4 分别生成 aux_maps |
| `rep()` | 递归 rep 所有子 Head（含 aux heads） |

#### 3. `PFHeadLocal`

保持原始实现不变（无 aux 辅助分支改动）。

---

## 四、Loss（新增 & 修改）

### 1. `ppocr/losses/det_basic_loss.py`（修改，+221 行）

| Loss | 说明 |
|------|------|
| `MaskedFocalLoss` | 带 mask 的 Focal Loss：`FL(p_t) = -α_t(1-p_t)^γ log(p_t)`，自适应 hard-example 加权，替代 OHEM 的硬选择策略 |
| `DiceBCELoss` | Dice + Focal 组合：Dice 优化全局 F1/overlap，Focal 提供像素级 hard-example 监督，互补 |

### 2. `ppocr/losses/det_db_loss.py`（修改，+66 行）

| 改动 | 说明 |
|------|------|
| `main_loss_type` 扩展 | 支持 `'DiceBCELoss'` / `'DiceLoss'` 两种选择 |
| 新增参数 | `focal_alpha`、`focal_gamma`、`dice_weight`、`focal_weight` 等 |
| `aux_weight_p2/p3/p4` | 辅助 loss 权重，对 aux_maps_p2/p3/p4 各自计算完整 DB loss 并加权 |

---

## 五、Optimizer（修改）

### 1. `ppocr/optimizer/regularizer.py`（修改，+29 行）

新增 `CosineL2Decay`：weight decay 余弦退火调度器。

- 从 `factor`（初始 WD）退火到 `end_factor`（最终 WD），支持 warmup
- 避免小模型在训练后期被过度正则化
- 参考 EfficientNetV2 的 WD annealing 策略

### 2. `ppocr/optimizer/__init__.py`（修改，+59 行）

| 改动 | 说明 |
|------|------|
| `CosineWeightDecayScheduler` | 每个训练 step 动态调整 WD：`wd(t) = end + 0.5*(start-end)*(1+cos(πt/T))`，warmup 期间保持初始值 |
| `build_optimizer` | 返回值从 2 元组 `(optimizer, lr_scheduler)` 改为 3 元组 `(optimizer, lr_scheduler, wd_scheduler)` |

---

## 六、数据增强（修改 & 新增）

### 1. `ppocr/data/imaug/iaa_augment.py`（重写，+225 行）

- **imgaug → albumentations 全面迁移**：消除对已废弃 imgaug 库的依赖
- 实现 `ImgaugLikeResize` 等自定义变换，保持与原始增强行为的兼容性
- 支持 albumentations 新旧版本 API

### 2. `ppocr/data/imaug/make_border_map.py`（修改）

保持原始 `MakeBorderMap` 不变。

### 3. `ppocr/data/imaug/make_shrink_map.py`（修改）

保持原始 `MakeShrinkMap` 不变。

### 4. `ppocr/data/imaug/random_crop_data.py`（重构，+337 行）

重构 `EastRandomCropData`：更鲁棒的随机裁切逻辑，更好地保留文本区域。

### 5. `ppocr/data/imaug/operators.py`（修改，+9 行）

配合增强改动的算子调整。

### 6. `ppocr/data/imaug/__init__.py`（修改）

更新导入。

---

## 七、Tools（修改）

| 文件 | 改动 | 说明 |
|------|------|------|
| `tools/export_model.py` | +9 行 | 导出时遍历所有 sublayers 调用统一 `rep()` 接口，确保所有可重参数化模块融合 |
| `tools/program.py` | +8 行 | 适配 `build_optimizer` 返回 3 元组；训练循环中每步调用 `wd_scheduler.step()` |
| `tools/train.py` | +3 行 | 解包 `wd_scheduler`，传递到训练循环 |
| `deploy/slim/prune/sensitivity_anal.py` | 1 行 | `build_optimizer` 返回值适配（3 元组解包） |
| `deploy/slim/quantization/quant.py` | 1 行 | 同上 |

---

## 八、配置文件（新增）

| 文件 | Backbone | Neck | Head | 特殊配置 |
|------|----------|------|------|---------|
| `mobile_config_exp.yml` | RepSVTR_det | UniRepLKFPN(96) | DBHead(rep_conv1, aux) | AMP、DiceBCELoss |
| `ocr_detV4_large.yml` | PPLCNetV4_det_large | UniRepLKPAN(256, intracl) | PFHeadLocal(large) | CosineL2Decay |
| `mobile_dwfpn_pplcnetV4.yml` | PPLCNetV4_det | UniRepLKFPN(96) | DBHead | - |
| `mobile_dwfpn_pplcnetV4_tiny.yml` | PPLCNetV4_det_tiny | UniRepLKFPN(96) | DBHead | - |
| `mobile_dwfpn_pplcnetV4_acn.yml` | PPLCNetV4_det_acn | UniRepLKFPN(96) | DBHead | ACNet+MobileOne |
| `mobile_dwfpn_pplcnetV4_acn_tiny.yml` | PPLCNetV4_det_acn_tiny | UniRepLKFPN(96) | DBHead | ACNet+MobileOne |
| `mobile_dwfpn_pplcnetV4_v7b_tiny.yml` | PPLCNetV4_det_v7b_tiny | UniRepLKFPN(96) | DBHead | V7B block |
| `mobile_dwfpn_pplcnetV4_v7b_tiny_v2.yml` | PPLCNetV4_det_v7b_tiny_v2 | UniRepLKFPN(96) | DBHead | V7B v2 |
| `official_v5_server_aux.yml` | PPHGNetV2_B4 | LKPAN(256, intracl) | PFHeadLocal(large, aux) | 辅助监督 |

---

## 核心设计主线

1. **PPLCNetV4 backbone 系列**：从 tiny（~405K）到 large（~22M），统一 rep 接口，推理时多分支融合为单卷积零开销
2. **高效 Neck**：UniRepLKPAN（参数 -96%）、UniRepLKFPN（-65%），保持大感受野的同时大幅降低参数和计算量
3. **Loss 增强**：DiceBCELoss / TverskyFocalLoss 替代 OHEM+Dice，自适应 hard-example 加权 + 辅助深监督
4. **数据增强现代化**：imgaug→albumentations 迁移、cosine weight decay 退火

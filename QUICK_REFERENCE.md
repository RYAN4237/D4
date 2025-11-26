# 快速使用指南

## 🎯 核心改进对应关系

### 问题1：边角重叠畸形 → 解决方案

| 问题细节 | 解决方案 | 代码位置 |
|---------|---------|---------|
| 整个小地图直接覆盖 | 增量拼接（只复制新增区域） | `stitch()` 方法，confidence > 0.85分支 |
| 边界像素不对齐 | 特征匹配+RANSAC校正 | `stitch()` 方法，confidence ≤ 0.85分支 |
| 色差线明显 | 高斯权重渐变融合 | `_blend_region()` 方法 |

### 问题2：has_moved漏拼 → 解决方案

| 问题根因 | 解决方案 | 代码位置 |
|---------|---------|---------|
| 硬编码阈值不适应光照变化 | 自适应mean_diff阈值 | `detect_movement()`，第二层 |
| 0.5像素抖动导致误判 | Kalman滤波平滑 | `detect_movement()`，第四层 |
| 多条件不协调 | 多条件AND逻辑 | `detect_movement()`，第五层 |
| 阈值不区分状态 | 状态机 + 动态阈值 | `detect_movement()`，第六层和`_get_response_threshold()` |

---

## 📝 关键参数说明

### 1. 自适应阈值计算

```python
# 基础阈值 = baseline + 倍数 × 标准差
baseline = np.mean(last_30_frames_mean_diff)  # 最近30帧平均
std_dev = np.std(last_30_frames_mean_diff)    # 最近30帧标准差

# 倍数根据状态调整
IDLE:    1.5x std  # 严格（防止误触）
MOVING:  0.8x std  # 宽松（快速响应）
STOPPED: 1.2x std  # 平衡
```

**调优建议**：
- 如果漏检过多：降低倍数（1.5→1.3, 0.8→0.6）
- 如果误检过多：提高倍数（0.8→1.0, 1.2→1.5）

---

### 2. Kalman滤波权重

```python
smooth_dx = 0.7 * prev_dx + 0.3 * new_dx
smooth_dy = 0.7 * prev_dy + 0.3 * new_dy
```

**权重含义**：
- 70% 历史值 + 30% 新值 → 响应较慢但平稳
- 调整为 60% + 40% → 响应更快但更敏感
- 调整为 80% + 20% → 响应更慢但更稳定

---

### 3. 状态机转移条件

```python
IDLE → MOVING:
  条件：4个检测条件全部满足（has_moved=True）
  置信度要求：response > 0.80

MOVING → STOPPED:
  条件：连续3帧无移动（no_move_frames >= 3）
  自动转移，无需额外条件

STOPPED → IDLE:
  条件：连续5帧无移动（no_move_frames >= 5）
  自动转移，无需额外条件

STOPPED → MOVING:
  条件：检测到移动，但要求较低
  置信度要求：response > 0.70
```

**调优建议**：
- 转移太频繁：增加帧数阈值（3→5, 5→8）
- 转移太缓慢：减少帧数阈值（3→2, 5→3）

---

### 4. 拼接置信度阈值

```python
if confidence > 0.85:
    # 增量拼接（高效）
else:
    # 特征匹配对齐（精确）
```

**调优建议**：
- 如果特征匹配耗时过长：提高阈值（0.85→0.90）
- 如果边界畸形明显：降低阈值（0.85→0.80）

---

## 🔍 运行时调试

### 观察的关键指标

```
[001] 移动: (+12.3, -5.6), 位置: (1500, 1502), 置信度: 0.892, 状态: MOVING
      ↓     ↓                    ↓              ↓             ↓
     序号  平滑后位移         画布中心       confidence    当前状态
```

### 异常情况诊断

| 现象 | 可能原因 | 解决方案 |
|------|--------|--------|
| 拼接率过低（<5%） | 阈值过高 | 降低MOVING状态的response阈值（0.60→0.55） |
| 频繁出现"超出画布范围"警告 | 位移估计有偏差 | 检查特征匹配是否失败，或降低拼接置信度阈值 |
| 边界出现明显色差线 | 渐变融合不足 | 检查`_blend_region()`中的权重计算 |
| 拼接位置不准确 | 相位相关误差大 | 检查是否在UI变化时运行，或提高特征匹配阈值 |
| 状态频繁闪烁（MOVING↔STOPPED） | 阈值设置不当 | 增加转移帧数阈值 |

---

## 🎮 实际使用流程

### 第一次运行

```python
from stitch import SmartMinimapStitcher

# 初始化（调整截图区域为实际小地图位置）
stitcher = SmartMinimapStitcher(x1=100, y1=100, x2=300, y2=300)

# 运行
stitcher.run()

# 观察输出，调整参数（如果拼接率不理想）
```

### 参数微调

```python
# 如果拼接率过低，修改detect_movement()中的阈值
def _get_response_threshold(self):
    thresholds = {
        "IDLE": 0.75,    # ← 从0.80降低到0.75
        "MOVING": 0.55,  # ← 从0.60降低到0.55
        "STOPPED": 0.70,
    }
    return thresholds.get(self.state, 0.70)
```

### 监控文件保存

程序会自动保存：
- `final_smart_stitch.png` - 完整画布
- `final_smart_stitch_cropped.png` - 去除边框的裁剪版
- `stitched_smart_<时间戳>.png` - 按S键手动保存

---

## 📊 性能指标参考

### 预期拼接率

| 场景 | 预期拼接率 | 备注 |
|------|-----------|------|
| 缓慢移动 | 15-25% | 每移动3-5像素拼接一次 |
| 正常速度 | 20-30% | 预期的推荐运行环境 |
| 快速移动 | 10-20% | 受相位相关算法限制 |

### 计算性能

| 操作 | 耗时 |
|------|------|
| 相位相关检测 | ~5ms |
| ORB特征检测 | ~20ms |
| 特征匹配 | ~10ms |
| RANSAC校正 | ~5ms |

**总体帧率**：~30-50 FPS（取决于是否触发特征匹配）

---

## ⚙️ 高级配置

### 禁用特征匹配（性能优先）

```python
# 在stitch()中修改
if confidence > 0.75:  # 提高阈值，减少特征匹配触发频率
    # ...增量拼接
else:
    # ...特征匹配
```

### 禁用状态机（简化模式）

```python
# 在detect_movement()中，直接返回has_moved，跳过状态转移
if has_moved:
    return True, smooth_dx, smooth_dy, smooth_response, "MOVING"
else:
    return False, 0, 0, smooth_response, "IDLE"
```

### 启用详细日志

```python
# 在detect_movement()的多条件验证前添加
print(f"Debug: mean_diff={mean_diff:.2f}, threshold={adaptive_threshold:.2f}")
print(f"Debug: conditions={conditions}")
```

---

## 💡 最佳实践

1. **第一次运行不要调参**：先用默认参数跑一遍，观察输出
2. **只改一个参数**：每次只调整一个参数，观察效果
3. **用数据说话**：看拼接率和视觉效果，而不是猜测
4. **保存中间结果**：按S键多次保存，对比边界质量
5. **逐步优化**：从改进1（自适应阈值）开始，再到4（状态机）

---

## 🆘 常见问题

**Q: 程序启动后黑屏？**  
A: 检查截图区域(x1, y1, x2, y2)是否正确对应小地图位置

**Q: 拼接出现严重畸形？**  
A: 这是边界重叠问题。确保运行的是最新版本，并检查相位相关是否失效

**Q: 能否进一步加快运行速度？**  
A: 可以禁用低置信度时的特征匹配，或降低ORB_create的nfeatures参数

**Q: 状态机不起作用？**  
A: 检查detect_movement()的返回值中是否正确传入state参数



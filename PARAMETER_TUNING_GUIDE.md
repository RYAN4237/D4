# 参数调试速查表

## 🎯 一键诊断和调优

### 问题1：拼接率过低 （< 5%）

**诊断**：
```
观察日志
拼接次数少 → 大多数帧被判定为无移动
```

**逐步调整**：

```python
# 步骤1：降低MOVING状态的置信度阈值
def _get_response_threshold(self):
    thresholds = {
        "IDLE": 0.75,    # ← 从0.80改为0.75
        "MOVING": 0.55,  # ← 从0.60改为0.55
        "STOPPED": 0.70,
    }
    return thresholds.get(self.state, 0.70)
```

```python
# 步骤2：如果仍然过低，降低自适应阈值倍数
if self.state == "MOVING":
    adaptive_threshold = baseline + 0.6 * std_dev  # ← 从0.8改为0.6
```

```python
# 步骤3：降低最小移动阈值
if abs(smooth_dx) >= 0.3 or abs(smooth_dy) >= 0.3:  # ← 从0.5改为0.3
    conditions["min_movement"] = True
```

```python
# 步骤4：如果还是不行，扩大最大位移限制
if abs(dx) <= 150 and abs(dy) <= 150:  # ← 从100改为150
    conditions["movement_valid"] = True
```

**最激进调整**（仅在上述都无效时使用）：
```python
# 禁用min_movement检查
conditions = {
    "mean_diff": mean_diff > adaptive_threshold,
    # "min_movement": abs(smooth_dx) >= 0.5 or abs(smooth_dy) >= 0.5,  ← 注释掉
    "confidence": smooth_response > self._get_response_threshold(),
    "movement_valid": abs(dx) <= 100 and abs(dy) <= 100,
}
has_moved = all(conditions.values())
```

---

### 问题2：拼接率过高 （> 50%，出现误拼）

**诊断**：
```
观察日志
UI闪动时频繁拼接 → 误检
没有移动时也拼接 → 阈值过低
```

**逐步调整**：

```python
# 步骤1：提高IDLE状态的置信度阈值（防止静止时误触）
def _get_response_threshold(self):
    thresholds = {
        "IDLE": 0.85,    # ← 从0.80改为0.85
        "MOVING": 0.65,  # ← 从0.60改为0.65
        "STOPPED": 0.75, # ← 从0.70改为0.75
    }
    return thresholds.get(self.state, 0.70)
```

```python
# 步骤2：提高自适应阈值倍数（对光照变化更严格）
if self.state == "IDLE":
    adaptive_threshold = baseline + 2.0 * std_dev  # ← 从1.5改为2.0

if self.state == "MOVING":
    adaptive_threshold = baseline + 1.0 * std_dev  # ← 从0.8改为1.0
```

```python
# 步骤3：增加状态转移的帧数需求（防止频繁切换）
if self.state == "MOVING" and self.no_move_frames >= 5:  # ← 从3改为5
    self.state = "STOPPED"

if self.state == "STOPPED" and self.no_move_frames >= 8:  # ← 从5改为8
    self.state = "IDLE"
```

```python
# 步骤4：提高min_movement阈值
if abs(smooth_dx) >= 1.0 or abs(smooth_dy) >= 1.0:  # ← 从0.5改为1.0
    conditions["min_movement"] = True
```

**最保守调整**（仅在上述都无效时使用）：
```python
# 禁用mean_diff检查，完全依靠相位相关
conditions = {
    # "mean_diff": mean_diff > adaptive_threshold,  ← 注释掉
    "min_movement": abs(smooth_dx) >= 0.5 or abs(smooth_dy) >= 0.5,
    "confidence": smooth_response > self._get_response_threshold(),
    "movement_valid": abs(dx) <= 100 and abs(dy) <= 100,
}
```

---

### 问题3：边界出现明显色差线

**诊断**：
```
观察图像
拼接处有突兀的颜色突变 → 融合不足
边界模糊 → 融合过度
```

**调整渐变融合**：

```python
def _blend_region(self, new_region, old_region, confidence, dx=0, dy=0):
    h, w = new_region.shape[:2]
    y, x = np.ogrid[:h, :w]

    if dx != 0:
        dist_x = np.minimum(x + 1, w - x)
        max_dist = w // 3  # ← 从w//4改为w//3（更大的融合区域）
        weight_x = np.clip(dist_x / max_dist, 0, 1)
    else:
        weight_x = np.ones((h, w))

    if dy != 0:
        dist_y = np.minimum(y + 1, h - y)
        max_dist = h // 3  # ← 从h//4改为h//3
        weight_y = np.clip(dist_y / max_dist, 0, 1)
    else:
        weight_y = np.ones((h, w))

    # ✅ 关键：根据confidence调整融合度
    # 如果色差还是明显，可以增加confidence的权重
    weight = (weight_x * weight_y) * (confidence ** 2) + (1 - confidence) * 0.5
    #                                              ↑ 平方可以增强confidence的影响
    weight = np.stack([weight] * 3, axis=2)

    blended = (new_region.astype(np.float32) * weight +
               old_region.astype(np.float32) * (1 - weight)).astype(np.uint8)
    return blended
```

**特殊情况：禁用融合（直接对比新旧效果）**

```python
# 在stitch()中临时禁用融合
if (dx != 0 or dy != 0) and confidence > 0.75:
    # 直接覆盖，看效果如何
    self.canvas[canvas_top:canvas_bottom, canvas_left:canvas_right] = new_region
else:
    self.canvas[canvas_top:canvas_bottom, canvas_left:canvas_right] = new_region
```

---

### 问题4：拼接位置不准确，出现错位

**诊断**：
```
观察图像
拼接的地物错位2-5像素 → 相位相关有偏差
拼接完全偏离 → 特征匹配失败
```

**步骤1：确认是否触发特征匹配**

```python
# 在stitch()中加入调试日志
print(f"拼接置信度: {confidence:.3f}, 触发特征匹配: {confidence <= 0.85}")
```

**如果经常触发特征匹配但仍有误差**：

```python
# 降低拼接置信度阈值，减少特征匹配的需要
if confidence > 0.80:  # ← 从0.85改为0.80
    # 增量拼接
else:
    # 特征匹配
```

**如果特征匹配成功但仍有误差**：

```python
# 降低RANSAC的鲁棒性阈值（允许更多离群点）
# 或增加特征点数量
self.orb = cv2.ORB_create(nfeatures=800)  # ← 从500改为800
```

**激进方案：禁用特征匹配，依赖相位相关**

```python
# 在stitch()中，总是使用简单拼接
def stitch(self, minimap, last_minimap, dx, dy, confidence):
    # 直接跳过特征匹配
    return self._simple_stitch(minimap, dx, dy)
```

---

### 问题5：状态机频繁闪烁（MOVING ↔ STOPPED）

**诊断**：
```
观察日志
频繁切换状态 → 转移阈值设置不当
无法从MOVING进入STOPPED → no_move_frames阈值太高
```

**调整状态转移阈值**：

```python
# 在detect_movement()中，修改转移条件
if self.state == "MOVING" and self.no_move_frames >= 5:  # ← 从3改为5
    self.state = "STOPPED"

if self.state == "STOPPED" and self.no_move_frames >= 10:  # ← 从5改为10
    self.state = "IDLE"
```

**或者调整置信度累积**：

```python
# 添加一个累积置信度的概念
if has_moved:
    self.state_confidence = min(1.0, self.state_confidence + 0.2)  # ← 从0.1改为0.2
    if self.state_confidence > 0.8 and self.state == "STOPPED":
        self.state = "MOVING"

if not has_moved:
    self.state_confidence = max(0.0, self.state_confidence - 0.1)
    if self.state_confidence < 0.3 and self.state == "MOVING":
        self.state = "STOPPED"
```

---

## 📋 参数速查总表

### 核心参数

| 参数 | 位置 | 默认值 | 范围 | 说明 |
|------|------|--------|------|------|
| `mean_diff_history.maxlen` | `__init__` | 30 | 10-60 | 越大越平稳，但响应慢 |
| `movement_history.maxlen` | `__init__` | 5 | 2-10 | 历史帧数 |
| Kalman dx权重 | `detect_movement` | 0.7 | 0.5-0.9 | 越高越平稳 |
| Kalman response权重 | `detect_movement` | 0.6 | 0.4-0.8 | 越高越平稳 |
| 自适应阈值倍数(IDLE) | `detect_movement` | 1.5 | 0.8-2.5 | 越大越严格 |
| 自适应阈值倍数(MOVING) | `detect_movement` | 0.8 | 0.4-1.5 | 越大越严格 |
| 最小移动阈值 | `detect_movement` | 0.5 | 0.1-1.0 | 越小越灵敏 |
| 最大移动限制 | `detect_movement` | 100 | 50-200 | 越大越容许快速移动 |
| 拼接置信度阈值 | `stitch` | 0.85 | 0.7-0.95 | 越高越依赖增量拼接 |
| 融合启动置信度 | `stitch` | 0.75 | 0.6-0.9 | 越低越容易融合 |
| 融合边界宽度(x方向) | `_blend_region` | w//4 | w//6-w//2 | 越大融合区越大 |
| ORB特征点数 | `__init__` | 500 | 200-1000 | 越多越精确但慢 |
| 转移到STOPPED的帧数 | `detect_movement` | 3 | 2-10 | 越小切换越快 |
| 转移到IDLE的帧数 | `detect_movement` | 5 | 3-15 | 越小切换越快 |

### 置信度阈值表

| 状态 | IDLE | MOVING | STOPPED | 说明 |
|------|------|--------|---------|------|
| response阈值 | 0.80 | 0.60 | 0.70 | 相位相关置信度 |
| mean_diff倍数 | 1.5x | 0.8x | 1.2x | 标准差倍数 |
| 调整建议(严格) | 0.85 | 0.65 | 0.75 | 减少误检 |
| 调整建议(宽松) | 0.75 | 0.55 | 0.65 | 增加拼接率 |

---

## 🔧 常用调优配置预设

### 预设1：高精度模式（精确拼接，牺牲拼接率）

```python
def _get_response_threshold(self):
    return {
        "IDLE": 0.85,
        "MOVING": 0.70,
        "STOPPED": 0.80,
    }.get(self.state, 0.75)

# 在detect_movement()中
if self.state == "IDLE":
    adaptive_threshold = baseline + 2.0 * std_dev
elif self.state == "MOVING":
    adaptive_threshold = baseline + 1.2 * std_dev

# 在__init__中
self.mean_diff_history = deque(maxlen=50)  # 更多历史
```

**特点**：
- ✅ 拼接精度高
- ❌ 拼接率低（5-15%）
- ✅ 误检率极低

---

### 预设2：高效率模式（快速拼接，适度精度）

```python
def _get_response_threshold(self):
    return {
        "IDLE": 0.75,
        "MOVING": 0.55,
        "STOPPED": 0.65,
    }.get(self.state, 0.65)

# 在detect_movement()中
if self.state == "IDLE":
    adaptive_threshold = baseline + 1.2 * std_dev
elif self.state == "MOVING":
    adaptive_threshold = baseline + 0.6 * std_dev

# 增加Kalman新值权重
smooth_dx = 0.6 * self.smooth_dx + 0.4 * dx
```

**特点**：
- ❌ 拼接精度中等
- ✅ 拼接率高（25-40%）
- ⚠️ 误检率中等

---

### 预设3：平衡模式（推荐，精度和效率均衡）

```python
# 保持默认值即可！
```

**特点**：
- ✅ 拼接精度好
- ✅ 拼接率中等（20-30%）
- ✅ 误检率低

---

## 📊 调参前后对比模板

```
【调整前】
置信度阈值: IDLE=0.80, MOVING=0.60, STOPPED=0.70
自适应倍数: IDLE=1.5x, MOVING=0.8x, STOPPED=1.2x
拼接率: 5%
误检率: 30%
边界质量: 有色差线
总体评分: ⭐⭐⭐

【调整后】
置信度阈值: IDLE=0.75, MOVING=0.55, STOPPED=0.65
自适应倍数: IDLE=1.3x, MOVING=0.7x, STOPPED=1.0x
拼接率: 22%
误检率: 8%
边界质量: 无色差线
总体评分: ⭐⭐⭐⭐⭐

【改进总结】
✅ 拼接率提升4倍
✅ 误检率降低73%
✅ 边界质量显著改善
```

---

## 🎯 调参建议流程

1. **第一步**：不调参，运行默认设置，记录基准数据
2. **第二步**：根据问题诊断表，找到对应问题
3. **第三步**：按照推荐步骤，一次只改一个参数
4. **第四步**：运行5分钟，记录拼接率和视觉效果
5. **第五步**：对比改进，决定是否继续调整
6. **第六步**：保存最优配置为预设

---

## 💾 参数导出/导入

```python
# 导出当前参数配置
import json

config = {
    "response_thresholds": {
        "IDLE": 0.80,
        "MOVING": 0.60,
        "STOPPED": 0.70,
    },
    "adaptive_multipliers": {
        "IDLE": 1.5,
        "MOVING": 0.8,
        "STOPPED": 1.2,
    },
    "kalman_weights": {
        "dx_dy": 0.7,
        "response": 0.6,
    },
}

# 保存
with open("config.json", "w") as f:
    json.dump(config, f, indent=2)

# 恢复
with open("config.json", "r") as f:
    config = json.load(f)
    # 在__init__中应用这些配置
```



# Unity 小游戏 — 子弹躲避机 🛩️

## 游戏简介

玩家驾驶一架小飞机，在不断升级的子弹雨和随机怪物中生存。  
每过一分钟会随机掉落奖励道具。  
碰到任何子弹或怪物即游戏结束。

---

## 功能列表

| 功能 | 说明 |
|------|------|
| 四面八方子弹 | 8 个方向同时发射，带随机扩散角 |
| 子弹加速 | 随时间推移，子弹速度与生成频率持续提升 |
| 随机怪物 | 怪物从屏幕边缘随机出现，缓慢追踪玩家 |
| 碰撞即死 | 撞到子弹或怪物立即 Game Over |
| 每分钟奖励 | 60 秒触发一次，随机掉落以下道具之一 |

### 奖励道具

| 图标 | 名称 | 效果 |
|------|------|------|
| 🛡 | Shield（护盾） | 抵挡一次伤害，持续 5 秒 |
| ⚡ | Speed Boost（加速） | 飞机移速 +3，持续 8 秒 |
| 🐢 | Slow Bullets（减速弹） | 子弹速度减半，持续 8 秒 |
| 💯 | Score Bonus（分数奖励） | 立即 +100 分 |
| 💥 | Clear Screen（清屏） | 消灭当前所有子弹和怪物 |

---

## 目录结构

```
unity_game/
└── Assets/
    └── Scripts/
        ├── GameManager.cs       # 游戏状态、计时、奖励触发
        ├── PlayerController.cs  # 玩家飞机移动与碰撞
        ├── BulletSpawner.cs     # 8 方向子弹生成
        ├── Bullet.cs            # 子弹运动逻辑
        ├── MonsterSpawner.cs    # 随机怪物生成
        ├── Monster.cs           # 怪物追踪逻辑
        ├── RewardSpawner.cs     # 奖励道具生成
        ├── RewardItem.cs        # 奖励效果定义
        └── UIManager.cs         # HUD / Game-Over 界面
```

---

## 在 Unity 中导入步骤

### 1. 创建项目
1. 打开 **Unity Hub** → New Project → **2D (URP)** 模板, Unity 版本建议 **2022.3 LTS** 或以上。
2. 项目名随意，例如 `BulletDodge`。

### 2. 拷贝脚本
将 `unity_game/Assets/Scripts/` 下的所有 `.cs` 文件复制到 Unity 项目的 `Assets/Scripts/` 目录。

### 3. 创建预制体（Prefabs）

#### 玩家飞机 (Player)
1. 创建 2D Sprite（三角形或飞机贴图），命名 `Player`。
2. 添加组件：`PlayerController`、`Rigidbody2D`（Gravity Scale = 0）、`Collider2D`（Is Trigger = ✅）。
3. 设置 Tag = `Player`。

#### 子弹 (Bullet)
1. 创建细长 Sprite，命名 `Bullet`。
2. 添加 `Bullet`、`Rigidbody2D`（Gravity Scale = 0）、`Collider2D`（Is Trigger = ✅）。
3. Tag = `Bullet`，保存为 Prefab。

#### 怪物 (Monster) — 可创建多种
1. 创建方形/圆形 Sprite，命名 `Monster_A`（可以复制多份改颜色）。
2. 添加 `Monster`、`Rigidbody2D`（Gravity Scale = 0）、`Collider2D`（Is Trigger = ✅）。
3. Tag = `Monster`，保存为 Prefab。

#### 奖励道具 (Rewards)
分别为 5 种奖励各创建一个 Prefab，每个添加 `RewardItem`，设置对应的 `Reward Type`，Tag = `Reward`，Is Trigger = ✅。

### 4. 创建游戏场景 (Scene)

1. 新建 Scene，保存为 `GameScene`。
2. **Camera** — 设置为 Orthographic，Size = 5。

#### 创建空对象并挂脚本

| 游戏对象名 | 组件 |
|-----------|------|
| `GameManager` | `GameManager` |
| `BulletSpawner` | `BulletSpawner` |
| `MonsterSpawner` | `MonsterSpawner` |
| `RewardSpawner` | `RewardSpawner` |

在 `GameManager` 的 Inspector 中，将上述引用字段分别拖入对应对象。

#### 创建 UI (Canvas)

使用 **UI > Canvas**（Screen Space Overlay）添加以下元素：

| 元素 | 对应字段 |
|------|---------|
| `Text (TMP)` — 左上角 | `scoreText` |
| `Text (TMP)` — 左上第二行 | `timeText` |
| `Text (TMP)` — 左上第三行 | `speedText` |
| `Text (TMP)` — 屏幕中央 | `rewardText`（初始隐藏） |
| `Panel` — 居中 | `gameOverPanel`（初始隐藏） |
| `Text (TMP)` — Panel 内 | `finalScoreText` |
| `Button` — Panel 内 | `restartButton`（文字"Restart"） |

将 `UIManager` 脚本挂到 Canvas 对象，并在 Inspector 中拖入上述所有引用。

### 5. 运行游戏

按 **Play** 即可。  
- **WASD / 方向键** 移动飞机  
- 躲避子弹和怪物  
- 收集屏幕上的奖励道具（触碰即拾取）  

---

## 参数调节

所有核心参数均暴露在 Inspector 中，无需修改代码：

| 脚本 | 参数 | 作用 |
|------|------|------|
| `GameManager` | `baseBulletSpeed` | 初始子弹速度 |
| `GameManager` | `speedIncreasePerSecond` | 每秒加速量 |
| `GameManager` | `maxBulletSpeed` | 子弹速度上限 |
| `BulletSpawner` | `initialInterval` / `minInterval` | 子弹生成频率 |
| `BulletSpawner` | `spreadAngle` | 子弹扩散角度 |
| `MonsterSpawner` | `initialInterval` / `minInterval` | 怪物生成频率 |
| `Monster` | `speed` / `homingStrength` | 怪物速度与追踪强度 |
| `PlayerController` | `moveSpeed` | 飞机基础移速 |

---

## 扩展建议

- 添加粒子系统作为子弹/爆炸效果
- 用 `Animator` 制作飞机飞行动画
- 实现本地排行榜（`PlayerPrefs`）
- 加入背景音乐与音效（`AudioSource`）
- 添加更多怪物类型（螺旋运动、分裂子弹等）

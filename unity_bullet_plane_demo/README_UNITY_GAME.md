# Unity 小游戏 MVP — 吸血鬼幸存者风格飞机射击

## 游戏核心循环

```
躲子弹 + 吸收金色子弹获取XP → 升级三选一 → 更强 → 更难的波次 → 循环
```

| 机制 | 说明 |
|------|------|
| **子弹地狱** | 普通白色子弹从四边射来，碰到即扣血 |
| **特殊子弹吸收** | 金色子弹碰到即吸收 → 获得 XP（核心差异化机制） |
| **自动武器** | 自动瞄准最近的神风敌机射击 |
| **神风敌机** | 红色敌机追击玩家，被击落掉落 XP 球 |
| **升级系统** | XP 满后暂停游戏，三选一强化 |
| **波次升级** | 每 30 秒更快、更密、敌人更强 |

---

## 脚本一览（13 个）

| 脚本 | 挂载对象 | 说明 |
|------|----------|------|
| `GameManager.cs` | GameManager 空物体 | 状态机：Playing/LevelUp/GameOver，XP/Level 管理 |
| `PlayerController.cs` | Player | 8方向移动、HP系统、特殊子弹吸收 |
| `AutoWeapon.cs` | Player | 自动瞄准射击 |
| `BulletSpawner.cs` | Spawner | 从四边生成子弹（含金色特殊子弹） |
| `Bullet.cs` | Bullet Prefab | 子弹移动、isSpecial 标记、金色着色 |
| `KamikazeEnemy.cs` | Enemy Prefab | 追击玩家的敌机，有 HP，死亡掉落 XP |
| `EnemySpawner.cs` | EnemySpawner 空物体 | 定时从屏幕外生成敌机 |
| `WaveManager.cs` | WaveManager 空物体 | 每波次提升难度 |
| `XPOrb.cs` | XPOrb Prefab | XP 球，靠近自动吸引，接触得分 |
| `PlayerProjectile.cs` | PlayerProjectile Prefab | 玩家子弹，击中敌机造成伤害 |
| `LevelUpManager.cs` | LevelUpManager 空物体 | 升级面板逻辑，三选一 |
| `HudController.cs` | HUD 空物体 | 显示时间/XP条/等级/波次/HP |

---

## 在 Unity 中搭建（2D 项目）

### 第 1 步：Tag 设置

Project Settings → Tags and Layers，确保存在：
- `Player`
- `Bullet`

### 第 2 步：创建预制体

#### Bullet Prefab
1. 新建 `Sprite` 物体，命名 `Bullet`
2. 添加 `Rigidbody2D`（Gravity Scale: 0，Collision Detection: Continuous）
3. 添加 `CircleCollider2D`（Is Trigger: ✓）
4. Tag 设为 `Bullet`
5. 挂 `Bullet.cs`
6. 拖入 Project 保存为 Prefab

#### PlayerProjectile Prefab
1. 新建小 `Sprite`（黄色小点）
2. 添加 `Rigidbody2D`（Gravity Scale: 0，Collision Detection: Continuous）
3. 添加 `CircleCollider2D`（Is Trigger: ✓）
4. 挂 `PlayerProjectile.cs`
5. 保存为 Prefab

#### Enemy Prefab（KamikazeEnemy）
1. 新建 `Sprite` 物体，命名 `Enemy`（会被脚本自动染红）
2. 添加 `Rigidbody2D`（Gravity Scale: 0）
3. 添加 `CircleCollider2D`（Is Trigger: ✓）
4. 挂 `KamikazeEnemy.cs`
5. 将 `XPOrb Prefab` 拖入 `xpOrbPrefab` 槽
6. 保存为 Prefab

#### XPOrb Prefab
1. 新建小 `Sprite`（绿色小球）
2. 添加 `CircleCollider2D`（Is Trigger: ✓）
3. 挂 `XPOrb.cs`
4. 保存为 Prefab

### 第 3 步：创建场景对象

```
Scene Hierarchy
├── GameManager          (空物体，挂 GameManager.cs)
├── WaveManager          (空物体，挂 WaveManager.cs)
├── LevelUpManager       (空物体，挂 LevelUpManager.cs)
├── Spawner              (空物体，挂 BulletSpawner.cs)
├── EnemySpawner         (空物体，挂 EnemySpawner.cs)
├── HUD                  (空物体，挂 HudController.cs)
├── Player               (Sprite，挂 PlayerController + AutoWeapon)
└── Canvas
    ├── SurviveTimeText  (Text)
    ├── LevelText        (Text)
    ├── WaveText         (Text)
    ├── HPText           (Text)
    ├── XPBarBG          (Image 背景)
    │   └── XPBarFill    (Image，Image Type: Filled, Fill Method: Horizontal)
    ├── GameOverPanel    (Panel)
    │   └── GameOverText (Text)
    └── LevelUpPanel     (Panel)
        ├── UpgradeBtn0  (Button，含 Text 子物体)
        ├── UpgradeBtn1  (Button，含 Text 子物体)
        └── UpgradeBtn2  (Button，含 Text 子物体)
```

### 第 4 步：Inspector 连线

**Player 物体**
- `PlayerController` → Tag 设为 `Player`
- `AutoWeapon.projectilePrefab` → PlayerProjectile Prefab

**Spawner**
- `BulletSpawner.bulletPrefab` → Bullet Prefab
- `BulletSpawner.player` → Player 物体

**EnemySpawner**
- `enemyPrefab` → Enemy Prefab

**WaveManager**
- `bulletSpawner` → Spawner 物体（含 BulletSpawner）
- `enemySpawner` → EnemySpawner 物体

**GameManager**
- `gameOverPanel` → GameOverPanel

**HudController**
- 将各 Text 和 XPBarFill 拖入对应槽

**LevelUpManager**
- `levelUpPanel` → LevelUpPanel
- `upgradeButtons[0..2]` → 三个 Button

---

## 升级选项（5 种）

| 升级 | 效果 |
|------|------|
| 火力加速 | 射速 ×1.3 |
| 引擎强化 | 移动速度 ×1.15 |
| 多重射击 | 每次多发 1 颗子弹 |
| 吸收扩张 | 金色子弹自动吸收范围 +0.8 |
| 护盾充能 | 最大HP +1，恢复 1 HP |

---

## 参数建议

### BulletSpawner（初始值）
| 参数 | 推荐值 |
|------|--------|
| spawnIntervalMin | 0.20 |
| spawnIntervalMax | 0.60 |
| bulletSpeedMin | 3.0 |
| bulletSpeedMax | 9.0 |
| aimAtPlayerChance | 0.65 |
| specialBulletChance | 0.12 |

### PlayerController
| 参数 | 推荐值 |
|------|--------|
| speed | 8 |
| maxHP | 3 |
| absorbRadius | 0（初始不自动吸收，需升级） |
| xpPerAbsorb | 3 |

### WaveManager
| 参数 | 推荐值 |
|------|--------|
| waveInterval | 30 |
| bulletIntervalScale | 0.80 |
| enemyIntervalScale | 0.75 |
| enemySpeedAdd | 0.4 |

---

## 下一步扩展

- Boss 在第 10 波出现（高HP、弹幕扇形）
- 音效：普通子弹碰撞声、吸收音效、升级音效
- 视觉特效：粒子爆炸（enemy 死亡）、金色粒子（吸收）
- 本地高分榜（`PlayerPrefs`）
- 局外解锁（`PlayerPrefs` 存累计 XP）


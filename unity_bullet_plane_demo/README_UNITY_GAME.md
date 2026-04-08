# Unity 小游戏: 躲子弹小飞机

这个目录提供了一套完整脚本，用于快速实现你要的玩法:
- 小飞机可移动
- 子弹从四面八方出现
- 子弹速度有快有慢
- 小飞机碰到子弹立即失败
- 按 `R` 重开

## 目录

- `Assets/Scripts/PlayerController.cs`: 玩家移动和碰撞失败
- `Assets/Scripts/BulletSpawner.cs`: 四边随机刷子弹和快慢速度
- `Assets/Scripts/Bullet.cs`: 子弹移动与自动销毁
- `Assets/Scripts/GameManager.cs`: 游戏状态与重开
- `Assets/Scripts/HudController.cs`: 生存时间和失败提示

## 在 Unity 中搭建步骤 (2D)

1. 新建 `Unity 2D` 项目。
2. 将本目录 `Assets/Scripts` 下所有脚本复制到你的 Unity 项目 `Assets/Scripts`。
3. 创建场景对象:
   - `GameManager` 空物体，挂 `GameManager`。
   - `Spawner` 空物体，挂 `BulletSpawner`。
   - `Player` 物体 (建议 Sprite: 小飞机图)。
4. 配置 `Player`:
   - 添加 `Rigidbody2D` (Body Type: Dynamic, Gravity Scale: 0)
   - 添加碰撞器 (`CircleCollider2D` 或 `BoxCollider2D`)
   - 挂 `PlayerController`
5. 创建 `Bullet` 预制体:
   - 新建一个小圆点 Sprite 物体命名 `Bullet`
   - 加 `Rigidbody2D` (Gravity Scale: 0)
   - 加 `CircleCollider2D`
   - Tag 设为 `Bullet`
   - 挂 `Bullet` 脚本
   - 拖入 Project 面板保存成 Prefab
6. 配置 `Spawner`:
   - 将 Bullet Prefab 拖到 `bulletPrefab`
   - 将 `Player` 拖到 `player`
   - `spawnIntervalMin/Max` 控制刷弹频率
   - `bulletSpeedMin/Max` 控制子弹快慢
7. 创建 UI:
   - `Canvas` 下建两个 `Text` (Legacy):
     - 一个显示生存时间
     - 一个显示失败文字
   - 新建空物体 `HUD` 挂 `HudController`
   - 将两个 Text 拖到 `surviveTimeText` 和 `gameOverText`
8. 可选: `GameManager` 的 `gameOverPanel` 指向一个失败面板对象；不设也能运行。
9. 点击 Play 测试。

## 参数建议

- 新手难度:
  - `spawnIntervalMin = 0.3`
  - `spawnIntervalMax = 0.8`
  - `bulletSpeedMin = 2.5`
  - `bulletSpeedMax = 7`
- 高难度:
  - `spawnIntervalMin = 0.08`
  - `spawnIntervalMax = 0.35`
  - `bulletSpeedMin = 4`
  - `bulletSpeedMax = 12`

## 玩法扩展 (可选)

- 加无敌时间
- 加分数和排行榜
- 子弹波次模式
- 道具系统 (减速、护盾、清屏)

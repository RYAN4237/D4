# POE2 自动化 README

面向《Path of Exile 2》小地图拼接、路径规划与战斗辅助的独立说明。

## 功能概览
- 窗口截屏：基于窗口标题 "Path of Exile 2" 获取小地图区域。
- 智能拼接：`stitch_new.SmartMinimapStitcher` 做亚像素位移检测与防裂缝拼接。
- 路径规划：`a_star.a_star_new` 在拼好的大地图上进行 A* 寻路。
- 战斗辅助：`attack.attack_nearby_enemies` 简单近战循环。
- 多进程：分拆拼接、寻路与主循环，减少卡顿。

## 环境与依赖
- Python 3.10+（建议 64 位）
- Windows，窗口标题需为 "Path of Exile 2"（默认捕获目标）
- OpenCV、numpy、keyboard 等依赖：见根目录 `maple_v2/requirements.txt`
- 可选：`poe2/driver.dll`（用于驱动输入；已在代码中引用位置）

安装示例（PowerShell）：
```powershell
cd C:\Users\Administrator\PycharmProjects\D4
python -m venv .venv
. .venv\Scripts\activate
pip install -r maple_v2\requirements.txt
```

## 快速运行
主流程（捕获→拼接→定位→寻路→移动→战斗）：
```powershell
. .venv\Scripts\activate
python -m poe2.route
```

## 运行要点
- 保证游戏窗口标题精确匹配 "Path of Exile 2"，否则 `CaptureScreen.get_hwnd` 找不到窗口。
- 拼接与捕获区域当前写死为 `(1081, 33) -> (1261, 185)`，对应 1080p/100% 缩放；若分辨率或 UI 缩放不同，请在 `poe2/route.py` 与 `poe2/stitch_new.py` 中调整。
- 小地图需保持可见，避免 UI 遮挡。
- OpenCV 窗口在前台显示，按 `q` 可退出窗口显示循环。

## 热键
- `F1`：在 `poe2.route` 主循环中触发 `stop_flag`，安全停止。

## 目录速览
- `route.py`：主入口，负责协程/多进程分发与 UI 预览。
- `stitch_new.py`：智能小地图拼接（含多种融合模式）。
- `a_star.py`：A* 寻路与小地图匹配 (`mini_map_matching`)。
- `attack.py`：基础战斗逻辑。
- `map_utils.py` / `visited_recorder.py`：地图栅格转换与访问记录。
- `find_char.py`：角色位置检测与标注。
- `draft_idea/`：实验性路径测试脚本。

## 常见问题
- **窗口未找到**：确认窗口标题；窗口需非最小化。
- **拼接空白/漂移**：检查分辨率与 DPI；校正截屏坐标；保证 minimap 未被遮挡。
- **寻路不动或卡住**：确认 `a_star_new` 输入的 `weight_grid` 与 `res` 坐标在同一尺度；必要时打印坐标或降低 `offset`。
- **键鼠无响应**：检查 `driver.dll` 是否存在并匹配当前系统；必要时在 `route.py` 调整 `driver` 初始化路径。

## TODO
- 将截屏坐标、DPI、热键、窗口标题改为可配置项。
- 增加日志与可视化调试开关。
- 为拼接与寻路添加单元/集成测试。


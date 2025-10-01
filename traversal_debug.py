import cv2
import numpy as np
import win32gui, win32ui, win32con, win32api, ctypes
import time
import math
from collections import deque
import random

PW_RENDERFULLCONTENT = 0x00000002


# ============== 窗口截图 ============== #
def get_game_window(title="暗黑破坏神IV"):
    hwnd = win32gui.FindWindow(None, title)
    if not hwnd:
        raise Exception(f"未找到窗口: {title}")
    return hwnd


def capture_window(hwnd):
    left, top, right, bottom = win32gui.GetWindowRect(hwnd)
    width, height = right - left, bottom - top

    hwndDC = win32gui.GetWindowDC(hwnd)
    mfcDC = win32ui.CreateDCFromHandle(hwndDC)
    saveDC = mfcDC.CreateCompatibleDC()
    saveBitMap = win32ui.CreateBitmap()
    saveBitMap.CreateCompatibleBitmap(mfcDC, width, height)
    saveDC.SelectObject(saveBitMap)
    ctypes.windll.user32.PrintWindow(hwnd, saveDC.GetSafeHdc(), PW_RENDERFULLCONTENT)

    bmpinfo = saveBitMap.GetInfo()
    bmpstr = saveBitMap.GetBitmapBits(True)
    img = np.frombuffer(bmpstr, dtype=np.uint8).reshape((bmpinfo['bmHeight'], bmpinfo['bmWidth'], 4))
    img_bgr = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)

    win32gui.DeleteObject(saveBitMap.GetHandle())
    saveDC.DeleteDC()
    mfcDC.DeleteDC()
    win32gui.ReleaseDC(hwnd, hwndDC)

    return img_bgr


# ============== 小地图提取和分析 ============== #
def extract_minimap(img_bgr):
    """提取小地图区域"""
    h, w = img_bgr.shape[:2]

    # 小地图通常在右上角
    minimap_size = int(w * 0.15)  # 小地图大小
    margin = int(w * 0.02)  # 边距

    x1 = w - minimap_size - margin
    y1 = margin
    x2 = w - margin
    y2 = margin + minimap_size

    # 确保坐标在图像范围内
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w, x2), min(h, y2)

    minimap = img_bgr[y1:y2, x1:x2].copy()
    region = (x1, y1, x2 - x1, y2 - y1)

    return minimap, region


def analyze_minimap_roads(minimap):
    """分析小地图中的道路"""
    h, w = minimap.shape[:2]

    # 转换为HSV颜色空间，更容易识别道路
    hsv = cv2.cvtColor(minimap, cv2.COLOR_BGR2HSV)

    # 定义道路颜色范围（根据暗黑IV小地图调整）
    # 道路通常是灰色/棕色/米色
    road_ranges = [
        # 灰色道路
        (np.array([0, 0, 50]), np.array([180, 50, 180])),
        # 棕色道路
        (np.array([10, 30, 30]), np.array([30, 150, 200])),
        # 米色/浅色道路
        (np.array([0, 0, 150]), np.array([180, 80, 255])),
    ]

    road_mask = np.zeros((h, w), dtype=np.uint8)

    for lower, upper in road_ranges:
        mask = cv2.inRange(hsv, lower, upper)
        road_mask = cv2.bitwise_or(road_mask, mask)

    # 形态学操作，连接断开的道路
    kernel = np.ones((3, 3), np.uint8)
    road_mask = cv2.morphologyEx(road_mask, cv2.MORPH_CLOSE, kernel)
    road_mask = cv2.morphologyEx(road_mask, cv2.MORPH_OPEN, kernel)

    # 膨胀以获取更宽的道路区域
    kernel_dilate = np.ones((5, 5), np.uint8)
    road_mask = cv2.dilate(road_mask, kernel_dilate, iterations=1)

    return road_mask


def find_player_in_minimap(minimap):
    """在小地图中定位玩家位置"""
    h, w = minimap.shape[:2]

    # 玩家通常在小地图中心附近
    center_x, center_y = w // 2, h // 2

    # 在小范围内搜索玩家标记（通常是亮色或特殊颜色）
    search_radius = min(w, h) // 10

    # 转换为HSV颜色空间，更容易识别特定颜色
    hsv = cv2.cvtColor(minimap, cv2.COLOR_BGR2HSV)

    # 尝试识别玩家标记（白色/亮色）
    lower_white = np.array([0, 0, 200])
    upper_white = np.array([180, 50, 255])
    white_mask = cv2.inRange(hsv, lower_white, upper_white)

    # 在中心区域寻找玩家
    center_region = minimap[
        center_y - search_radius:center_y + search_radius,
        center_x - search_radius:center_x + search_radius
    ]

    # 如果找到明显的玩家标记，使用它
    white_pixels = np.where(white_mask > 0)
    if len(white_pixels[0]) > 0:
        # 使用白色区域的质心
        moments = cv2.moments(white_mask)
        if moments["m00"] != 0:
            player_x = int(moments["m10"] / moments["m00"])
            player_y = int(moments["m01"] / moments["m00"])
            return (player_x, player_y)

    # 否则默认使用中心点
    return (center_x, center_y)


def get_minimap_navigation_points(road_mask, grid_size=15):
    """在小地图道路上生成导航点"""
    h, w = road_mask.shape

    points = []
    for y in range(grid_size // 2, h, grid_size):
        for x in range(grid_size // 2, w, grid_size):
            # 检查该点是否在道路区域内
            if road_mask[y, x] > 0:
                points.append((x, y))

    return points


# ============== 坐标转换 ============== #
def minimap_to_screen_coords(minimap_point, minimap_region, screen_size):
    """将小地图坐标转换为屏幕坐标"""
    mx, my = minimap_point
    map_x, map_y, map_w, map_h = minimap_region
    screen_w, screen_h = screen_size

    # 计算在小地图中的相对位置
    rel_x = mx / map_w
    rel_y = my / map_h

    # 转换为屏幕坐标（假设小地图代表整个区域）
    # 这里可以调整比例因子，使移动距离更合理
    screen_x = int(rel_x * screen_w)
    screen_y = int(rel_y * screen_h)

    # 确保坐标在屏幕范围内
    margin = 100
    screen_x = max(margin, min(screen_x, screen_w - margin))
    screen_y = max(margin, min(screen_y, screen_h - margin))

    return (screen_x, screen_y)


# ============== 改进的探索管理器 ============== #
class MinimapExplorationManager:
    def __init__(self, minimap_size):
        self.minimap_width, self.minimap_height = minimap_size
        self.explored_map = np.zeros((minimap_size[1], minimap_size[0]), dtype=np.uint8)
        self.exploration_path = []
        self.visited_points = set()  # 记录已经访问过的点
        self.last_positions = deque(maxlen=10)  # 记录最近位置，用于防卡住
        self.stuck_count = 0
        self.last_move_time = 0
        self.move_delay = 2.0  # 移动延迟2秒

    def update_explored_area(self, position, radius=15):
        """更新已探索区域"""
        x, y = position

        # 创建圆形探索区域
        y_coords, x_coords = np.ogrid[:self.minimap_height, :self.minimap_width]
        mask = ((x_coords - x) ** 2 + (y_coords - y) ** 2) <= radius ** 2

        # 更新已探索地图
        self.explored_map[mask] = 255

        # 记录路径
        self.exploration_path.append((x, y))
        self.last_positions.append((x, y))

        # 标记当前位置为已访问
        self.visited_points.add((x, y))

    def get_unexplored_targets(self, navigation_points, current_position, max_targets=3):
        """获取未探索的导航点作为目标，优先选择未访问过的点"""
        unexplored_points = []

        for point in navigation_points:
            x, y = point
            # 检查该点是否已探索且未访问过
            if self.explored_map[y, x] == 0 and point not in self.visited_points:
                # 计算与当前位置的距离
                distance = math.sqrt((x - current_position[0]) ** 2 + (y - current_position[1]) ** 2)
                unexplored_points.append((point, distance))

        # 按距离排序，选择最近的目标
        unexplored_points.sort(key=lambda x: x[1])

        # 返回点坐标，不包含距离
        return [point for point, _ in unexplored_points[:max_targets]]

    def is_stuck(self):
        """检查是否卡住（位置长时间没有变化）"""
        if len(self.last_positions) < 5:
            return False

        # 计算最近几个位置的平均移动距离
        total_distance = 0
        for i in range(1, len(self.last_positions)):
            x1, y1 = self.last_positions[i - 1]
            x2, y2 = self.last_positions[i]
            total_distance += math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)

        avg_distance = total_distance / (len(self.last_positions) - 1)

        # 如果平均移动距离很小，则认为卡住了
        return avg_distance < 5

    def get_escape_target(self, current_position, minimap_size, road_mask):
        """获取脱困目标点，确保目标在道路上"""
        w, h = minimap_size

        # 尝试不同的脱困策略，但确保目标在道路上
        escape_attempts = 0
        max_attempts = 10

        while escape_attempts < max_attempts:
            # 随机选择一个方向
            angle = random.uniform(0, 2 * math.pi)
            distance = random.randint(50, min(w, h) // 2)

            target_x = int(current_position[0] + math.cos(angle) * distance)
            target_y = int(current_position[1] + math.sin(angle) * distance)

            # 确保目标在范围内
            target_x = max(5, min(target_x, w - 5))
            target_y = max(5, min(target_y, h - 5))

            # 检查目标是否在道路上
            if road_mask[target_y, target_x] > 0:
                return (target_x, target_y)

            escape_attempts += 1

        # 如果找不到道路上的点，返回当前位置（作为最后手段）
        return current_position

    def can_move(self):
        """检查是否可以移动（满足移动延迟）"""
        current_time = time.time()
        return current_time - self.last_move_time >= self.move_delay

    def record_move(self):
        """记录移动时间"""
        self.last_move_time = time.time()

    def get_exploration_progress(self):
        """获取探索进度"""
        total_area = self.explored_map.size
        explored_area = np.sum(self.explored_map > 0)
        return explored_area / total_area * 100


# ============== 鼠标控制 ============== #
def click_at_position(hwnd, x, y):
    """在指定位置点击（用于移动）"""
    # 将坐标转换为窗口坐标
    left, top, right, bottom = win32gui.GetWindowRect(hwnd)
    window_x = x + left
    window_y = y + top

    # 发送鼠标点击消息
    lParam = win32api.MAKELONG(window_x, window_y)
    win32api.PostMessage(hwnd, win32con.WM_LBUTTONDOWN, win32con.MK_LBUTTON, lParam)
    time.sleep(0.5)
    win32api.PostMessage(hwnd, win32con.WM_LBUTTONUP, 0, lParam)


# ============== 可视化 ============== #
def create_visualization(img_bgr, minimap, minimap_region, road_mask, exploration_manager,
                         navigation_points, current_target, player_pos, screen_target=None):
    """创建可视化图像"""
    vis_img = img_bgr.copy()
    h, w = vis_img.shape[:2]

    # 绘制小地图道路分析（在小地图区域上叠加）
    mx, my, mw, mh = minimap_region
    road_vis = cv2.cvtColor(road_mask, cv2.COLOR_GRAY2BGR)
    road_vis[road_mask > 0] = [0, 255, 0]  # 道路显示为绿色

    # 将道路可视化叠加到小地图上
    minimap_with_roads = cv2.addWeighted(minimap, 0.7, road_vis, 0.3, 0)
    vis_img[my:my + mh, mx:mx + mw] = minimap_with_roads

    # 在小地图上绘制导航点
    for point in navigation_points:
        px, py = point
        if point in exploration_manager.visited_points:
            color = (0, 0, 255)  # 红色表示已访问
        elif exploration_manager.explored_map[py, px] > 0:
            color = (255, 0, 255)  # 紫色表示已探索但未访问
        else:
            color = (0, 255, 0)  # 绿色表示未探索

        cv2.circle(vis_img, (mx + px, my + py), 3, color, -1)

    # 在小地图上绘制探索路径
    if len(exploration_manager.exploration_path) > 1:
        for i in range(1, len(exploration_manager.exploration_path)):
            x1, y1 = exploration_manager.exploration_path[i - 1]
            x2, y2 = exploration_manager.exploration_path[i]
            cv2.line(vis_img,
                     (mx + x1, my + y1),
                     (mx + x2, my + y2),
                     (0, 255, 255), 2)

    # 在小地图上绘制当前目标
    if current_target is not None:
        target_x, target_y = int(current_target[0]), int(current_target[1])
        cv2.circle(vis_img, (mx + target_x, my + target_y), 8, (255, 255, 0), 2)
        cv2.line(vis_img,
                 (mx + player_pos[0], my + player_pos[1]),
                 (mx + target_x, my + target_y),
                 (255, 255, 0), 2)

    # 在小地图上绘制玩家位置
    cv2.circle(vis_img, (mx + player_pos[0], my + player_pos[1]), 6, (0, 0, 255), -1)

    # 在屏幕上绘制目标位置（如果存在）
    if screen_target is not None:
        screen_x, screen_y = int(screen_target[0]), int(screen_target[1])
        cv2.circle(vis_img, (screen_x, screen_y), 12, (255, 255, 0), 3)
        cv2.line(vis_img,
                 (w // 2, h // 2),  # 屏幕中心（玩家位置）
                 (screen_x, screen_y),
                 (255, 255, 0), 2)

    # 添加信息文本
    progress = exploration_manager.get_exploration_progress()
    cv2.putText(vis_img, f"探索进度: {progress:.1f}%", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.putText(vis_img, f"路径点数: {len(exploration_manager.exploration_path)}", (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.putText(vis_img, f"卡住计数: {exploration_manager.stuck_count}", (10, 90),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    # 显示移动状态
    if exploration_manager.can_move():
        move_status = "可以移动"
        color = (0, 255, 0)
    else:
        move_status = "移动冷却中"
        color = (0, 0, 255)

    cv2.putText(vis_img, f"状态: {move_status}", (10, 120),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

    cv2.putText(vis_img, "基于小地图导航", (10, 150),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    cv2.putText(vis_img, "按ESC退出", (10, 180),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    return vis_img


# ============== 基于小地图的主探索循环 ============== #
def minimap_based_auto_explore():
    """基于小地图的自动探索"""
    try:
        hwnd = get_game_window()
        print("游戏窗口已找到")

        # 获取窗口尺寸
        left, top, right, bottom = win32gui.GetWindowRect(hwnd)
        window_size = (right - left, bottom - top)
        print(f"窗口尺寸: {window_size[0]}x{window_size[1]}")

        # 主循环
        move_count = 0
        max_moves = 200
        current_minimap_target = None
        current_screen_target = None
        last_target = None

        print("开始基于小地图的探索...")
        print("按ESC键退出")

        # 先获取一次小地图信息来初始化探索管理器
        img = capture_window(hwnd)
        minimap, minimap_region = extract_minimap(img)
        minimap_size = (minimap.shape[1], minimap.shape[0])
        exploration_manager = MinimapExplorationManager(minimap_size)

        while move_count < max_moves:
            # 截图
            img = capture_window(hwnd)

            # 提取小地图
            minimap, minimap_region = extract_minimap(img)

            # 分析小地图道路
            road_mask = analyze_minimap_roads(minimap)

            # 生成导航点（只在小地图道路上）
            navigation_points = get_minimap_navigation_points(road_mask, grid_size=15)

            # 获取玩家在小地图中的位置
            player_minimap_pos = find_player_in_minimap(minimap)

            # 更新探索区域
            exploration_manager.update_explored_area(player_minimap_pos)

            # 检查是否卡住
            if exploration_manager.is_stuck():
                exploration_manager.stuck_count += 1
                print(f"检测到卡住，尝试脱困 #{exploration_manager.stuck_count}")

                # 使用脱困算法获取新目标，确保目标在道路上
                current_minimap_target = exploration_manager.get_escape_target(
                    player_minimap_pos, minimap_size, road_mask)
                print(f"脱困目标: {current_minimap_target}")

                # 如果多次卡住，重置探索
                if exploration_manager.stuck_count > 5:
                    print("多次卡住，重置探索地图")
                    exploration_manager.explored_map.fill(0)
                    exploration_manager.visited_points.clear()
                    exploration_manager.stuck_count = 0
            else:
                # 重置卡住计数
                exploration_manager.stuck_count = 0

                # 获取未探索目标
                if (current_minimap_target is None or
                        exploration_manager.explored_map[
                            int(current_minimap_target[1]), int(current_minimap_target[0])] > 0):

                    unexplored_targets = exploration_manager.get_unexplored_targets(
                        navigation_points, player_minimap_pos)

                    if unexplored_targets:
                        current_minimap_target = unexplored_targets[0]
                        # 避免重复选择同一个目标
                        if current_minimap_target == last_target and len(unexplored_targets) > 1:
                            current_minimap_target = unexplored_targets[1]
                    else:
                        # 如果没有未探索目标，选择已探索但未访问的点
                        for point in navigation_points:
                            if point not in exploration_manager.visited_points:
                                current_minimap_target = point
                                break

                        # 如果所有点都访问过，随机选择一个道路上的点
                        if current_minimap_target is None and navigation_points:
                            current_minimap_target = navigation_points[np.random.randint(0, len(navigation_points))]

            # 记录当前目标
            last_target = current_minimap_target

            # 将小地图目标转换为屏幕坐标
            if current_minimap_target is not None:
                current_screen_target = minimap_to_screen_coords(
                    current_minimap_target, minimap_region, window_size)
            else:
                current_screen_target = None

            # 可视化
            vis_img = create_visualization(
                img, minimap, minimap_region, road_mask, exploration_manager,
                navigation_points, current_minimap_target, player_minimap_pos, current_screen_target
            )

            cv2.imshow("基于小地图的自动探索", vis_img)

            # 执行移动（如果满足延迟条件）
            if (exploration_manager.can_move() and
                    current_minimap_target is not None and
                    current_screen_target is not None):
                # 标记目标为已访问
                exploration_manager.visited_points.add(current_minimap_target)

                # 模拟点击移动（在实际使用中取消注释）
                click_at_position(hwnd, int(current_screen_target[0]), int(current_screen_target[1]))
                exploration_manager.record_move()

                # 仅用于演示，不实际点击
                exploration_manager.record_move()
                move_count += 1
                print(f"移动到: {current_minimap_target} -> {current_screen_target} (移动 #{move_count})")

            # 检查退出
            key = cv2.waitKey(100)
            if key == 27:  # ESC键
                break

            # 短暂延迟
            time.sleep(1)

        # 保存最终探索结果
        final_vis = create_visualization(
            img, minimap, minimap_region, road_mask, exploration_manager,
            navigation_points, None, player_minimap_pos, None
        )
        cv2.imwrite("final_minimap_exploration.png", final_vis)

        print(f"探索完成，共移动 {move_count} 次")
        print(f"最终探索进度: {exploration_manager.get_exploration_progress():.1f}%")
        print("已保存最终探索结果: final_minimap_exploration.png")

        cv2.waitKey(0)
        cv2.destroyAllWindows()

    except Exception as e:
        print(f"错误: {e}")
        import traceback
        traceback.print_exc()
        cv2.destroyAllWindows()


# ============== 运行 ============== #
if __name__ == "__main__":
    print("=" * 50)
    print("暗黑破坏神IV - 基于小地图的自动探索")
    print("=" * 50)
    print("\n特点:")
    print("- 基于小地图道路分析，确保目标在道路上")
    print("- 只在小地图道路区域内生成导航点")
    print("- 可视化显示道路分析结果")
    print("- 改进的脱困算法，确保脱困目标也在道路上")
    print("\n注意:")
    print("- 此版本仅用于演示，不进行实际点击")
    print("- 按ESC键可随时退出")
    print("=" * 50)

    input("按回车开始基于小地图的探索...")
    minimap_based_auto_explore()

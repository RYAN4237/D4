import math
from collections import deque

import cv2
import numpy as np
import time

from poe2.map_utils import MapUtils
from collections import deque
from pathfinding.core.grid import Grid
from pathfinding.core.diagonal_movement import DiagonalMovement
from pathfinding.finder.a_star import AStarFinder

global current_path
global idx


def _is_image_like(x):
    return isinstance(x, np.ndarray)


def preprocess_route_img(route_img, mini_img, kernel_size=3, close_iter=1, debug_save_path=None):
    """
    Preprocess a route image to improve skeleton/connectivity and produce a weight grid.

    - route_img: BGR or grayscale image (numpy array) or path string.
    - kernel_size: size for morphological operations to connect thin lines.
    - close_iter: number of times to apply closing (helps bridge gaps).
    - debug_save_path: if provided, save debug overlay images here (prefix).

    Returns: weight_grid (H,W) where np.inf = obstacle, and debug dict
    """
    # If a path provided, try to load
    if isinstance(route_img, str):
        img = cv2.imread(route_img)
        if img is None:
            raise FileNotFoundError(f"preprocess_route_img: cannot read '{route_img}'")
    elif _is_image_like(route_img):
        img = route_img.copy()
    else:
        raise TypeError("route_img must be a numpy image or a filepath string")

    # Convert to grayscale for thresholding
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img

    print(np.unique(gray))
    # Heuristic thresholds (may be adjusted):
    # - dark (near 0): obstacle
    # - mid/gray: unexplored (prefer)
    # - bright/white: explored or background (less preferred)
    # Create three masks
    obstacle_mask = gray < 40
    white_mask = gray > 220
    gray_mask = (~obstacle_mask) & (~white_mask)

    # Create binary for morphological ops: consider non-obstacle as foreground
    bin_fg = (~obstacle_mask).astype(np.uint8) * 255

    # Morphological closing to bridge small gaps in skeleton / thin lines
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    proc = bin_fg.copy()
    for _ in range(max(1, close_iter)):
        proc = cv2.morphologyEx(proc, cv2.MORPH_CLOSE, kernel)

    # Optional small dilation to improve connectivity then erode back (closing already does this)
    proc = cv2.medianBlur(proc, 3)

    # Build weight grid: lower = preferred
    h, w = gray.shape[:2]
    weight = np.full((h, w), np.inf, dtype=np.float32)

    # Preferred cost for gray/unexplored
    weight[gray_mask] = 1.0
    # Slightly higher cost for white/explored
    weight[white_mask] = 3.0
    # Obstacles remain inf

    # Apply processed binary as further obstacle mask (if processed says obstacle, set inf)
    proc_bool = (proc == 0)
    weight[proc_bool] = np.inf

    debug = {
        'gray_count': int(np.count_nonzero(gray_mask)),
        'white_count': int(np.count_nonzero(white_mask)),
        'obstacle_count': int(np.count_nonzero(obstacle_mask)),
        'proc_shape': proc.shape,
    }

    return weight, debug

def mini_map_matching(mini_map, big_map, current_pos, threshold=0.7, debug=True):
    """原始的简单匹配实现：裁剪 mini_map、template matching，然后将 current_pos 从 mini_map 坐标映射到 big_map 全局坐标并返回。"""
    # Convert to grayscale
    big_map_hsv = cv2.cvtColor(big_map, cv2.COLOR_BGR2HSV)
    mini_map_hsv = cv2.cvtColor(mini_map, cv2.COLOR_BGR2HSV)
    # Enhance mini_map to reduce noise
    mini_map_mask = cv2.inRange(mini_map_hsv,
                       np.array([20, 60, 150], dtype=np.uint8),
                       np.array([130, 190, 220], dtype=np.uint8))
    mini_map_new = cv2.bitwise_and(mini_map, mini_map, mask=mini_map_mask)
    mini_map_new_gray = cv2.cvtColor(mini_map_new, cv2.COLOR_BGR2GRAY)
    mini_map_new_gray = cv2.adaptiveThreshold(mini_map_new_gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                                 cv2.THRESH_BINARY, 5, 1)
    mini_map_new_gray = cv2.medianBlur(mini_map_new_gray, 3)
    cv2.imshow("mini_map_new_gray", mini_map_new_gray)

    # Enhance big_map to reduce noise
    big_map_mask = cv2.inRange(big_map_hsv,
                                np.array([20, 60, 150], dtype=np.uint8),
                                np.array([130, 190, 220], dtype=np.uint8))
    big_map_new = cv2.bitwise_and(big_map, big_map, mask=big_map_mask)
    big_map_new_gray = cv2.cvtColor(big_map_new, cv2.COLOR_BGR2GRAY)
    big_map_new_gray = cv2.adaptiveThreshold(big_map_new_gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                              cv2.THRESH_BINARY, 5, 1)
    big_map_new_gray = cv2.medianBlur(big_map_new_gray, 3)
    # cv2.imshow("big_map_new_gray", big_map_new_gray)

    # original crop used in the project
    h, w = mini_map_new_gray.shape
    cx1, cy1 = 5, 5
    cx2, cy2 = max(cx1 + 1, w - 100), max(cy1 + 1, h - 5)

    # Determine provided player coordinates early so we can ensure the
    # template we match contains the player point. If current_pos is invalid,
    # fall back to the center of the mini map.
    try:
        px = int(current_pos[0])
        py = int(current_pos[1])
    except Exception:
        # fall back to center
        py, px = h // 2, w // 2

    # If player is outside the default crop, use full mini map as template so
    # the mapping is consistent (we'll set crop offsets to 0).
    use_full_template = not (cx1 <= px < cx2 and cy1 <= py < cy2)

    if use_full_template:
        template = mini_map_new_gray
        cx1, cy1 = 0, 0
        cx2, cy2 = w, h
    else:
        # guard against invalid crop
        if cx2 <= cx1 or cy2 <= cy1:
            template = mini_map_new_gray
            cx1, cy1 = 0, 0
            cx2, cy2 = w, h
        else:
            template = mini_map_new_gray[cy1:cy2, cx1:cx2]

    if template.size == 0:
        template = mini_map_new_gray
        cx1, cy1 = 0, 0
        cx2, cy2 = w, h



    # top-left of matched template in big_map
    top_left = MapUtils.findpic(big_map_new_gray, template, threshold, method=1)

    # template height (th) and width (tw)
    th, tw = template.shape

    # map current_pos (in full mini_map coords) to big_map coords
    try:
        px = int(current_pos[0])
        py = int(current_pos[1])
    except Exception:
        print("mini_map_matching: invalid current_pos:", current_pos)
        return None

    # account for the crop offset (cx1, cy1)
    crop_x_off = cx1
    crop_y_off = cy1

    # convert to coordinates inside the template used for matching
    adj_x = px - crop_x_off
    adj_y = py - crop_y_off

    # clamp adj to template bounds
    adj_x = max(0, min(adj_x, tw - 1))
    adj_y = max(0, min(adj_y, th - 1))

    # compute global position
    player_global_x = top_left[0] + adj_x
    player_global_y = top_left[1] + adj_y

    return player_global_x, player_global_y


def _find_nearest_gray_v1(start, weight_grid, max_radius=None, recorder=None):
    """
    查找最近的灰色像素 (weight==1.0)，使用向量化优化。
    Returns (x,y) or None.
    """
    H, W = weight_grid.shape[:2]
    sx, sy = int(start[0]), int(start[1])
    if not (0 <= sx < W and 0 <= sy < H):
        return None

    target_val = 1.0
    previous_points = recorder.load_point()
    # print("previous_points:", previous_points)

    # 找所有灰色像素坐标 (y, x)
    gray_coords = np.argwhere(weight_grid == target_val)

    if len(gray_coords) == 0:
        return None

    # 过滤已访问的点
    if previous_points:
        mask = np.array([
            (int(x), int(y)) not in previous_points
            for y, x in gray_coords
        ])
        gray_coords = gray_coords[mask]


    if len(gray_coords) == 0:
        return None

    # 计算到起点的欧几里得距离
    distances = np.sqrt((gray_coords[:, 1] - sx) ** 2 + (gray_coords[:, 0] - sy) ** 2)

    # 应用 max_radius 限制
    if max_radius is not None:
        valid = distances <= max_radius
        if not np.any(valid):
            return None
        distances = distances[valid]
        gray_coords = gray_coords[valid]

    # 找最近的点
    idx = np.argmin(distances)
    ny, nx = gray_coords[idx]

    return (int(nx), int(ny))


def _line_is_passable(x0, y0, x1, y1, weight_grid):
    """Bresenham 线段遍历，检查从 (x0,y0) 到 (x1,y1) 的所有像素是否通行"""
    points = []
    dx, dy = abs(x1 - x0), abs(y1 - y0)
    sx, sy = (1 if x0 < x1 else -1), (1 if y0 < y1 else -1)
    err = dx - dy
    x, y = x0, y0
    H, W = weight_grid.shape[:2]

    while True:
        if not (0 <= x < W and 0 <= y < H) or np.isinf(weight_grid[y, x]):
            return False
        points.append((x, y))
        if x == x1 and y == y1:
            break
        e2 = 2 * err
        if e2 > -dy:
            err -= dy
            x += sx
        if e2 < dx:
            err += dx
            y += sy
    return True

def a_star_new(start, goal, route_img, mini_img, padding=10, debug_save_prefix=None, recorder=None):
    """
    Weighted A* that prefers gray (unexplored) pixels when provided a three-map image
    where gray = unexplored (preferred), white = explored (less preferred), black = obstacle.

    start, goal: (x,y) integer tuples in image coordinates. If goal is None, the function will
    automatically pick the nearest gray (unexplored) pixel as the exploration target.
    route_img: image array (BGR or grayscale) or precomputed weight grid (numpy array float32)
    padding: step size (1 recommended). Using 8-neighbors implicitly via neighbors list.
    debug_save_prefix: if provided, used to save debug images.

    Returns: path as list of (x,y) coordinates from start to goal (inclusive) or [] if no path.
    """


    # If route_img is an image, build weight grid
    if _is_image_like(route_img) and route_img.dtype == np.uint8:
        weight_grid, debug = preprocess_route_img(route_img, mini_img, kernel_size=3, close_iter=1,
                                                  debug_save_path=debug_save_prefix)
    elif _is_image_like(route_img) and route_img.dtype in (np.float32, np.float64):
        weight_grid = route_img.astype(np.float32)
        debug = {}
    else:
        # try reading if string
        if isinstance(route_img, str):
            img = cv2.imread(route_img)
            if img is None:
                raise FileNotFoundError(f"a_star: cannot read '{route_img}'")
            weight_grid, debug = preprocess_route_img(img, mini_img, debug_save_path=debug_save_prefix)
        else:
            raise TypeError("route_img must be image array, weight grid array, or filepath string")


    H, W = weight_grid.shape[: 2]

    # If goal is None, choose nearest gray pixel (weight==1.0) as exploration target
    if goal is None:
        tgt = _find_nearest_gray_v1(start, weight_grid, recorder=recorder)
        if tgt is None:
            # no gray found, fallback: choose any finite pixel (e.g., nearest finite)
            visited = np.zeros((H, W), dtype=np.bool_)
            q = deque()
            sx0, sy0 = int(start[0]), int(start[1])
            sx0 = max(0, min(W - 1, sx0))
            sy0 = max(0, min(H - 1, sy0))
            q.append((sx0, sy0))
            visited[sy0, sx0] = True
            found = None
            while q:
                x, y = q.popleft()
                if np.isfinite(weight_grid[y, x]):
                    found = (x, y)
                    break
                for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (1, -1), (-1, 1), (1, 1)]:
                    nx, ny = x + dx, y + dy
                    if 0 <= nx < W and 0 <= ny < H and not visited[ny, nx]:
                        visited[ny, nx] = True
                        q.append((nx, ny))
            if found is None:
                return []
            tgt = found
        goal = tgt
        try:
            debug['explore_target'] = goal
        except Exception:
            pass

    print("goal", goal)
    # clamp start/goal
    orig_start = (int(start[0]), int(start[1]))
    orig_goal = (int(goal[0]), int(goal[1]))
    sx = max(0, min(W - 1, orig_start[0]))
    sy = max(0, min(H - 1, orig_start[1]))
    gx = max(0, min(W - 1, orig_goal[0]))
    gy = max(0, min(H - 1, orig_goal[1]))
    start = (sx, sy)
    goal = (gx, gy)

    if orig_start != start or orig_goal != goal:
        print(
            f"a_star: clamped start from {orig_start} to {start}, goal from {orig_goal} to {goal} (image size {W}x{H})")


    # Nudge start/goal if on obstacle
    def _nudge(pt, max_r=5):
        x0, y0 = pt
        if not np.isinf(weight_grid[y0, x0]):
            return pt
        for r in range(1, max_r + 1):
            for dx in range(-r, r + 1):
                for dy in range(-r, r + 1):
                    x = x0 + dx
                    y = y0 + dy
                    if 0 <= x < W and 0 <= y < H and not np.isinf(weight_grid[y, x]):
                        return (x, y)
        return None

    ns = _nudge(start)
    ng = _nudge(goal)
    if ns is None or ng is None:
        print("a_star: cannot nudge start/goal off obstacles")
        return []
    start = ns
    goal = ng

    # === 使用 pathfinding 库 ===
    # 将 weight_grid 转换为 pathfinding 需要的矩阵格式
    # pathfinding:  0 = 障碍物, >0 = 可通行 (值越大代价越低)
    # weight_grid:  inf = 障碍物, 1. 0 = 灰色(优先), 3.0 = 白色(次优先)

    # 转换:  weight 越低越优先 -> pathfinding weight 越高越优先
    # 使用 10/weight 转换，inf -> 0
    matrix = np.zeros((H, W), dtype=np.float32)
    finite_mask = np.isfinite(weight_grid)
    matrix[finite_mask] = 10.0 / weight_grid[finite_mask]  # weight=1 -> 10, weight=3 -> 3. 33
    matrix[~finite_mask] = 0  # 障碍物

    # 转为 int 矩阵 (pathfinding 需要)
    # 放大以保留精度:  1.0 -> 100, 3.0 -> 33
    matrix_int = (matrix * 10).astype(np.int32)
    matrix_int = np.clip(matrix_int, 0, 100)

    # 创建 Grid (注意:  pathfinding 使用 [y][x] 索引)
    grid = Grid(matrix=matrix_int.tolist())

    start_node = grid.node(start[0], start[1])
    end_node = grid.node(goal[0], goal[1])

    finder = AStarFinder(
        diagonal_movement=DiagonalMovement.always,
        weight=1,
        time_limit=30.0
    )

    path, runs = finder.find_path(start_node, end_node, grid)

    if not path:
        return []

    # 转换为 (x, y) 元组列表
    path = [(node.x, node.y) for node in path]

    if recorder is not None:
        final_x, final_y = path[-1]
        # print("dest:", final_x, final_y)
        if abs(start[0] - final_x) <= 10 and abs(start[1] - final_y) <= 10:
            recorder.mark_point(final_x, final_y)

    return path



if __name__ == "__main__":
    pass
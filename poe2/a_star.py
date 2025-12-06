from collections import deque

import cv2
import numpy as np
import time
from visited_recorder import VisitedRecorder
import heapq
from collections import deque

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
    cv2.imshow("big_map_new_gray", big_map_new_gray)

    # original crop used in the project
    h, w = mini_map_new_gray.shape
    cx1, cy1 = 5, 5
    cx2, cy2 = max(cx1 + 1, w - 100), max(cy1 + 1, h - 5)

    # Determine provided player coordinates early so we can ensure the
    # template we match contains the player point. If current_pos is invalid,
    # fall back to the center of the mini map.
    px = None
    py = None
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

    # If debug, save the template image used for matching so user can inspect
    template_path = None

    # Template match
    res = cv2.matchTemplate(big_map_new_gray, template, cv2.TM_CCOEFF_NORMED)
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
    if max_val < threshold:
        print(f"[mini_map_matching] match failed: max_val={max_val} < threshold={threshold}")
        return 0, 0  # match failed

    # top-left of matched template in big_map
    top_left = max_loc

    # template height (th) and width (tw)
    th, tw = template.shape

    def _annotate_and_save(big_img, top_left, template_w, template_h, mapped_pt, score, tag=None, show=True):
        vis = big_img.copy()
        x0, y0 = int(top_left[0]), int(top_left[1])
        x1, y1 = x0 + int(template_w), y0 + int(template_h)
        # rectangle for template
        cv2.rectangle(vis, (x0, y0), (x1, y1), (0, 0, 255), 2)
        # mapped player point
        cv2.circle(vis, (int(mapped_pt[0]), int(mapped_pt[1])), 5, (0, 255, 0), -1)
        # label text
        label = f"score:{score:.3f} mapped:{int(mapped_pt[0])},{int(mapped_pt[1])}"
        if tag:
            label = f"{tag} " + label
        cv2.putText(vis, label, (max(0, x0), max(12, y0 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255,255,0), 1)
        # timestamp
        ts = int(time.time())
        fname = rf"c:\Repo\D4\debug\mini_match_dbg_{ts}_{int(time.time_ns()%1000)}.png"

        # if show:
        #     try:
        #         cv2.imshow('mini_map_matching_debug', vis)
        #         cv2.waitKey(1)
        #     except Exception:
        #         pass
        return fname

    # Prepare debug info container
    info = {
        'score': float(max_val),
        'top_left': (int(top_left[0]), int(top_left[1])),
        'crop': (int(cx1), int(cy1), int(cx2), int(cy2)),
        'template_size': (int(tw), int(th)),
        'debug_image': None,
        'template_image_path': template_path,
        'ok_in_template': False,
        'ok_in_matched_rect': False,
    }
    # if debug:
    #     print('--- mini_map_matching debug ---')
    #     print('mini size:', mini_map.shape)
    #     print('template shape (h,w):', (th, tw))
    #     print('crop offsets (cx1,cy1,cx2,cy2):', (cx1, cy1, cx2, cy2))
    #     print('match top_left (x,y):', top_left)
    #     print('match score max_val:', max_val)

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

    if debug:
        info.update({
            'player_on_mini': (int(px), int(py)),
            'player_in_template': (int(adj_x), int(adj_y)),
            'mapped_global': (int(player_global_x), int(player_global_y)),
        })
        # check whether original player point was inside the template before clamping
        orig_in_template = ( (px - cx1) >= 0 and (px - cx1) < tw and (py - cy1) >= 0 and (py - cy1) < th )
        info['ok_in_template'] = bool(orig_in_template)
        # check whether mapped global point lies inside the matched rectangle on big_map
        bx0, by0 = int(top_left[0]), int(top_left[1])
        bx1, by1 = bx0 + int(tw), by0 + int(th)
        in_rect = (player_global_x >= bx0 and player_global_x < bx1 and player_global_y >= by0 and player_global_y < by1)
        info['ok_in_matched_rect'] = bool(in_rect)
        # save annotated overlay and record path
        dbg_path = _annotate_and_save(big_map, top_left, tw, th, (player_global_x, player_global_y), float(max_val), tag='mini_map_matching')
        info['debug_image'] = dbg_path
        print('mini_map_matching info:', info)
        return player_global_x, player_global_y, info

    return player_global_x, player_global_y

def _find_nearest_gray(start, weight_grid, max_radius=None, recorder=None):
    """BFS to find nearest pixel with weight==1.0 (gray/unexplored). Excludes start if it's gray and seeks another pixel.
    Returns (x,y) or None."""
    H, W = weight_grid.shape[:2]
    sx, sy = int(start[0]), int(start[1])
    if not (0 <= sx < W and 0 <= sy < H):
        return None
    target_val = 1.0
    visited = np.zeros((H, W), dtype=np.bool_)
    q = deque()
    q.append((sx, sy, 0))
    visited[sy, sx] = True
    previous_points = recorder.load_point()
    # if start itself matches but we want a different point, still allow returning start
    while q:
        x, y, d = q.popleft()
        if (x, y) in previous_points:
            continue
        if weight_grid[y, x] == target_val:
            recorder.mark_point(x, y)
            return (x, y)
        if max_radius is not None and d >= max_radius:
            continue
        for dx, dy in [(-1, -1), (0, -1), (1, -1), (-1, 0), (1, 0), (-1, 1), (0, 1), (1, 1)]:
            nx, ny = x + dx, y + dy
            if 0 <= nx < W and 0 <= ny < H and not visited[ny, nx] and np.isfinite(weight_grid[ny, nx]):
                visited[ny, nx] = True
                q.append((nx, ny, d + 1))
    return None


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


def a_star(start, goal, route_img, mini_img, padding=10, debug_save_prefix=None):
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

    recorder = VisitedRecorder(route_img, mask_path=r"record_mask.png")
    H, W = weight_grid.shape[:2]

    # If goal is None, choose nearest gray pixel (weight==1.0) as exploration target
    if goal is None:
        tgt = _find_nearest_gray(start, weight_grid, recorder=recorder)
        if tgt is None:
            # no gray found, fallback: choose any finite pixel (e.g., nearest finite)
            # perform BFS to find nearest finite
            visited = np.zeros((H, W), dtype=np.bool_)
            q = deque()
            sx0, sy0 = int(start[0]), int(start[1])
            sx0 = max(0, min(W - 1, sx0)); sy0 = max(0, min(H - 1, sy0))
            q.append((sx0, sy0))
            visited[sy0, sx0] = True
            found = None
            while q:
                x, y = q.popleft()
                if np.isfinite(weight_grid[y, x]):
                    found = (x, y); break
                for dx, dy in [(-1,0),(1,0),(0,-1),(0,1),(-1,-1),(1,-1),(-1,1),(1,1)]:
                    nx, ny = x+dx, y+dy
                    if 0<=nx<W and 0<=ny<H and not visited[ny, nx]:
                        visited[ny, nx] = True
                        q.append((nx, ny))
            if found is None:
                # totally blocked
                return []
            tgt = found
        goal = tgt
        # attach chosen target to debug if possible
        try:
            debug['explore_target'] = goal
        except Exception:
            pass

    cv2.circle(route_img, goal, 5, (127, 255, 127), cv2.FILLED)
    def heuristic(a, b):
        # Euclidean heuristic
        return np.hypot(a[0] - b[0], a[1] - b[1])

    # neighbors 8-connected
    # neigh_offsets = [(-1, -1), (0, -1), (1, -1), (-1, 0), (1, 0), (-1, 1), (0, 1), (1, 1)]
    neigh_offsets = [
        # 4-connected (step=1)
        (0, -1), (1, 0), (-1, 0), (0, 1),
        # 8-connected (step=1)
        (-1, -1), (1, -1), (-1, 1), (1, 1),
        # larger steps (step=5)
        (0, -5), (5, 0), (-5, 0), (0, 5),
        (-5, -5), (5, -5), (-5, 5), (5, 5),
    ]

    # clamp start/goal
    orig_start = (int(start[0]), int(start[1]))
    orig_goal = (int(goal[0]), int(goal[1]))
    sx = max(0, min(W - 1, orig_start[0]))
    sy = max(0, min(H - 1, orig_start[1]))
    gx = max(0, min(W - 1, orig_goal[0]))
    gy = max(0, min(H - 1, orig_goal[1]))
    start = (sx, sy)
    goal = (gx, gy)
    # Inform if clamp changed coordinates (e.g., requested point outside image)
    if orig_start != start or orig_goal != goal:
        print(f"a_star: clamped start from {orig_start} to {start}, goal from {orig_goal} to {goal} (image size {W}x{H})")

    # If start or goal is on an obstacle, try to nudge to nearest non-obstacle within small radius
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
        return []
    start = ns
    goal = ng

    close_set = set()
    came_from = {}
    gscore = {start: 0.0}
    fscore = {start: heuristic(start, goal)}
    oheap = []

    heapq.heappush(oheap, (fscore[start], start))

    max_iterations = H * W * 4
    iters = 0
    found = False

    while oheap:
        iters += 1
        if iters > max_iterations:
            break
        current = heapq.heappop(oheap)[1]
        if current == goal:
            found = True
            break

        close_set.add(current)
        cx, cy = current
        for dx, dy in neigh_offsets:
            nx = cx + dx
            ny = cy + dy
            if not (0 <= nx < W and 0 <= ny < H):
                continue
            if not _line_is_passable(cx, cy, nx, ny, weight_grid):
                continue
            # movement cost = average of current and neighbor pixel costs * movement distance
            move_cost = (weight_grid[cy, cx] + weight_grid[ny, nx]) * 0.5
            # incorporate diagonal distance
            dist = np.hypot(dx, dy )
            tentative_g_score = gscore[current] + move_cost * dist

            neighbor = (nx, ny)
            if neighbor in close_set and tentative_g_score >= gscore.get(neighbor, np.inf):
                continue
            if neighbor[0] < 0 or neighbor[0] >= W or neighbor[1] < 0 or neighbor[1] >= H:
                continue

            if tentative_g_score < gscore.get(neighbor, np.inf) or neighbor not in [i[1] for i in oheap]:
                came_from[neighbor] = current
                gscore[neighbor] = tentative_g_score
                fscore[neighbor] = tentative_g_score + heuristic(neighbor, goal)
                heapq.heappush(oheap, (fscore[neighbor], neighbor))

    if not found:
        return []

    # reconstruct path
    data = []
    cur = goal
    while cur in came_from:
        data.append(cur)
        cur = came_from[cur]
    data.append(start)
    path = data[::-1]

    for i in range(len(path) - 1):
        cv2.line(route_img, path[i], path[i + 1], (127, 255, 127), 2)
    return path





def merge_blue_into_binary(img_bgr, blue_lower=(90,120,80), blue_upper=(110,255,255),
                           dilate_before_close=True, kernel_size=(3,3), blue_value=127):
    """
    把 blue_mask 合并到二值图 bin_gray（来自 adaptiveThreshold），并把 blue 区设为 blue_value。
    参数:
      - img_bgr: 原始 BGR 图
      - bin_gray: adaptiveThreshold 的结果 (单通道 0/255)
      - blue_lower/blue_upper: HSV 范围（建议基于样本 [100,188,178] 使用 H~100, S 下限 <=188）
      - dilate_before_close: 是否先 dilate 再 close（避免 open 吃掉细线）
      - kernel_size: 形态学核尺寸
      - blue_value: 合并后蓝色像素值 (127)
    返回:
      - three_map: 单通道 uint8，值为 {0, blue_value, 255}
      - vis: BGR 可视化图（blue->蓝色, obstacle->白, free->黑）
    """
    big_map_mask_hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)

    # 你的原 pipeline
    big_map_mask = cv2.inRange(big_map_mask_hsv,
                               np.array([20, 60, 150], dtype=np.uint8),
                               np.array([130, 190, 220], dtype=np.uint8))
    big_map_new = cv2.bitwise_and(img_bgr, img_bgr, mask=big_map_mask)
    big_map_new_gray = cv2.cvtColor(big_map_new, cv2.COLOR_BGR2GRAY)
    big_map_new_gray = cv2.adaptiveThreshold(big_map_new_gray, 255,
                                             cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                             cv2.THRESH_BINARY, 5, 1)
    big_map_new_gray = cv2.medianBlur(big_map_new_gray, 3)
    big_map_new_gray = cv2.dilate(big_map_new_gray, (3, 3), iterations=1)
    big_map_new_gray = cv2.morphologyEx(big_map_new_gray, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8), iterations=1)
    big_map_new_gray = cv2.erode(big_map_new_gray, (3, 3), iterations=1)


    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    lower = np.array(blue_lower, dtype=np.uint8)
    upper = np.array(blue_upper, dtype=np.uint8)
    blue_mask = cv2.inRange(hsv, lower, upper)

    k = np.ones(kernel_size, np.uint8)

    # 推荐先把线变粗以保持连通性：dilate -> close -> erode
    if dilate_before_close:
        blue_mask = cv2.dilate(blue_mask, k, iterations=1)
    blue_mask = cv2.morphologyEx(blue_mask, cv2.MORPH_CLOSE, k, iterations=1)
    if dilate_before_close:
        blue_mask = cv2.erode(blue_mask, k, iterations=1)


    # 合并：把 adaptiveThreshold 的结果作为基础，然后覆盖 blue 区为 127
    three_map = big_map_new_gray.copy().astype(np.uint8)
    three_map[blue_mask > 0] = blue_value

    return three_map

if __name__ == "__main__":
    pass
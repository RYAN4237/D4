import os
import sys

from ultralytics import YOLO

from final.actions import move_target
from final.utils import _enqueue_latest_move

# When this module is executed as a script (python final\demo.py), __package__ will be None
# which causes relative top-level package imports like `from final import ...` to fail because
# the interpreter places the script's directory (final/) on sys.path instead of the repo root.
# Ensure the repo root (parent of this file) is on sys.path so `import final` works.
if __package__ is None:
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
    if repo_root not in sys.path:
        sys.path.insert(0, repo_root)

import time
import cv2
from concurrent.futures import ThreadPoolExecutor
from queue import Queue, Empty
import threading
from functools import partial
from typing import Callable, Any, cast
import win32gui
import random
import ctypes

from final import capture
from final import actions
from final import minimap
from final import templates
from final import status
from final import utils
from final.minimap import get_nearest_color_code, prepare_route_map

key_queue = Queue()
hp_queue = Queue()  # high-priority queue for healing actions
attack_queue = Queue()  # medium-priority queue for attacks/skills
move_queue = Queue()  # low-priority queue for movement actions


def keyboard_worker():
    """独立线程：HP 优先处理（第一优先级），其余使用加权队列。
    Priority: HP always first, then weighted: Attack=60%, Key=20%, Move=20%
    """
    # track which queues have received shutdown sentinel
    closed = {'hp': False, 'atk': False, 'key': False, 'move': False}

    # Define queue weights for non-HP queues (Attack, Key, Move)
    other_queue_weights = {
        'atk': 70,     # 70% priority for attacks
        'key': 20,     # 20% priority for misc keys
        'move': 10     # 10% priority for movement
    }

    queue_map = {
        'hp': hp_queue,
        'atk': attack_queue,
        'key': key_queue,
        'move': move_queue
    }

    sleep_durations = {
        'hp': 0.05,
        'atk': 0.08,
        'key': 0.1,
        'move': 0.1
    }

    def get_active_other_queues():
        """Return list of non-HP queue names and weights, excluding closed queues"""
        active = {k: v for k, v in other_queue_weights.items() if not closed[k]}
        return list(active.keys()), list(active.values())

    while True:
        # Check if all queues are closed
        if all(closed.values()):
            break

        # 1) ALWAYS check HP queue first (highest priority)
        if not closed['hp']:
            try:
                hp_func = hp_queue.get_nowait()

                # Check for shutdown sentinel
                if hp_func is None:
                    hp_queue.task_done()
                    closed['hp'] = True
                else:
                    # Validate and execute
                    if not callable(hp_func):
                        hp_queue.task_done()
                        continue

                    hp_callable = cast(Callable[..., Any], hp_func)
                    try:
                        hp_callable()
                        time.sleep(sleep_durations['hp'])
                    except Exception as e:
                        print(f"keyboard_worker HP error: {e}, func {hp_func}")
                    finally:
                        hp_queue.task_done()
                    # After processing HP, loop back to check HP again
                    continue

            except Empty:
                pass  # HP queue empty, proceed to other queues

        # 2) Process other queues with weighted selection
        active_names, active_weights = get_active_other_queues()

        if not active_names:
            # Only HP queue might be active, or all closed
            if not closed['hp']:
                time.sleep(0.05)  # Brief sleep if only waiting for HP
            continue

        # Weighted random selection among Attack, Key, Move
        selected_queue_name = random.choices(active_names, weights=active_weights, k=1)[0]
        selected_queue = queue_map[selected_queue_name]

        try:
            # Try to get item without blocking
            func = selected_queue.get_nowait()

            # Check for shutdown sentinel
            if func is None:
                selected_queue.task_done()
                closed[selected_queue_name] = True
                continue

            # Validate callable
            if not callable(func):
                if selected_queue_name == 'move':
                    print(f"Dequeued {selected_queue_name} item is not callable:", repr(func))
                selected_queue.task_done()
                continue

            # Execute the function
            func_callable = cast(Callable[..., Any], func)
            try:
                if selected_queue_name == 'move':
                    print(f"Executing {selected_queue_name} task...")
                func_callable()
                if selected_queue_name == 'move':
                    print(f"{selected_queue_name.capitalize()} task executed")
                time.sleep(sleep_durations[selected_queue_name])
            except Exception as e:
                print(f"keyboard_worker {selected_queue_name.upper()} error: {e}, func {func}")
            finally:
                selected_queue.task_done()

        except Empty:
            # Selected queue was empty, brief sleep to avoid CPU spinning
            time.sleep(0.02)


def main():
    args = utils.parse_args()

    # init
    templates_cache = templates.load_templates_cache(args.templates_dir)
    # prefer SIFT_create when available, otherwise fall back to ORB_create/BRISK_create
    sift_creator = getattr(cv2, 'SIFT_create', None)
    orb_creator = getattr(cv2, 'ORB_create', None)
    brisk_creator = getattr(cv2, 'BRISK_create', None)

    print(f"Using feature detector creator: SIFT_create={bool(sift_creator)}, ORB_create={bool(orb_creator)}, BRISK_create={bool(brisk_creator)}")
    if sift_creator:
        detector = sift_creator()
        matcher_norm = cv2.NORM_L2
    elif orb_creator:
        detector = orb_creator(nfeatures=1000)
        matcher_norm = cv2.NORM_HAMMING
    elif brisk_creator:
        detector = brisk_creator()
        matcher_norm = cv2.NORM_HAMMING
    else:
        raise RuntimeError('No supported feature detector available in cv2 (SIFT_create/ORB_create/BRISK_create).')

    time.sleep(1)  # wait for user to switch to game window

    # 启动按键线程（守护线程）
    t = threading.Thread(target=keyboard_worker, daemon=True)
    t.start()
    print("keyboard_worker thread started")

    flame = 0

    buff_time = time.time()

    red_flag = False

    img, hwnd = capture.capture_window(args.window_title)
    x, y, rw, rh = minimap.detect_minimap_dynamic_v2(frame=img)

    if args.debug:
        x = 10
        y = 30
        rw = 235
        rh = 130

    print(f"Detected minimap at x={x}, y={y}, rw={rw}, rh={rh}")
    # 简单寻路
    route = [70, 120]

    is_left = False

    routes = prepare_route_map(args.route_path)
    route_idx = 0;
    prev_pos = None

    while True:
        start = time.time()
        route_img = cv2.imread(routes[route_idx])
        img, hwnd = capture.capture_window(args.window_title)
        # x, y, rw, rh = minimap.detect_minimap_dynamic(frame=img)
        minimap_img = img[y:y+rh, x:x+rw]
        # 线程池并行处理小地图、人物、状态栏检测

        with ThreadPoolExecutor(max_workers=6) as ex:
            f_min = ex.submit(minimap.find_yellow_dot, minimap_img)
            f_char = ex.submit(templates.template_match_character, img, args.template_name)
            f_status = ex.submit(status.extract_status_bars_precise, img)

            hp_region, mp_region, _, _ = f_status.result()
            hp_pct, _, _, _ = status.calculate_bar_fill_ratio(hp_region, 'red')
            mp_pct, _, _, _ = status.calculate_bar_fill_ratio(mp_region, 'blue')

            if hp_pct < args.hp_threshold:
                # enqueue to high-priority hp_queue so healing is handled before others
                p =partial(actions.press_del_key, hwnd)
                _enqueue_latest_move(hp_queue, p)
            if mp_pct < args.mp_threshold:
                p = partial(actions.press_end_key, hwnd)
                _enqueue_latest_move(hp_queue, p)

            min_pos, red_flag = f_min.result()
            big_res = f_char.result()
            big_pos = big_res.get("center", (0, 0))
            big_conf = big_res.get('confidence', 0.0)

            # diagnostics: ensure templates cache non-empty
            if not templates_cache:
                print("templates_cache is empty - no templates to match against")

            # submit monster detection tasks only if in debug mode or character is confidently found
            if big_conf >= args.min_char_confidence and not args.enable_yolo:
                f_left = ex.submit(templates.template_match_monster, img, detector, templates_cache, big_pos, True,
                                   match_ratio=args.match_ratio, matcher_norm=matcher_norm, key_queue=key_queue,
                                   attack_queue=attack_queue, move_queue=move_queue, current_hwnd=hwnd,
                                   debug=args.debug)
                f_right = ex.submit(templates.template_match_monster, img, detector, templates_cache, big_pos, False,
                                    match_ratio=args.match_ratio, matcher_norm=matcher_norm, key_queue=key_queue,
                                    attack_queue=attack_queue, move_queue=move_queue, current_hwnd=hwnd,
                                    debug=args.debug)

            # using yolo
            if args.enable_yolo:
                model = YOLO(r'C:\Repo\D4\final\runs\detect\train\weights\best.pt')
                f_yolo = ex.submit(templates.yolo_detect_monsters, img, model, 0.6,
                                   attack_queue, current_hwnd=hwnd)


        if min_pos is None:
            p = partial(actions.move_right, hwnd, duration=0.5)
            print("Enqueue default move_right because min_pos is None:", repr(p))
            move_queue.put(p)
            continue

        if args.show_windows:
            # Get screen width to position windows on the right side
            screen_width = ctypes.windll.user32.GetSystemMetrics(0)
            screen_height = ctypes.windll.user32.GetSystemMetrics(1)

            # Draw and show minimap
            cv2.circle(minimap_img, (int(min_pos[0]), int(min_pos[1])), 5, (255, 0, 0), 2)
            cv2.namedWindow("mini", cv2.WINDOW_NORMAL)
            cv2.imshow("mini", minimap_img)
            cv2.moveWindow("mini", screen_width - 250, 800)  # Top right corner

            # Draw and show main image
            cv2.namedWindow("img", cv2.WINDOW_NORMAL)
            cv2.resizeWindow("img", 800, 600)
            cv2.rectangle(img, (int(big_pos[0]) - 10, int(big_pos[1]) - 10), (int(big_pos[0]) + 10, int(big_pos[1]) + 10), (0, 255, 0), 2)
            cv2.imshow("img", img)
            cv2.moveWindow("img", screen_width - 850, 50)  # Right side, main window

            # Draw and show route image
            route_img_debug = route_img.copy()
            cv2.circle(route_img_debug, (int(min_pos[0]), int(min_pos[1])), 1, (255, 255, 255), 2)
            cv2.namedWindow("route", cv2.WINDOW_NORMAL)
            cv2.imshow("route", route_img_debug)
            cv2.moveWindow("route", screen_width - 500, 700)  # Below minimap

            if cv2.waitKey(1) & 0xFF == 27:
                break

        # buffing decision
        if time.time() - buff_time > args.buffer_time:
            buff_time = time.time()
            p = partial(actions.press_buff, hwnd)
            _enqueue_latest_move(attack_queue, p)

        # check player 换线
        # if flame % 60 == 0 and red_flag:
        #     p = partial(actions.press_sequence, hwnd)
        #     _enqueue_latest_move(attack_queue, p)
        #     time.sleep(10)

        # 不同层级的移动策略
        _, _, route_idx = get_nearest_color_code(route_img, min_pos, move_queue=move_queue, current_hwnd=hwnd, attck_queue=attack_queue, misc_queue=key_queue, route_idx=route_idx)
        route_idx = route_idx % len(routes)

        # simple move 偷懒用平面地图
        # if is_left:
        #     move_target(hwnd, min_pos, (route[0], 0))
        #     if min_pos[0] <= route[0]:
        #         is_left = Falsez
        # else:
        #     move_target(hwnd, min_pos, (route[1], 0))
        #     if min_pos[0] >= route[1]:
        #         is_left = True

        for i in range(3):
            p = partial(actions.press_z_key, hwnd)
            _enqueue_latest_move(key_queue, p)

        # 防卡
        if flame % 10 == 0:
            if prev_pos is not None:
                dist = ((min_pos[0] - prev_pos[0]) ** 2 + (min_pos[1] - prev_pos[1]) ** 2) ** 0.5
                if dist < 2.0:
                    p = partial(actions.move_target, hwnd, cur=min_pos, target=(x, y), is_rope=True, jump_dump=0.2, is_right=True)
                    _enqueue_latest_move(move_queue, p)
            prev_pos = min_pos

        flame += 1
        print(f"Loop time: {time.time() - start:.2f}s, HP: {hp_pct:.1f}%, MP: {mp_pct:.1f}%, red_flag {red_flag}")

    # 退出前，通知按键线程退出并等待队列处理完毕（向两个队列分别发结束信号）
    hp_queue.put(None)
    key_queue.put(None)
    attack_queue.put(None)
    move_queue.put(None)
    hp_queue.join()
    key_queue.join()
    attack_queue.join()
    move_queue.join()

    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()

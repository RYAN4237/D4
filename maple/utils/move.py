import random

from maple.utils.key import *

def random_movement(hwnd, move_count=10):
    """
    随机左右移动
    move_count: 移动次数
    """
    print(f"\n🎲 开始随机移动（{move_count}次）")
    direction = random.choice([0, 1])

    for i in range(move_count):
        # 随机选择方向：0=左，1=右

        # 随机移动时长：0.2-0.5秒
        duration = random.uniform(0.2, 0.5)

        if direction == 0:
            print(f"  [{i + 1}/{move_count}] 向左移动 {duration:.2f}秒")
            press_left_key(hwnd, duration)
        else:
            print(f"  [{i + 1}/{move_count}] 向右移动 {duration:.2f}秒")
            press_right_key(hwnd, duration)

    print(f"✓ 随机移动完成")


def get_character_position(track):
    return track.get_current_minimap_position()


def move_to_target(target, track, hwnd, tolerance=1, key_action = False):
    now = time.time()
    duration = 1
    if key_action:
        duration = 0.05
    while True:
        current = get_character_position(track)
        if not key_action:
            auto_pickup_items(hwnd, pickup_count=1)
        if current is None:
            # 异常处理
            return
        print(f"当前位置 {current}, 目标位置 {target}")
        if time.time() - now >= 8:
            print("x轴位置未变化，可能卡住，退出移动")
            press_jump(hwnd)
            return

        if abs(target[0] - current[0]) < tolerance:
            print("x轴到达目标位置")
            if key_action:
                time.sleep(0.5)
                press_up(hwnd)
            return
        elif current[0] < target[0]:
            press_right_key(hwnd, duration)
        elif current[0] > target[0]:
            press_left_key(hwnd, duration)

def traversal_map(left, right, track, hwnd, key_pos = [], tolerance=8, slice = 10, idx = 0):
    print(f"开始横向遍历地图，范围: {left} - {right}, {slice}")
    diff = (right[0] - left[0]) // slice
    target = [left[0] + ((diff * idx) % (right[0] - left[0])), 0]

    current = get_character_position(track)
    if current is None:
        # 异常处理
        return 0
    # if current[0] < left[0]:
    #     print("角色不在指定范围内，移动到起始位置")
    #     move_to_target(left, track, hwnd, tolerance)
    # elif current[0] > right[0]:
    #     print("角色不在指定范围内，移动到起始位置")
    #     move_to_target(right, track, hwnd, tolerance)


    for pos in key_pos:
        if abs(current[0] - target[0]) <= 30:
            target = pos
            print(f"移动到关键位置 {pos}")
            move_to_target(target, track, hwnd, 1, True)
            return (idx + 1) % slice

    if target[0] < left[0]:
        target[0] = left[0] + diff
    elif target[0] > right[0]:
        target[0] = right[0] - diff


    move_to_target(target, track, hwnd, tolerance)
    return (idx + 1) % slice





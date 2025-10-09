import random
from key import *

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


def move_to_target(target, track, hwnd, tolerance=1):
    last = None
    while True:
        current = get_character_position(track)
        if current[0] < target[0]:
            press_right_key(hwnd, 0.5)
        elif current[0] > target[0]:
            press_left_key(hwnd, 0.5)
        elif target[0] - current[0] <= tolerance:
            print("x轴到达目标位置")
            return
        elif last is not None and last[0] - current[0] < 1:
            print("x轴位置未变化，可能卡住，退出移动")
            press_up(hwnd)
            return
        last = current

def traversal_map(left, right, track, hwnd, key_pos = [], tolerance=1, slice = 10):
    print(f"开始横向遍历地图，范围: {left} - {right}")
    diff = (right[0] - left[0]) // slice
    direction = random.choice([-1, 1])

    current = get_character_position(track)
    if current[0] < left:
        print("角色不在指定范围内，移动到起始位置")
        move_to_target(left, track, hwnd, tolerance)
    elif current[0] > right:
        print("角色不在指定范围内，移动到起始位置")
        move_to_target(right, track, hwnd, tolerance)
    target = current[0] + diff * direction
    if target < left:
        target = left + diff
    elif target > right:
        target = right - diff

    for pos in key_pos:
        if abs(pos[0] - target) <= 30:
            target = pos
            print(f"移动到关键位置 {pos}")
            break
    move_to_target(target, track, hwnd, tolerance)





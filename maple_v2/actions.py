import time
import win32api
import win32con
import win32gui


def press_key(hwnd, key_code, duration=0.05):
    """发送按键"""
    try:
        # print(f"发送按键 {key_code}")
        # win32gui.SetForegroundWindow(hwnd)
        time.sleep(0.02)

        win32api.keybd_event(key_code, 0, 0, 0)
        time.sleep(duration)
        win32api.keybd_event(key_code, 0, win32con.KEYEVENTF_KEYUP, 0)
    except Exception as e:
        print(f"按键发送失败: {e}")


def attack_action(hwnd):
    press_key(hwnd, win32con.VK_SPACE)


def move_left(hwnd, duration=0.5):
    print(f"move_left called: hwnd={hwnd}, duration={duration}")
    press_key(hwnd, win32con.VK_LEFT, duration)


def move_right(hwnd, duration=0.5):
    print(f"move_right called: hwnd={hwnd}, duration={duration}")
    press_key(hwnd, win32con.VK_RIGHT, duration)


def jump(hwnd):
    print(f"jump called: hwnd={hwnd}")
    press_key(hwnd, win32con.VK_MENU, 0.05)  # Alt as jump


def climb_up(hwnd, duration=0.35):
    print(f"climb_up called: hwnd={hwnd}, duration={duration}")
    press_key(hwnd, win32con.VK_UP, duration)


def move_target(hwnd, cur, target, duration=0.4, is_rope=False, jump_dump=0.2, is_left = False, is_right = False):
    """Move left/right to approach target; if is_rope, perform rope climb sequence and return True."""
    # simple move 偷懒是打开
    # if cur[0] < target[0]:
    #     move_right(hwnd, duration)
    # elif cur[0] > target[0]:
    #     move_left(hwnd, duration)

    # real func 偷懒时注释
    if is_rope:
        if is_left:
            move_left(hwnd, 0.1)
        if is_right:
            move_right(hwnd, 0.1)
        jump(hwnd)
        climb_up(hwnd, jump_dump)
        print("move_target performed rope sequence")
        return True
    return False


def press_del_key(hwnd):
    press_key(hwnd, win32con.VK_DELETE)


def press_end_key(hwnd):
    press_key(hwnd, win32con.VK_END)


def press_z_key(hwnd):
    press_key(hwnd, ord('Z'))


def press_sequence(hwnd):
    """Press ESC -> Enter -> Right -> Enter sequence (used in original)."""
    press_key(hwnd, 0x1B)
    time.sleep(0.3)
    press_key(hwnd, 0x0D)
    time.sleep(0.3)
    press_key(hwnd, 0x27)
    time.sleep(0.3)
    press_key(hwnd, 0x0D)

def press_buff(hwnd):
    press_key(hwnd, 0x24)
    time.sleep(0.5)
    press_key(hwnd, 0x2D)
    time.sleep(0.5)
    press_key(hwnd, 0x21)

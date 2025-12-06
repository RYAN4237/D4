import time
from ctypes import *
import random


class Driver:
    def __init__(self):
        self.vk = {'w': 302, 'a': 401, 's': 402, 'd': 403}
        self.left = 0
        self.top = 0
        self.driver = None
        self.x = 0
        self.y = 0
        
    def d_set_ini(self, path_name):
        """
        驱动初始化，成功返回1，失败返回0
        :param path_name: 驱动dll全路径，即使与脚本文件同一个安装路径，也需要写完整路径，比如：X:/XX/XX.dll
        :return:成功返回1，失败返回0
        """
        self.driver = windll.LoadLibrary(path_name)
        st = self.driver.DD_btn(0)
        if st == 1:
            print("驱动初始化成功！")
            self.x = self.left + random.randint(10, 20)
            self.y = self.top + random.randint(10, 20)
            self.driver.DD_mov(self.x, self.y)
            return 1
        else:
            print("驱动初始化失败，请尝试管理员身份运行！")
            return 0
        
    def d_key_press(self, key_name):
        """
        驱动键盘 单击 某键
        :param key_name: 键盘名称，对应键帽上的字符
        :return: 无
        """
        # print(self.vk[key_name])
        self.driver.DD_key(self.vk[key_name], 1)
        time.sleep(0.5)
        self.driver.DD_key(self.vk[key_name], 2)
        time.sleep(0.03)


    def move(self, x, y, path, offset = 3):
        if len(path) < 2:
            return
        target_x, target_y = path.pop(2)
        # if abs(target_x - x) <= offset and abs(target_y - y) <= offset:
        #     return
        if target_x >= x + offset:

            self.d_key_press('d')
        elif target_x <= x - offset:

            self.d_key_press('a')
        elif target_y >= y + offset:

            self.d_key_press('s')
        elif target_y <= y - offset:


            self.d_key_press('w')




if __name__ == "__main__":
    time.sleep(1)
    driver = Driver()
    driver.d_set_ini(r"C:\Repo\D4\poe2\driver.dll")
    while True:
        driver.d_key_press("w")
        time.sleep(1)
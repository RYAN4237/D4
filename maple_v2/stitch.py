"""
智能小地图拼接 - 带实时监控版本
只在检测到真实移动时才拼接，避免累积误差

解决的问题：
1. 增量拼接：只拼接新增的部分，避免边界重叠导致的地图畸形
2. 多层验证：使用状态机和自适应阈值，避免has_moved误判导致的拼接缺失
"""
import cv2
import numpy as np
from JYMOKUAI import *
import time
from collections import deque


class SmartMinimapStitcher:
    def __init__(self, x1, y1, x2, y2):
        self.jy = Jy_screen()
        self.jy.p_bind("MapleStory Worlds-Old School Maple")
        time.sleep(0.5)

        self.x1, self.y1 = x1, y1
        self.x2, self.y2 = x2, y2

        # 创建画布
        self.canvas_size = 3000
        self.canvas = np.ones((self.canvas_size, self.canvas_size, 3), dtype=np.uint8) * 200

        # 初始化位置（画布中心）
        self.canvas_x = self.canvas_size // 2
        self.canvas_y = self.canvas_size // 2

        # 统计信息
        self.frame_count = 0
        self.stitch_count = 0
        self.last_minimap = None

        # ========== 问题2优化：自适应阈值 + 状态机 ==========
        # 移动历史（用于计算自适应阈值）
        self.mean_diff_history = deque(maxlen=30)  # 最近30帧的mean_diff
        self.movement_history = deque(maxlen=5)    # 最近5帧的(dx, dy, response)

        # Kalman滤波器（平滑dx/dy，消除抖动）
        self.smooth_dx = 0.0
        self.smooth_dy = 0.0
        self.smooth_response = 0.0

        # 状态机
        self.state = "IDLE"  # IDLE, MOVING, STOPPED
        self.state_confidence = 0  # 状态确认度[0,1]
        self.no_move_frames = 0

        # ========== 问题1优化：特征匹配 ==========
        # ORB特征检测器
        self.orb = cv2.ORB_create(nfeatures=500)
        self.bf_matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)

        print("SmartMinimapStitcher initialized successfully!")

    def capture_minimap(self):
        """截取小地图"""
        img = self.jy.call_capture()
        return img[self.y1:self.y2, self.x1:self.x2]

    def detect_movement(self, gray1, gray2, threshold=10):
        """
        检测两帧之间是否有真实移动（优化版：自适应阈值+状态机+Kalman滤波）

        Args:
            gray1: 上一帧灰度图
            gray2: 当前帧灰度图
            threshold: 基础阈值

        Returns:
            (has_moved, smooth_dx, smooth_dy, confidence, state)
        """
        # ===== 第一层：计算基础的mean_diff =====
        mean_diff = np.mean(np.abs(gray1.astype(np.float32) - gray2.astype(np.float32)))

        # 提高阈值，因为去掉边框后应该更稳定
        if mean_diff < threshold:
            return False, 0, 0, 0, self.state

        # ===== 第二层：相位相关检测精确位移 =====
        try:
            shift, response = cv2.phaseCorrelate(
                np.float32(gray1),
                np.float32(gray2)
            )
        except Exception as e:
            print(f"相位相关检测异常: {e}")
            return False, 0, 0, 0, self.state

        dx, dy = float(shift[0]), float(shift[1])

        # 如果位移太小，认为没有移动
        if abs(dx) < 1 and abs(dy) < 1:
            return False, 0, 0, response, self.state

        if response < 0.5:
            return False, 0, 0, response, self.state

        # 记录历史用于自适应阈值
        self.mean_diff_history.append(mean_diff)

        # ===== 第三层：计算自适应阈值 =====
        # 根据最近30帧的mean_diff，计算自适应阈值
        # 公式: baseline + 1.5倍标准差（适应光照变化）
        if len(self.mean_diff_history) >= 10:
            baseline = np.mean(list(self.mean_diff_history))
            std_dev = np.std(list(self.mean_diff_history))

            # 根据状态调整倍数
            if self.state == "IDLE":
                adaptive_threshold = baseline + 1.5 * std_dev  # 高要求
            elif self.state == "MOVING":
                adaptive_threshold = baseline + 0.8 * std_dev  # 低要求
            else:  # STOPPED
                adaptive_threshold = baseline + 1.2 * std_dev  # 中等要求
        else:
            # 初始化阶段，用固定阈值
            adaptive_threshold = threshold

        # 快速检查：如果mean_diff过低，直接判定无移动
        if mean_diff < adaptive_threshold * 0.5:
            self.no_move_frames += 1
            return False, 0, 0, 0, self.state

        # ===== 第四层：Kalman滤波平滑dx/dy =====
        # 简化的Kalman滤波：新值权重30%，历史值权重70%
        # 这样可以消除0.5像素级别的噪声抖动
        smooth_dx = 0.7 * self.smooth_dx + 0.3 * dx
        smooth_dy = 0.7 * self.smooth_dy + 0.3 * dy
        smooth_response = 0.6 * self.smooth_response + 0.4 * response

        # ===== 第五层：多条件验证（AND逻辑）=====
        conditions = {
            "mean_diff": mean_diff > adaptive_threshold,
            "min_movement": abs(smooth_dx) >= 0.5 or abs(smooth_dy) >= 0.5,
            "confidence": smooth_response > self._get_response_threshold(),
            "movement_valid": abs(dx) <= 100 and abs(dy) <= 100,  # 过滤极端值
        }

        has_moved = all(conditions.values())

        # ===== 第六层：状态机转移 =====
        if has_moved:
            # 移动被确认
            if self.state == "IDLE":
                self.state = "MOVING"
                self.state_confidence = 0.5
            elif self.state == "MOVING":
                self.state_confidence = min(1.0, self.state_confidence + 0.1)
            elif self.state == "STOPPED":
                self.state = "MOVING"
                self.state_confidence = 0.5

            self.no_move_frames = 0

        else:
            # 无移动被检测
            self.no_move_frames += 1

            # 从MOVING转到STOPPED（连续3帧无移动）
            if self.state == "MOVING" and self.no_move_frames >= 3:
                self.state = "STOPPED"
                self.state_confidence = 0.5

            # 从STOPPED回到IDLE（连续5帧无移动）
            if self.state == "STOPPED" and self.no_move_frames >= 5:
                self.state = "IDLE"
                self.state_confidence = 0

            smooth_dx = 0
            smooth_dy = 0

        # ===== 保存平滑值供下次使用 =====
        self.smooth_dx = smooth_dx
        self.smooth_dy = smooth_dy
        self.smooth_response = smooth_response

        # 记录移动历史
        self.movement_history.append((smooth_dx, smooth_dy, smooth_response))

        return has_moved, smooth_dx, smooth_dy, smooth_response, self.state

    def run(self):
        """主循环"""
        minimap = self.capture_minimap()
        h, w = minimap.shape[:2]

        if self.last_minimap is None:
            self.last_minimap = minimap
            return False, 0, 0, 0, self.state

        # 转灰度图进行运动检测
        gray1 = cv2.cvtColor(self.last_minimap, cv2.COLOR_BGR2GRAY)
        gray2 = cv2.cvtColor(minimap, cv2.COLOR_BGR2GRAY)

        # 检测移动
        has_moved, smooth_dx, smooth_dy, response, state = self.detect_movement(gray1, gray2)

        if has_moved:
            # 拼接
            self.stitch(minimap, self.last_minimap, smooth_dx, smooth_dy, response)
            self.stitch_count += 1

        self.last_minimap = minimap
        self.frame_count += 1

        return has_moved, smooth_dx, smooth_dy, response, state

    def _get_response_threshold(self):
        """根据状态返回对应的置信度阈值"""
        thresholds = {
            "IDLE": 0.80,      # 严格要求，avoid误触
            "MOVING": 0.60,    # 宽松要求，快速响应
            "STOPPED": 0.70,   # 中等要求，验证停止
        }
        return thresholds.get(self.state, 0.70)

    def stitch(self, minimap, last_minimap, dx, dy, confidence):
        """
        将小地图拼接到画布上（优化版：增量拼接+特征匹配+渐变融合）

        问题1解决：使用增量拼接而不是全覆盖，避免边界重叠导致的畸形

        Args:
            minimap: 新帧
            last_minimap: 上一帧
            dx, dy: 检测到的位移
            confidence: 位移置信度

        Returns:
            success: 是否拼接成功
        """
        h, w = minimap.shape[:2]

        # ===== 高置信度：直接增量拼接（性能优先）=====
        if confidence > 0.85:
            # 只复制新增的移动部分，而不是整个小地图
            # 这避免了边界重叠导致的色差和畸形

            # 计算新增区域
            if dx != 0:
                if dx > 0:
                    # 向右移动，复制左边的新内容
                    new_left = 0
                    new_right = min(abs(int(dx)), w)
                    canvas_left = self.canvas_x
                else:
                    # 向左移动，复制右边的新内容
                    new_left = max(0, w + int(dx))
                    new_right = w
                    canvas_left = self.canvas_x + w + int(dx)
            else:
                new_left = 0
                new_right = w
                canvas_left = self.canvas_x

            if dy != 0:
                if dy > 0:
                    # 向下移动，复制上边的新内容
                    new_top = 0
                    new_bottom = min(abs(int(dy)), h)
                    canvas_top = self.canvas_y
                else:
                    # 向上移动，复制下边的新内容
                    new_top = max(0, h + int(dy))
                    new_bottom = h
                    canvas_top = self.canvas_y + h + int(dy)
            else:
                new_top = 0
                new_bottom = h
                canvas_top = self.canvas_y

            # 提取新增区域
            new_region = minimap[new_top:new_bottom, new_left:new_right]
            canvas_h = new_bottom - new_top
            canvas_w = new_right - new_left

            # 边界检查
            canvas_right = canvas_left + canvas_w
            canvas_bottom = canvas_top + canvas_h

            if (canvas_left >= 0 and canvas_top >= 0 and
                canvas_right <= self.canvas_size and canvas_bottom <= self.canvas_size):

                # 对新增区域进行渐变融合（边界羽化，避免色差）
                if (dx != 0 or dy != 0) and confidence > 0.75:
                    # 创建渐变权重（边界处权重低，内部权重高）
                    blend_region = self._blend_region(
                        new_region,
                        self.canvas[canvas_top:canvas_bottom, canvas_left:canvas_right],
                        confidence,
                        dx=dx if dx != 0 else 0,
                        dy=dy if dy != 0 else 0
                    )
                    self.canvas[canvas_top:canvas_bottom, canvas_left:canvas_right] = blend_region
                else:
                    # 直接覆盖
                    self.canvas[canvas_top:canvas_bottom, canvas_left:canvas_right] = new_region

                return True
            else:
                print(f"  ⚠️ 警告: 增量拼接超出画布范围 ({canvas_left},{canvas_top})-({canvas_right},{canvas_bottom})")
                return False

        # ===== 低置信度：特征匹配对齐 + 完全拼接 =====
        else:
            # 使用ORB特征进行精确对齐
            try:
                # 检测特征点
                kp1, des1 = self.orb.detectAndCompute(last_minimap, None)
                kp2, des2 = self.orb.detectAndCompute(minimap, None)

                if des1 is None or des2 is None or len(kp1) < 4 or len(kp2) < 4:
                    # 特征不足，使用普通方式拼接
                    return self._simple_stitch(minimap, dx, dy)

                # 特征匹配
                matches = self.bf_matcher.knnMatch(des1, des2, k=2)

                # Lowe's ratio test，保留置信度高的匹配
                good_matches = []
                for match_pair in matches:
                    if len(match_pair) == 2:
                        m, n = match_pair
                        if m.distance < 0.75 * n.distance:
                            good_matches.append(m)

                if len(good_matches) < 4:
                    # 匹配点不足，使用普通方式
                    return self._simple_stitch(minimap, dx, dy)

                # 计算仿射变换矩阵
                pts1 = np.float32([kp1[m.queryIdx].pt for m in good_matches])
                pts2 = np.float32([kp2[m.trainIdx].pt for m in good_matches])

                # 使用RANSAC计算准确的变换
                matrix, mask = cv2.estimateAffinePartial2D(pts1, pts2, method=cv2.RANSAC)

                if matrix is None:
                    return self._simple_stitch(minimap, dx, dy)

                # 计算校正后的位移
                corrected_dx = matrix[0, 2]
                corrected_dy = matrix[1, 2]

                # 用校正后的位移
                return self._simple_stitch(minimap, corrected_dx, corrected_dy)

            except Exception as e:
                print(f"  警告: 特征匹配失败 - {e}")
                return self._simple_stitch(minimap, dx, dy)

    def _simple_stitch(self, minimap, dx, dy):
        """简单拼接模式：直接覆盖整个小地图到指定位置"""
        h, w = minimap.shape[:2]

        # 更新画布位置
        self.canvas_x -= int(dx)
        self.canvas_y -= int(dy)

        # 进行拼接
        canvas_x = self.canvas_x
        canvas_y = self.canvas_y
        canvas_right = canvas_x + w
        canvas_bottom = canvas_y + h

        # 边界检查和裁剪
        if canvas_x < 0:
            minimap = minimap[:, -canvas_x:]
            canvas_x = 0
        if canvas_y < 0:
            minimap = minimap[-canvas_y:, :]
            canvas_y = 0
        if canvas_right > self.canvas_size:
            minimap = minimap[:, :self.canvas_size - canvas_x]
        if canvas_bottom > self.canvas_size:
            minimap = minimap[:self.canvas_size - canvas_y, :]

        h, w = minimap.shape[:2]
        canvas_right = canvas_x + w
        canvas_bottom = canvas_y + h

        if h > 0 and w > 0 and canvas_x < self.canvas_size and canvas_y < self.canvas_size:
            self.canvas[canvas_y:canvas_bottom, canvas_x:canvas_right] = minimap
            return True

        return False

    def _blend_region(self, new_region, old_region, confidence, dx=0, dy=0):
        """
        对两个区域进行渐变融合，避免边界色差

        Args:
            new_region: 新区域
            old_region: 旧区域
            confidence: 置信度[0,1]
            dx, dy: 移动方向

        Returns:
            融合后的区域
        """
        h, w = new_region.shape[:2]

        # 创建权重矩阵（边界处权重低，内部权重高）
        weight_map = np.ones((h, w, 1), dtype=np.float32) * confidence

        # 根据移动方向对边界进行羽化
        fade_width = max(2, min(h, w) // 8)  # 羽化宽度

        if dx != 0:  # 水平移动
            if dx > 0:  # 向右移动，左边界羽化
                for x in range(fade_width):
                    weight_map[:, x] *= (x / fade_width)
            else:  # 向左移动，右边界羽化
                for x in range(fade_width):
                    weight_map[:, w - 1 - x] *= (x / fade_width)

        if dy != 0:  # 垂直移动
            if dy > 0:  # 向下移动，上边界羽化
                for y in range(fade_width):
                    weight_map[y, :] *= (y / fade_width)
            else:  # 向上移动，下边界羽化
                for y in range(fade_width):
                    weight_map[h - 1 - y, :] *= (y / fade_width)

        # 融合
        blended = (new_region.astype(np.float32) * weight_map +
                   old_region.astype(np.float32) * (1 - weight_map)).astype(np.uint8)

        return blended

    def get_canvas(self):
        """获取当前画布"""
        return self.canvas

    def get_stats(self):
        """获取统计信息"""
        return {
            "frame_count": self.frame_count,
            "stitch_count": self.stitch_count,
            "state": self.state,
            "state_confidence": self.state_confidence
        }
    #


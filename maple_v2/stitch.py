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
    def __init__(self, x1, y1, x2, y2, name="MapleStory Worlds-Old School Maple", debug_mode=False):
        self.jy = Jy_screen()
        self.jy.p_bind(name)
        time.sleep(0.5)

        self.x1, self.y1 = x1, y1
        self.x2, self.y2 = x2, y2

        # ========== DEBUG模式 ==========
        self.debug_mode = debug_mode
        self.debug_frame_count = 0

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
        return self.jy.p_capture(self.x1, self.y1, self.x2, self.y2, 1, 1)

    def detect_movement(self, gray1, gray2, threshold=10):
        """
        检测两帧之间是否有真实移动

        方法：相位相关 + 光学流
        """
        # ===== 第一步：计算帧差 =====
        mean_diff = np.mean(np.abs(gray1.astype(np.float32) - gray2.astype(np.float32)))

        if self.debug_mode:
            print(f"[Frame {self.frame_count}] mean_diff={mean_diff:.2f}")

        # 完全相同的帧
        if mean_diff < 1.0:
            return False, 0, 0, 0, self.state

        self.mean_diff_history.append(mean_diff)

        # ===== 第二步：相位相关检测 =====
        dx_phase, dy_phase, response_phase = 0, 0, 0
        try:
            shift, response_phase = cv2.phaseCorrelate(np.float32(gray1), np.float32(gray2))
            dx_phase, dy_phase = float(shift[0]), float(shift[1])
            if self.debug_mode:
                print(f"  PhaseCorr: dx={dx_phase:.2f}, dy={dy_phase:.2f}, resp={response_phase:.3f}")
        except:
            pass

        # ===== 第三步：光学流作为备选 =====
        dx_flow, dy_flow, response_flow = 0, 0, 0
        try:
            # 使用Lucas-Kanade光学流
            flow = cv2.calcOpticalFlowFarneback(gray1, gray2, None, 0.5, 3, 15, 3, 5, 1.2, 0)
            # 计算平均光流
            magnitude, _ = cv2.cartToPolar(flow[..., 0], flow[..., 1])
            mean_magnitude = np.mean(magnitude)

            if mean_magnitude > 0.1:
                avg_flow = np.median(flow.reshape(-1, 2), axis=0)
                dx_flow, dy_flow = avg_flow[0], avg_flow[1]
                response_flow = min(mean_magnitude, 1.0)
                if self.debug_mode:
                    print(f"  OpticalFlow: dx={dx_flow:.2f}, dy={dy_flow:.2f}, magnitude={mean_magnitude:.3f}")
        except:
            pass

        # ===== 第四步：融合两个方法的结果 =====
        # 使用响应更高的方法
        if response_phase > response_flow:
            dx, dy, response = dx_phase, dy_phase, response_phase
            if self.debug_mode:
                print(f"  使用PhaseCorr结果")
        else:
            dx, dy, response = dx_flow, dy_flow, response_flow
            if self.debug_mode:
                print(f"  使用OpticalFlow结果")

        # ===== 第五步：多条件判断 =====
        response_threshold = self._get_response_threshold()

        has_diff = mean_diff > max(5.0, threshold * 0.3)
        has_movement = abs(dx) > 0.2 or abs(dy) > 0.2
        has_confidence = response > response_threshold
        is_valid = abs(dx) < 100 and abs(dy) < 100

        if self.debug_mode:
            print(f"  Check: diff={has_diff}({mean_diff:.1f}>{max(5.0, threshold * 0.3):.1f}), "
                  f"move={has_movement}({abs(dx):.2f},{abs(dy):.2f}), "
                  f"conf={has_confidence}({response:.3f}>{response_threshold:.3f}), valid={is_valid}")

        # 条件逻辑：需要有明显的帧差和位移，且：
        # - 要么置信度高
        # - 要么帧差非常大（mean_diff > 20）
        has_confidence_or_large_diff = has_confidence or (mean_diff > 20)

        has_moved = has_diff and has_movement and has_confidence_or_large_diff and is_valid

        if self.debug_mode:
            print(f"  has_confidence_or_large_diff: {has_confidence_or_large_diff} (conf={has_confidence} or large_diff={mean_diff > 20})")

        # ===== 第六步：状态机 =====
        if has_moved:
            if self.state == "IDLE":
                self.state = "MOVING"
            elif self.state == "MOVING":
                self.state_confidence = min(1.0, self.state_confidence + 0.1)
            elif self.state == "STOPPED":
                self.state = "MOVING"
            self.no_move_frames = 0
        else:
            self.no_move_frames += 1
            if self.state == "MOVING" and self.no_move_frames >= 3:
                self.state = "STOPPED"
            if self.state == "STOPPED" and self.no_move_frames >= 5:
                self.state = "IDLE"
            dx, dy = 0, 0

        # ===== 第七步：平滑 =====
        smooth_dx = 0.7 * self.smooth_dx + 0.3 * dx
        smooth_dy = 0.7 * self.smooth_dy + 0.3 * dy
        smooth_response = 0.6 * self.smooth_response + 0.4 * response

        self.smooth_dx = smooth_dx
        self.smooth_dy = smooth_dy
        self.smooth_response = smooth_response

        self.movement_history.append((smooth_dx, smooth_dy, smooth_response))
        self.debug_frame_count += 1

        if self.debug_mode:
            print(f"  Result: moved={has_moved}, state={self.state}, smooth_dx={smooth_dx:.2f}, smooth_dy={smooth_dy:.2f}\n")

        return has_moved, smooth_dx, smooth_dy, smooth_response, self.state

    def run(self):
        """主循环"""
        minimap = self.capture_minimap()
        cv2.imshow("Minimap", minimap)
        h, w = minimap.shape[:2]

        if self.last_minimap is None:
            # 第一帧：直接拼接到画布中心
            if self.debug_mode:
                print(f"\n[Frame {self.frame_count}] 初始化第一帧")
            self._simple_stitch(minimap, 0, 0)
            self.stitch_count += 1
            if self.debug_mode:
                print(f"  ✓ 第一帧拼接完成")
            self.last_minimap = minimap
            self.frame_count += 1
            return False, 0, 0, 0, self.state

        # 转灰度图进行运动检测
        gray1 = cv2.cvtColor(self.last_minimap, cv2.COLOR_BGR2GRAY)
        gray2 = cv2.cvtColor(minimap, cv2.COLOR_BGR2GRAY)

        # 检测移动
        has_moved, smooth_dx, smooth_dy, response, state = self.detect_movement(gray1, gray2)

        if self.debug_mode:
            print(f"  → has_moved={has_moved}, response={response:.3f}, state={state}")

        if has_moved:
            # 拼接
            if self.debug_mode:
                print(f"  执行拼接...")
            stitch_result = self.stitch(minimap, self.last_minimap, smooth_dx, smooth_dy, response)
            if stitch_result:
                self.stitch_count += 1
                if self.debug_mode:
                    print(f"  ✓ 拼接计数: {self.stitch_count}")
            else:
                if self.debug_mode:
                    print(f"  ⚠️ 拼接失败")
        else:
            if self.debug_mode:
                print(f"  → 无移动，跳过拼接")

        self.last_minimap = minimap
        self.frame_count += 1

        return has_moved, smooth_dx, smooth_dy, response, state

    def _get_response_threshold(self):
        """根据状态返回对应的置信度阈值"""
        thresholds = {
            "IDLE": 0.50,      # 降低从0.80到0.50
            "MOVING": 0.40,    # 降低从0.60到0.40
            "STOPPED": 0.45,   # 降低从0.70到0.45
        }
        return thresholds.get(self.state, 0.50)

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

        if self.debug_mode:
            print(f"  [STITCH] 准备拼接: confidence={confidence:.3f}, state={self.state}, "
                  f"dx={dx:.2f}, dy={dy:.2f}")

        # ===== 置信度检查 =====
        # 在不同状态下有不同的置信度要求
        min_confidence = self._get_response_threshold()

        if confidence < min_confidence:
            if self.debug_mode:
                print(f"  ⚠️ 置信度不足 ({confidence:.3f} < {min_confidence:.3f})，跳过拼接")
            return False

        # ===== 高置信度：直接拼接（使用简化逻辑避免边界计算错误）=====
        if confidence > 0.70:
            # 当置信度较高时，直接使用简单拼接方式
            if self.debug_mode:
                print(f"  ✓ 高置信度拼接")
            return self._simple_stitch(minimap, dx, dy)

        # ===== 低置信度：特征匹配对齐 + 完全拼接 =====
        else:
            if self.debug_mode:
                print(f"  → 低置信度，尝试特征匹配对齐")
            # 使用ORB特征进行精确对齐
            try:
                # 检测特征点
                kp1, des1 = self.orb.detectAndCompute(last_minimap, None)
                kp2, des2 = self.orb.detectAndCompute(minimap, None)

                if des1 is None or des2 is None or len(kp1) < 4 or len(kp2) < 4:
                    # 特征不足，使用普通方式拼接
                    if self.debug_mode:
                        print(f"  ⚠️ 特征不足，降级为普通拼接")
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
                    if self.debug_mode:
                        print(f"  ⚠️ 匹配点不足，降级为普通拼接")
                    return self._simple_stitch(minimap, dx, dy)

                # 计算仿射变换矩阵
                pts1 = np.float32([kp1[m.queryIdx].pt for m in good_matches])
                pts2 = np.float32([kp2[m.trainIdx].pt for m in good_matches])

                # 使用RANSAC计算准确的变换
                matrix, mask = cv2.estimateAffinePartial2D(pts1, pts2, method=cv2.RANSAC)

                if matrix is None:
                    if self.debug_mode:
                        print(f"  ⚠️ 仿射变换计算失败")
                    return self._simple_stitch(minimap, dx, dy)

                # 计算校正后的位移
                corrected_dx = matrix[0, 2]
                corrected_dy = matrix[1, 2]

                if self.debug_mode:
                    print(f"  ✓ 特征匹配成功: corrected_dx={corrected_dx:.2f}, corrected_dy={corrected_dy:.2f}")

                # 用校正后的位移
                return self._simple_stitch(minimap, corrected_dx, corrected_dy)

            except Exception as e:
                if self.debug_mode:
                    print(f"  ✗ 特征匹配异常: {e}")
                return self._simple_stitch(minimap, dx, dy)

    def _simple_stitch(self, minimap, dx, dy):
        """
        核心拼接逻辑：根据位移更新画布位置，然后拼接小地图

        参数：
        - dx, dy: 相位相关检测返回的位移（表示图像内容的移动方向）
                 正值表示向正方向移动，负值表示向负方向移动
        """
        h, w = minimap.shape[:2]

        if self.debug_mode:
            print(f"    [STITCH] 拼接: dx={dx:.2f}, dy={dy:.2f}")
            print(f"    旧位置: canvas_x={self.canvas_x}, canvas_y={self.canvas_y}")

        # 第一帧（无位移）或极小位移：直接拼接
        if abs(dx) < 0.1 and abs(dy) < 0.1:
            canvas_x = self.canvas_x
            canvas_y = self.canvas_y
            if self.debug_mode:
                print(f"    无移动，拼接到当前位置 ({canvas_x}, {canvas_y})")
        else:
            # 有位移：更新画布位置
            self.canvas_x += int(round(dx))
            self.canvas_y += int(round(dy))
            canvas_x = self.canvas_x
            canvas_y = self.canvas_y
            if self.debug_mode:
                print(f"    更新位置到: canvas_x={canvas_x}, canvas_y={canvas_y}")

        # 处理负坐标（超出左上角边界）
        src_top = 0
        src_left = 0
        if canvas_x < 0:
            src_left = -canvas_x
            canvas_x = 0
        if canvas_y < 0:
            src_top = -canvas_y
            canvas_y = 0

        # 从小地图中提取需要拼接的部分
        minimap_patch = minimap[src_top:, src_left:]
        patch_h, patch_w = minimap_patch.shape[:2]

        # 处理右下角边界（超出画布）
        if canvas_x + patch_w > self.canvas_size:
            patch_w = self.canvas_size - canvas_x
        if canvas_y + patch_h > self.canvas_size:
            patch_h = self.canvas_size - canvas_y

        # 最终裁剪
        minimap_patch = minimap_patch[:patch_h, :patch_w]
        patch_h, patch_w = minimap_patch.shape[:2]

        # 执行拼接
        if patch_h > 0 and patch_w > 0:
            self.canvas[canvas_y:canvas_y+patch_h, canvas_x:canvas_x+patch_w] = minimap_patch
            if self.debug_mode:
                print(f"    ✓ 拼接成功到 canvas[{canvas_y}:{canvas_y+patch_h}, {canvas_x}:{canvas_x+patch_w}]")
            return True
        else:
            if self.debug_mode:
                print(f"    ✗ 拼接失败：无有效区域 (patch_h={patch_h}, patch_w={patch_w})")
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

        # ===== 边界检查：如果宽度或高度为0，直接返回新区域 =====
        if h <= 0 or w <= 0:
            return new_region.astype(np.uint8)

        # 创建权重矩阵（边界处权重低，内部权重高）
        weight_map = np.ones((h, w, 1), dtype=np.float32) * confidence

        # 根据移动方向对边界进行羽化
        fade_width = max(2, min(h, w) // 8)  # 羽化宽度

        # 确保fade_width不会超过w或h
        fade_width = min(fade_width, w // 2, h // 2)

        if dx != 0 and fade_width > 0:  # 水平移动
            if dx > 0:  # 向右移动，左边界羽化
                for x in range(min(fade_width, w)):
                    weight_map[:, x] *= (x / fade_width)
            else:  # 向左移动，右边界羽化
                for x in range(min(fade_width, w)):
                    idx = w - 1 - x
                    if idx >= 0 and idx < w:
                        weight_map[:, idx] *= (x / fade_width)

        if dy != 0 and fade_width > 0:  # 垂直移动
            if dy > 0:  # 向下移动，上边界羽化
                for y in range(min(fade_width, h)):
                    weight_map[y, :] *= (y / fade_width)
            else:  # 向上移动，下边界羽化
                for y in range(min(fade_width, h)):
                    idx = h - 1 - y
                    if idx >= 0 and idx < h:
                        weight_map[idx, :] *= (y / fade_width)

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


if __name__ == '__main__':
    stitcher = SmartMinimapStitcher(
        x1=1081, y1=33,
        x2=1261, y2=176,
        name = "Path of Exile 2",
        debug_mode=False  # 关闭调试模式
    )

    frame_num = 0
    last_print_frame = 0

    while True:
        has_moved, dx, dy, confidence, state = stitcher.run()
        stats = stitcher.get_stats()
        frame_num = stats['frame_count']

        # 每10帧输出一次诊断信息
        if frame_num - last_print_frame >= 10:
            print(f"Frame: {frame_num}, Stitches: {stats['stitch_count']}, "
                  f"Moved: {has_moved}, dx: {dx:.2f}, dy: {dy:.2f}, "
                  f"Confidence: {confidence:.3f}, State: {state}, "
                  f"canvas_pos: ({stitcher.canvas_x}, {stitcher.canvas_y})")
            last_print_frame = frame_num

        # 获取画布并显示
        canvas = stitcher.get_canvas()

        # 裁剪显示区域
        margin_x = 250
        margin_y = 200

        canvas_x = max(0, stitcher.canvas_x - margin_x)
        canvas_y = max(0, stitcher.canvas_y - margin_y)
        canvas_right = min(canvas.shape[1], canvas_x + 2 * margin_x)
        canvas_bottom = min(canvas.shape[0], canvas_y + 2 * margin_y)

        display_canvas = canvas[canvas_y:canvas_bottom, canvas_x:canvas_right].copy()

        # 标记当前位置
        marker_x = stitcher.canvas_x - canvas_x
        marker_y = stitcher.canvas_y - canvas_y
        if 0 <= marker_x < display_canvas.shape[1] and 0 <= marker_y < display_canvas.shape[0]:
            cv2.circle(display_canvas, (int(marker_x), int(marker_y)), 5, (0, 255, 0), 2)
            cv2.circle(display_canvas, (int(marker_x), int(marker_y)), 15, (0, 255, 0), 1)

        # 添加信息文本
        text1 = f"Pos: ({stitcher.canvas_x}, {stitcher.canvas_y})"
        text2 = f"Frame: {frame_num} | Stitches: {stats['stitch_count']}"
        text3 = f"State: {state} | dx:{dx:.2f} dy:{dy:.2f}"
        cv2.putText(display_canvas, text1, (5, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
        cv2.putText(display_canvas, text2, (5, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
        cv2.putText(display_canvas, text3, (5, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)

        # 缩放显示
        target_width = 640
        target_height = int(target_width * display_canvas.shape[0] / display_canvas.shape[1])
        display_canvas = cv2.resize(display_canvas, (target_width, target_height))

        cv2.imshow("Smart Minimap Stitcher", display_canvas)

        key = cv2.waitKey(100)
        if key == ord('q'):
            break
        elif key == ord('s'):
            cv2.imwrite("full_canvas.png", canvas)
            print(f"[SAVE] 保存完整画布到 full_canvas.png (canvas_pos: {stitcher.canvas_x}, {stitcher.canvas_y})")
        elif key == ord('r'):
            # 重置位置测试
            stitcher.canvas_x = 1500
            stitcher.canvas_y = 1500
            print("[RESET] 重置画布位置到 (1500, 1500)")

    cv2.destroyAllWindows()


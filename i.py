# -*- coding: utf-8 -*-
import cv2
import mediapipe as mp
import moderngl
import pygame
import numpy as np
import math
import threading
import queue
from collections import deque
import time
import random
import json
import sys
import os
from PIL import Image

# =============================================================================
# 1. 設定管理クラス
# =============================================================================
class Config:
    """config.jsonから設定を読み込み、管理するクラス"""
    def __init__(self, path='config.json'):
        try:
            with open(path, 'r', encoding='utf-8') as f:
                self._data = json.load(f)
        except (FileNotFoundError, json.JSONDecodeError) as e:
            print(f"[エラー] 設定ファイル '{path}' が見つからないか、形式が正しくありません。: {e}")
            sys.exit(1)

    def get(self, *keys, default=None):
        """ネストしたキーで設定値を取得する"""
        try:
            value = self._data
            for key in keys:
                value = value[key]
            return value
        except KeyError:
            return default

# =============================================================================
# 2. 3Dオブジェクトとシーン管理クラス
# =============================================================================
class BaseSphere:
    """全ての球体の基底クラス"""
    def __init__(self, position, radius, color=None):
        self.position = np.array(position, dtype=np.float32)
        self.original_position = self.position.copy()
        self.radius = radius
        self.current_radius = radius # 予兆表現などで変化する半径
        self.color = color if color is not None else np.array([1.0, 1.0, 1.0], dtype=np.float32)
        self.rotation = np.array([0.0, 0.0, 0.0], dtype=np.float32)
        self.rotation_speed = np.array([random.uniform(-0.6, 0.6) for _ in range(3)], dtype=np.float32)
        self.alpha = 1.0
        self.visible = True

    def update(self, dt, current_time):
        self.rotation += self.rotation_speed * dt * 0.15

class DisappearingSphere(BaseSphere):
    """触れると消え、予兆や連鎖反応を起こす球体のクラス"""
    def __init__(self, position, radius, color, config):
        super().__init__(position, radius, color)
        self.config = config
        self.state = 'visible'  # 'visible', 'hover', 'disappearing', 'invisible', 'appearing'
        self.disappear_time = 0.0
        self.respawn_delay = random.uniform(3.0, 6.0)
        self.fade_speed = 4.0
        self.collision_radius = radius * 5.0 # 当たり判定の半径
        self.float_offset = random.uniform(0, 2 * math.pi)
        self.hover_scale = self.config.get('interaction', 'hover_scale', default=1.5)
        self.is_chain_reacted = False # 連鎖反応済みか

    def update(self, dt, current_time):
        super().update(dt, current_time)
        
        # ふわふわと浮遊する動き
        float_y = math.sin(current_time * 0.5 + self.float_offset) * 0.01
        self.position[1] = self.original_position[1] + float_y

        # 状態に応じたアルファ値と半径の更新
        if self.state == 'disappearing':
            self.alpha -= self.fade_speed * dt
            if self.alpha <= 0.0:
                self.alpha, self.state, self.disappear_time = 0.0, 'invisible', current_time
                self.is_chain_reacted = False # リセット
        elif self.state == 'invisible':
            if current_time - self.disappear_time >= self.respawn_delay:
                self.state = 'appearing'
        elif self.state == 'appearing':
            self.alpha += self.fade_speed * dt
            if self.alpha >= 1.0:
                self.alpha, self.state = 1.0, 'visible'
        elif self.state == 'hover':
            # ホバー状態の見た目（少し大きくする）
            target_radius = self.radius * self.hover_scale
            self.current_radius += (target_radius - self.current_radius) * 10.0 * dt
            self.alpha = 1.0
        elif self.state == 'visible':
            # 通常状態の見た目
            target_radius = self.radius
            self.current_radius += (target_radius - self.current_radius) * 10.0 * dt
            self.alpha = 0.9 + 0.1 * math.sin(current_time * 1.0)

class HandIndicatorSphere(BaseSphere):
    """手の位置を示すインジケーター球体"""
    def __init__(self, position, radius, color=None):
        super().__init__(position, radius, color)
        self.alpha = 0.8
        self.visible = False
        # ハンドインジケーターはテクスチャを使わないので、色を明るくする
        self.color = np.array([1.0, 1.0, 0.0], dtype=np.float32)

    def update(self, dt, current_time, target_position=None):
        super().update(dt, current_time)
        self.visible = target_position is not None
        if self.visible:
            self.position = np.array(target_position, dtype=np.float32)

class SphereManager:
    """シーン内の全ての球体を管理するクラス"""
    def __init__(self, config, world_width, world_height, particle_manager):
        self.config = config
        self.world_width = world_width
        self.world_height = world_height
        self.particle_manager = particle_manager
        self.spheres = []
        self.hand_indicators = []
        self.touch_count = 0
        
        self.hover_distance = self.config.get('interaction', 'hover_distance', default=1.0)
        self.chain_reaction_radius = self.config.get('interaction', 'chain_reaction_radius', default=2.5)
        self.chain_reaction_delay = self.config.get('interaction', 'chain_reaction_delay', default=0.05)
        self.chain_reaction_max_depth = self.config.get('interaction', 'chain_reaction_max_depth', default=3)
        self.chain_reaction_queue = deque()

        self._create_spheres()
        self._create_hand_indicators()

    def _hsv_to_rgb(self, h, s, v):
        h_i = int(h * 6.0); f = h * 6.0 - h_i
        p, q, t = v * (1.0 - s), v * (1.0 - f * s), v * (1.0 - (1.0 - f) * s)
        if h_i == 0: r, g, b = v, t, p
        elif h_i == 1: r, g, b = q, v, p
        elif h_i == 2: r, g, b = p, v, t
        elif h_i == 3: r, g, b = p, q, v
        elif h_i == 4: r, g, b = t, p, v
        else: r, g, b = v, p, q
        return np.array([r, g, b], dtype=np.float32)

    def _create_spheres(self):
        count = self.config.get('art', 'sphere_count')
        min_size = self.config.get('art', 'sphere_min_size')
        max_size = self.config.get('art', 'sphere_max_size')
        world_depth = self.config.get('hand_tracking', 'world_depth')
        
        half_w = self.world_width / 2.0
        half_h = self.world_height / 2.0
        half_d = world_depth / 2.0
        
        for _ in range(count):
            x = random.uniform(-half_w, half_w)
            y = random.uniform(-half_h, half_h)
            z = random.uniform(-half_d, half_d)
            pos = [x, y, z]
            size = random.uniform(min_size, max_size)
            # テクスチャの色味を活かすため、カラーは白に近づける
            color = self._hsv_to_rgb(random.random(), 0.1, 1.0)
            self.spheres.append(DisappearingSphere(pos, size, color, self.config))
        print(f"{len(self.spheres)}個の球体を3D空間内にランダム配置しました。")

    def _create_hand_indicators(self):
        indicator_size = self.config.get('hand_tracking', 'hand_indicator_size', default=0.08)
        indicator_count = self.config.get('hand_tracking', 'hand_indicator_count', default=10)
        for _ in range(indicator_count):
            indicator = HandIndicatorSphere([0,0,0], indicator_size)
            self.hand_indicators.append(indicator)
        print(f"{len(self.hand_indicators)}個の指先インジケーターを生成しました (サイズ: {indicator_size})。")

    def _trigger_chain_reaction(self, origin_sphere, depth):
        if depth > self.chain_reaction_max_depth:
            return
        
        origin_sphere.is_chain_reacted = True
        
        for sphere in self.spheres:
            if sphere.state in ['visible', 'hover'] and not sphere.is_chain_reacted:
                dist = np.linalg.norm(sphere.position - origin_sphere.position)
                if dist < self.chain_reaction_radius:
                    trigger_time = time.time() + self.chain_reaction_delay * depth
                    self.chain_reaction_queue.append((sphere, trigger_time, depth + 1))
                    sphere.is_chain_reacted = True

    def update(self, dt, fingertip_positions):
        current_time = time.time()
        
        while self.chain_reaction_queue and self.chain_reaction_queue[0][1] <= current_time:
            sphere, _, depth = self.chain_reaction_queue.popleft()
            if sphere.state in ['visible', 'hover']:
                sphere.state = 'disappearing'
                self.particle_manager.emit(sphere.position, sphere.color)
                self._trigger_chain_reaction(sphere, depth)

        for sphere in self.spheres:
            sphere.update(dt, current_time)
            
            if sphere.state in ['visible', 'hover']:
                is_hovering = False
                if fingertip_positions:
                    for tip_pos in fingertip_positions:
                        dist = np.linalg.norm(tip_pos - sphere.position)
                        if dist < sphere.collision_radius:
                            sphere.state = 'disappearing'
                            self.touch_count += 1
                            if self.particle_manager:
                                self.particle_manager.emit(sphere.position, sphere.color)
                            self._trigger_chain_reaction(sphere, 1)
                            is_hovering = False
                            break
                        elif dist < self.hover_distance:
                            is_hovering = True
                
                if sphere.state != 'disappearing':
                    sphere.state = 'hover' if is_hovering else 'visible'

    def update_indicators(self, fingertip_positions):
        num_tips = len(fingertip_positions)
        for i, indicator in enumerate(self.hand_indicators):
            target_pos = fingertip_positions[i] if i < num_tips else None
            indicator.update(0, 0, target_position=target_pos)

    def get_all_visible_objects(self):
        return [s for s in self.spheres if s.alpha > 0.01] + [s for s in self.hand_indicators if s.visible]

    def reset(self):
        for sphere in self.spheres:
            sphere.state = 'appearing'
            sphere.alpha = 0.0
            sphere.is_chain_reacted = False
        self.touch_count = 0
        self.chain_reaction_queue.clear()
        print("全ての球体をリセットしました。")

# =============================================================================
# 3. ユーティリティとヘルパークラス
# =============================================================================
class Camera:
    """3Dカメラと行列計算を管理するクラス"""
    def __init__(self, config, screen_width, screen_height, world_width, world_height):
        self.config = config
        self.world_width = world_width
        self.world_height = world_height
        self.eye = np.array(self.config.get('camera', 'eye'), dtype=np.float32)
        self.target = np.array(self.config.get('camera', 'target'), dtype=np.float32)
        self.up = np.array(self.config.get('camera', 'up'), dtype=np.float32)
        self.near = self.config.get('camera', 'near_plane')
        self.far = self.config.get('camera', 'far_plane')
        self.update_aspect_ratio(screen_width, screen_height)

    def update_aspect_ratio(self, screen_width, screen_height):
        self.aspect_ratio = screen_width / screen_height

    def get_mvp_matrix(self):
        view = self._look_at(self.eye, self.target, self.up)
        projection_type = self.config.get('camera', 'projection_type', default='perspective')
        if projection_type == 'orthographic':
            proj = self._orthographic(-self.world_width / 2.0, self.world_width / 2.0,
                                      -self.world_height / 2.0, self.world_height / 2.0,
                                      self.near, self.far)
        else:
            fovy = self.config.get('camera', 'fovy')
            proj = self._perspective(fovy, self.aspect_ratio, self.near, self.far)
        return (proj @ view).astype('f4')

    def _look_at(self, eye, target, up):
        f = target - eye
        f_norm = np.linalg.norm(f)
        if f_norm == 0: return np.eye(4, dtype='f4')
        f /= f_norm
        s = np.cross(f, up)
        s_norm = np.linalg.norm(s)
        if s_norm == 0: return np.eye(4, dtype='f4')
        s /= s_norm
        u = np.cross(s, f)
        m = np.eye(4, dtype='f4')
        m[0, :3], m[1, :3], m[2, :3] = s, u, -f
        m[:3, 3] = -s @ eye, -u @ eye, f @ eye
        return m

    def _perspective(self, fovy, aspect, near, far):
        f = 1.0 / np.tan(np.radians(fovy) / 2.0)
        m = np.zeros((4, 4), dtype='f4')
        m[0, 0] = f / aspect
        m[1, 1] = f
        m[2, 2] = (far + near) / (near - far)
        m[3, 2] = -1.0
        m[2, 3] = (2.0 * far * near) / (near - far)
        return m

    def _orthographic(self, left, right, bottom, top, near, far):
        m = np.eye(4, dtype='f4')
        m[0, 0] = 2 / (right - left)
        m[1, 1] = 2 / (top - bottom)
        m[2, 2] = -2 / (far - near)
        m[0, 3] = -(right + left) / (right - left)
        m[1, 3] = -(top + bottom) / (top - bottom)
        m[2, 3] = -(far + near) / (far - near)
        return m

class DebugInfo:
    """デバッグ情報とUIを画面に表示するクラス"""
    def __init__(self, config):
        self.config = config
        self.font = cv2.FONT_HERSHEY_SIMPLEX
        self.last_update = 0
        self.update_interval = 0.25
        self.cached_texts = {}
        self.show_debug = True
        self.show_depth_indicator = self.config.get('ui', 'show_depth_indicator', default=True)

    def _draw_text_with_bg(self, frame, text, pos, font_scale, text_color, thickness=1):
        x, y = pos
        (text_w, text_h), baseline = cv2.getTextSize(text, self.font, font_scale, thickness)
        padding = 5
        rect_start = (x - padding, y)
        rect_end = (x + text_w + padding, y + text_h + baseline + padding)
        x1, y1 = max(0, rect_start[0]), max(0, rect_start[1])
        x2, y2 = min(frame.shape[1], rect_end[0]), min(frame.shape[0], rect_end[1])
        if x1 >= x2 or y1 >= y2: return 0
        sub_frame = frame[y1:y2, x1:x2]
        black_rect = np.full(sub_frame.shape, (0, 0, 0), dtype=np.uint8)
        res = cv2.addWeighted(sub_frame, 0.5, black_rect, 0.5, 1.0)
        frame[y1:y2, x1:x2] = res
        cv2.putText(frame, text, (x, y + text_h + int(baseline/2)), self.font, font_scale, text_color, thickness, cv2.LINE_AA)
        return text_h + baseline + padding * 2

    def _draw_depth_indicator(self, frame, tracker):
        if not self.show_depth_indicator or not tracker.is_any_hand_detected():
            return
        h, w, _ = frame.shape
        bar_w, bar_h = 15, h // 2
        bar_x, bar_y = w - bar_w - 20, h // 4
        world_depth = self.config.get('hand_tracking', 'world_depth')
        z_offset = self.config.get('hand_tracking', 'z_offset')
        z_min, z_max = z_offset, z_offset - world_depth
        cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h), (255, 255, 255), 1)
        avg_z = tracker.get_average_fingertip_z()
        if avg_z is not None:
            normalized_z = np.clip((avg_z - z_min) / (z_max - z_min), 0, 1)
            marker_y = int(bar_y + normalized_z * bar_h)
            cv2.line(frame, (bar_x - 5, marker_y), (bar_x + bar_w + 5, marker_y), (0, 255, 255), 2)
            cv2.putText(frame, f"{avg_z:.1f}", (bar_x - 50, marker_y + 5), self.font, 0.5, (0, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(frame, "Depth", (bar_x - 15, bar_y - 10), self.font, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

    def update_and_draw(self, frame, fps, sphere_manager, tracker):
        help_text = "F11=Fullscreen | D=Debug | T=Reset"
        self._draw_depth_indicator(frame, tracker)
        if not self.show_debug:
            cv2.putText(frame, help_text, (10, frame.shape[0] - 15), self.font, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
            return frame
        current_time = time.time()
        if current_time - self.last_update >= self.update_interval:
            self.cached_texts['fps'] = f"FPS: {fps:.1f}"
            visible = len([s for s in sphere_manager.spheres if s.state in ['visible', 'hover']])
            self.cached_texts['spheres'] = f"Stars: {visible}/{len(sphere_manager.spheres)} | Touched: {sphere_manager.touch_count}"
            self.cached_texts['face_dir'] = f"Face: {tracker.get_face_direction()}"
            left_status = "OK" if tracker.left_hand_detected else "---"
            right_status = "OK" if tracker.right_hand_detected else "---"
            self.cached_texts['hands'] = f"Hands: L[{left_status}] R[{right_status}]"
            self.last_update = current_time
        y, padding = 10, 10
        info_lines = [
            (self.cached_texts.get('fps'), 0.7, (0, 255, 128), 2),
            (self.cached_texts.get('spheres'), 0.6, (255, 255, 180), 1),
            (self.cached_texts.get('face_dir'), 0.6, (180, 255, 255), 1),
            (self.cached_texts.get('hands'), 0.6, (255, 180, 255), 1)
        ]
        for text, scale, color, thickness in info_lines:
            if text: y += self._draw_text_with_bg(frame, text, (padding, y), scale, color, thickness)
        cv2.putText(frame, help_text, (10, frame.shape[0] - 15), self.font, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
        return frame

    def toggle(self):
        self.show_debug = not self.show_debug
        print(f"デバッグ情報表示: {'ON' if self.show_debug else 'OFF'}")

class FaceDirectionDetector:
    """顔のランドマークから向きを推定するクラス"""
    def __init__(self, config, frame_shape):
        self.config = config
        self.frame_shape = frame_shape
        self.direction_text = "Center"
        self.model_points = np.array([
            (0.0, 0.0, 0.0), (0.0, -330.0, -65.0), (-225.0, 170.0, -135.0),
            (225.0, 170.0, -135.0), (-150.0, -150.0, -125.0), (150.0, -150.0, -125.0)
        ])
        focal_length = self.frame_shape[1]
        center = (self.frame_shape[1]/2, self.frame_shape[0]/2)
        self.camera_matrix = np.array(
            [[focal_length, 0, center[0]], [0, focal_length, center[1]], [0, 0, 1]], dtype="double"
        )
        self.dist_coeffs = np.zeros((4,1))

    def update(self, face_landmarks):
        if not face_landmarks:
            self.direction_text = "N/A"
            return
        lm_indices = [1, 152, 33, 263, 61, 291]
        image_points = np.array([
            (face_landmarks.landmark[i].x * self.frame_shape[1], face_landmarks.landmark[i].y * self.frame_shape[0]) for i in lm_indices
        ], dtype="double")
        try:
            (_, rotation_vector, _) = cv2.solvePnP(
                self.model_points, image_points, self.camera_matrix, self.dist_coeffs, flags=cv2.SOLVEPNP_ITERATIVE
            )
            rmat, _ = cv2.Rodrigues(rotation_vector)
            angles, _, _, _, _, _ = cv2.RQDecomp3x3(rmat)
            yaw, pitch = angles[1], angles[0]
            yaw_threshold = self.config.get('face_detection', 'yaw_threshold')
            pitch_threshold = self.config.get('face_detection', 'pitch_threshold')
            if yaw < -yaw_threshold: self.direction_text = "Looking Right"
            elif yaw > yaw_threshold: self.direction_text = "Looking Left"
            elif pitch < -pitch_threshold: self.direction_text = "Looking Down"
            elif pitch > pitch_threshold: self.direction_text = "Looking Up"
            else: self.direction_text = "Center"
        except cv2.error:
            self.direction_text = "Calculating..."

    def get_direction(self):
        return self.direction_text

class BodyTracker:
    """手と顔の検出と座標変換を管理するクラス"""
    def __init__(self, config, world_width, world_height, frame_shape):
        self.config = config
        self.world_width = world_width
        self.world_height = world_height
        self.z_scale = self.config.get('hand_tracking', 'z_scale', default=30.0)
        self.z_offset = self.config.get('hand_tracking', 'z_offset', default=-5.0)
        self.fingertip_positions = []
        self.left_hand_detected = False
        self.right_hand_detected = False
        self.face_detector = FaceDirectionDetector(config, frame_shape)
        self.fingertip_indices = self.config.get('mediapipe', 'fingertip_indices', default=[4, 8, 12, 16, 20])
        print(f"3Dワールド空間のサイズを初期化: Width={self.world_width:.2f}, Height={self.world_height:.2f}")

    def update_from_landmarks(self, left_hand, right_hand, face_landmarks):
        self.left_hand_detected = left_hand is not None
        self.right_hand_detected = right_hand is not None
        self.fingertip_positions.clear()
        if self.left_hand_detected:
            for i in self.fingertip_indices:
                tip_pos = self._landmark_to_world(left_hand.landmark[i])
                if tip_pos is not None: self.fingertip_positions.append(tip_pos)
        if self.right_hand_detected:
            for i in self.fingertip_indices:
                tip_pos = self._landmark_to_world(right_hand.landmark[i])
                if tip_pos is not None: self.fingertip_positions.append(tip_pos)
        self.face_detector.update(face_landmarks)

    def _landmark_to_world(self, landmark):
        if not landmark: return None
        # ★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★
        # ★ 修正箇所：手の向きの修正
        # ★ カメラ映像(frame)は水平反転(flip)されている。
        # ★ そのため、landmark.xをそのまま使うことで、鏡像（ミラー）の動きになる。
        # ★ (0.5 - landmark.x) にすると、反転の反転となり、意図しない動きになる。
        # ★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★
        x = (landmark.x - 0.5) * self.world_width
        y = (0.5 - landmark.y) * self.world_height # Yは下がプラスなので反転
        z = (landmark.z * self.z_scale) + self.z_offset
        return np.array([x, y, z], dtype=np.float32)

    def get_current_fingertip_positions(self):
        return self.fingertip_positions
        
    def get_face_direction(self):
        return self.face_detector.get_direction()

    def is_any_hand_detected(self):
        return self.left_hand_detected or self.right_hand_detected

    def get_average_fingertip_z(self):
        if not self.fingertip_positions:
            return None
        return np.mean([pos[2] for pos in self.fingertip_positions])

class Starfield:
    """流れる星空を管理・描画するクラス"""
    def __init__(self, config, ctx, camera):
        self.config = config
        self.ctx = ctx
        self.camera = camera
        self.star_count = self.config.get('starfield', 'star_count', default=1000)
        self.speed = self.config.get('starfield', 'star_speed', default=0.5)
        self.max_size = self.config.get('starfield', 'star_max_size', default=2.0)
        self.ww = self.camera.world_width * 1.5
        self.wh = self.camera.world_height * 1.5
        self.depth_range = (self.camera.near, self.camera.far)
        self.stars = np.zeros((self.star_count, 4), dtype='f4')
        self.stars[:, 0] = np.random.uniform(-self.ww, self.ww, self.star_count)
        self.stars[:, 1] = np.random.uniform(-self.wh, self.wh, self.star_count)
        self.stars[:, 2] = np.random.uniform(self.depth_range[0], self.depth_range[1], self.star_count)
        self.stars[:, 3] = np.random.uniform(0.2, 1.0, self.star_count)
        self.vbo = self.ctx.buffer(self.stars.tobytes())
        self.program = None
        self.vao = None
        print(f"{self.star_count}個の星を背景に生成しました。")

    def set_program(self, program):
        self.program = program
        self.vao = self.ctx.vertex_array(
            self.program,
            [(self.vbo, '3f 1f', 'in_vert', 'in_brightness')]
        )

    def update(self, dt):
        self.stars[:, 2] += self.speed * dt
        reset_indices = self.stars[:, 2] > self.camera.far
        if np.any(reset_indices):
            self.stars[reset_indices, 2] = self.camera.near
        self.vbo.write(self.stars.tobytes())

    def render(self, mvp):
        if self.vao and self.program:
            self.program['mvp'].write(mvp)
            self.program['max_size'].value = self.max_size
            self.ctx.point_size = self.max_size
            self.vao.render(mode=moderngl.POINTS)

class ParticleManager:
    """パーティクルを管理・描画するクラス"""
    def __init__(self, config, ctx):
        self.config = config
        self.ctx = ctx
        self.particle_count = self.config.get('particles', 'count', default=10000)
        self.lifespan = self.config.get('particles', 'lifespan', default=1.5)
        self.speed = self.config.get('particles', 'speed', default=4.0)
        self.gravity = np.array([0, self.config.get('particles', 'gravity', default=-2.0), 0], dtype='f4')
        self.emit_count = self.config.get('particles', 'emit_count', default=50)
        self.data = np.zeros((self.particle_count, 11), dtype='f4')
        self.vbo = self.ctx.buffer(reserve=self.data.nbytes)
        self.program = None
        self.vao = None
        self.next_particle_idx = 0
        print(f"{self.particle_count}個のパーティクルプールを生成しました。")

    def set_program(self, program):
        self.program = program
        self.vao = self.ctx.vertex_array(
            self.program,
            [(self.vbo, '3f 12x 1f 3f 4x', 'in_pos', 'in_life', 'in_color')]
        )

    def emit(self, position, color):
        start = self.next_particle_idx
        end = start + self.emit_count
        indices = np.arange(start, end) % self.particle_count
        self.next_particle_idx = end % self.particle_count
        velocities = np.random.normal(0.0, 1.0, (self.emit_count, 3)).astype('f4')
        norms = np.linalg.norm(velocities, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        velocities = velocities / norms * self.speed
        self.data[indices, 0:3] = position
        self.data[indices, 3:6] = velocities
        self.data[indices, 6] = self.lifespan
        self.data[indices, 7:10] = color

    def update(self, dt):
        active_mask = self.data[:, 6] > 0
        if not np.any(active_mask): return
        active_data = self.data[active_mask]
        active_data[:, 3:6] += self.gravity * dt
        active_data[:, 0:3] += active_data[:, 3:6] * dt
        active_data[:, 6] -= dt
        self.data[active_mask] = active_data
        self.vbo.write(self.data.tobytes())

    def render(self, mvp):
        if self.vao and self.program:
            self.program['mvp'].write(mvp)
            self.vao.render(mode=moderngl.POINTS, vertices=self.particle_count)

# =============================================================================
# 4. メインアプリケーションクラス
# =============================================================================
class App:
    def __init__(self, config):
        self.config = config
        self.running = True
        self.fullscreen = self.config.get('display', 'fullscreen', default=False)
        
        self._init_camera_capture()
        self._init_pygame()
        
        world_width, world_height = self._calculate_world_size()
        
        self._init_moderngl()
        
        self.particle_manager = ParticleManager(config, self.ctx)
        self.particle_manager.set_program(self.particle_program)

        self.sphere_manager = SphereManager(config, world_width, world_height, self.particle_manager)
        self.tracker = BodyTracker(config, world_width, world_height, (self.camera_height, self.camera_width))
        self.camera = Camera(config, self.screen_size[0], self.screen_size[1], world_width, world_height)
        
        self.starfield = Starfield(config, self.ctx, self.camera)
        self.starfield.set_program(self.star_program)

        self.debug_info = DebugInfo(config)
        
        self.light_direction = np.array([0.5, 1.0, 0.8], dtype='f4')
        self.light_smoothing = self.config.get('lighting', 'smoothing', default=0.05)

        self.landmarks_queue = queue.Queue(maxsize=2)
        self.video_queue = queue.Queue(maxsize=1)
        self.mediapipe_thread = threading.Thread(target=self._mediapipe_worker, daemon=True)
        
        self.clock = pygame.time.Clock()
        self.last_time = time.time()
        self.fps_history = deque(maxlen=15)

    def _calculate_world_size(self):
        screen_aspect_ratio = self.screen_size[0] / self.screen_size[1]
        projection_type = self.config.get('camera', 'projection_type')
        if projection_type == 'orthographic':
            world_height = self.config.get('camera', 'orthographic_height', default=18.0)
            world_width = world_height * screen_aspect_ratio
        else:
            fovy_rad = math.radians(self.config.get('camera', 'fovy'))
            cam_eye = self.config.get('camera', 'eye')
            cam_target = self.config.get('camera', 'target')
            distance = abs(cam_eye[2] - cam_target[2])
            world_height = 2 * distance * math.tan(fovy_rad / 2)
            world_width = world_height * screen_aspect_ratio
        return world_width, world_height

    def _init_camera_capture(self):
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened(): raise RuntimeError("カメラが見つかりません。")
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        self.camera_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.camera_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        print(f"カメラ解像度: {self.camera_width}x{self.camera_height}")

    def _init_pygame(self):
        pygame.init()
        display_index = self.config.get('display', 'display_index', default=0)
        try:
            displays = pygame.display.get_desktop_sizes()
            if display_index >= len(displays):
                print(f"[警告] ディスプレイ {display_index} が見つかりません。プライマリを使用します。")
                display_index = 0
        except pygame.error:
            print("[警告] ディスプレイサイズの取得に失敗しました。デフォルト設定を使用します。")
            displays = [(self.config.get('display', 'default_width'), self.config.get('display', 'default_height'))]
            display_index = 0
        if self.fullscreen:
            self.screen_size = displays[display_index]
            flags = pygame.OPENGL | pygame.DOUBLEBUF | pygame.FULLSCREEN
        else:
            self.screen_size = (self.config.get('display', 'default_width'), self.config.get('display', 'default_height'))
            flags = pygame.OPENGL | pygame.DOUBLEBUF
        pygame.display.set_mode(self.screen_size, flags, display=display_index)
        pygame.display.set_caption("Interactive Cosmic Art")
        print(f"Pygameウィンドウサイズ: {self.screen_size[0]}x{self.screen_size[1]}")

    def _init_moderngl(self):
        self.ctx = moderngl.create_context()
        self.ctx.enable(moderngl.DEPTH_TEST | moderngl.BLEND | moderngl.CULL_FACE | moderngl.PROGRAM_POINT_SIZE)
        self.ctx.blend_func = moderngl.SRC_ALPHA, moderngl.ONE_MINUS_SRC_ALPHA
        self.ctx.viewport = (0, 0, *self.screen_size)
        self._load_textures()
        self._create_shaders()
        self._init_buffers()

    def _load_shader(self, path):
        try:
            with open(path, 'r', encoding='utf-8') as f:
                return f.read()
        except FileNotFoundError:
            print(f"[エラー] シェーダーファイルが見つかりません: {path}")
            sys.exit(1)

    def _load_textures(self):
        path = self.config.get('art', 'texture_path')
        try:
            img = Image.open(path).convert('RGBA')
            self.sphere_texture = self.ctx.texture(img.size, 4, img.tobytes())
            self.sphere_texture.build_mipmaps()
            print(f"テクスチャを読み込みました: {path}")
        except FileNotFoundError:
            print(f"[エラー] テクスチャファイルが見つかりません: {path}")
            self.sphere_texture = None # フォールバック

    def _create_shaders(self):
        shader_dir = "shaders"
        self.sphere_program = self.ctx.program(
            vertex_shader=self._load_shader(os.path.join(shader_dir, 'sphere.vert')),
            fragment_shader=self._load_shader(os.path.join(shader_dir, 'sphere.frag'))
        )
        self.star_program = self.ctx.program(
            vertex_shader=self._load_shader(os.path.join(shader_dir, 'star.vert')),
            fragment_shader=self._load_shader(os.path.join(shader_dir, 'star.frag'))
        )
        self.particle_program = self.ctx.program(
            vertex_shader=self._load_shader(os.path.join(shader_dir, 'particle.vert')),
            fragment_shader=self._load_shader(os.path.join(shader_dir, 'particle.frag'))
        )
        if self.sphere_texture and 'u_texture' in self.sphere_program:
            self.sphere_program['u_texture'].value = 0 # テクスチャユニット0を使用

    def _init_buffers(self):
        resolution = self.config.get('art', 'sphere_resolution')
        vertices, indices = self._generate_sphere_mesh(1.0, resolution)
        self.sphere_vbo = self.ctx.buffer(vertices.tobytes())
        self.sphere_ibo = self.ctx.buffer(indices.tobytes())
        max_instances = self.config.get('art', 'sphere_count') + self.config.get('hand_tracking', 'hand_indicator_count')
        self.instance_vbo = self.ctx.buffer(reserve=max_instances * 11 * 4)
        self.sphere_vao = self.ctx.vertex_array(
            self.sphere_program,
            [
                (self.sphere_vbo, '3f 2f', 'position', 'in_uv'),
                (self.instance_vbo, '3f 3f 1f 3f 1f /i', 'sphere_pos', 'object_color', 'alpha', 'rotation', 'radius')
            ],
            self.sphere_ibo
        )

    def _generate_sphere_mesh(self, radius, resolution):
        verts, inds = [], []
        for i in range(resolution + 1):
            lat = np.pi * (-0.5 + i / resolution)
            u = i / resolution
            for j in range(resolution + 1):
                lon = 2 * np.pi * j / resolution
                v = j / resolution
                x = radius * np.cos(lat) * np.cos(lon)
                y = radius * np.cos(lat) * np.sin(lon)
                z = radius * np.sin(lat)
                verts.extend([x, y, z, v, u])
        for i in range(resolution):
            for j in range(resolution):
                v1 = i * (resolution + 1) + j
                v2 = v1 + resolution + 1
                inds.extend([v1, v2, v1 + 1, v2, v2 + 1, v1 + 1])
        return np.array(verts, dtype='f4'), np.array(inds, dtype='i4')

    def _mediapipe_worker(self):
        mp_holistic = mp.solutions.holistic
        holistic = mp_holistic.Holistic(
            model_complexity=self.config.get('mediapipe', 'model_complexity'),
            min_detection_confidence=self.config.get('mediapipe', 'min_detection_confidence'),
            min_tracking_confidence=self.config.get('mediapipe', 'min_tracking_confidence'),
            refine_face_landmarks=True
        )
        while self.running:
            ret, frame = self.cap.read()
            if not ret: continue
            frame = cv2.flip(frame, 1)
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            rgb_frame.flags.writeable = False
            results = holistic.process(rgb_frame)
            rgb_frame.flags.writeable = True
            left_hand = results.right_hand_landmarks
            right_hand = results.left_hand_landmarks
            try:
                self.landmarks_queue.put_nowait({
                    'left': left_hand, 'right': right_hand, 'face': results.face_landmarks
                })
                fps = self.fps_history[-1] if self.fps_history else 0
                debug_frame = self.debug_info.update_and_draw(frame, fps, self.sphere_manager, self.tracker)
                self.video_queue.put_nowait(debug_frame)
            except queue.Full:
                try:
                    self.landmarks_queue.get_nowait()
                    self.video_queue.get_nowait()
                except queue.Empty:
                    pass
        holistic.close()

    def run(self):
        self.mediapipe_thread.start()
        cv2.namedWindow('Motion Capture View', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('Motion Capture View', self.camera_width // 2, self.camera_height // 2)
        latest_landmarks = None
        while self.running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT or (event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE):
                    self.running = False
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_F11: self._toggle_fullscreen()
                    if event.key == pygame.K_d: self.debug_info.toggle()
                    if event.key == pygame.K_t: self.sphere_manager.reset()
            try:
                cv2.imshow('Motion Capture View', self.video_queue.get_nowait())
            except queue.Empty:
                pass
            if cv2.waitKey(1) & 0xFF == 27:
                self.running = False
            current_time = time.time()
            dt = current_time - self.last_time
            self.last_time = current_time
            if dt > 0: self.fps_history.append(1.0 / dt)
            try:
                latest_landmarks = self.landmarks_queue.get_nowait()
            except queue.Empty:
                pass
            if latest_landmarks:
                self.tracker.update_from_landmarks(
                    latest_landmarks['left'], latest_landmarks['right'], latest_landmarks['face']
                )
            self._update_lighting(self.tracker.get_face_direction())
            fingertip_positions = self.tracker.get_current_fingertip_positions()
            self.sphere_manager.update(dt, fingertip_positions)
            self.sphere_manager.update_indicators(fingertip_positions)
            self.starfield.update(dt)
            self.particle_manager.update(dt)
            self.ctx.clear(0.01, 0.01, 0.03, 1.0)
            mvp = self.camera.get_mvp_matrix()
            self.ctx.disable(moderngl.DEPTH_TEST)
            self.starfield.render(mvp)
            self.particle_manager.render(mvp)
            self.ctx.enable(moderngl.DEPTH_TEST)
            self._render_instanced(mvp)
            pygame.display.flip()
            self.clock.tick(60)
        self.cleanup()

    def _update_lighting(self, direction):
        target_light_dir = np.array([0.5, 1.0, 0.8], dtype='f4')
        if direction == "Looking Left": target_light_dir = np.array([-1.0, 0.5, 0.8], dtype='f4')
        elif direction == "Looking Right": target_light_dir = np.array([1.0, 0.5, 0.8], dtype='f4')
        elif direction == "Looking Up": target_light_dir = np.array([0.0, 1.0, 0.8], dtype='f4')
        elif direction == "Looking Down": target_light_dir = np.array([0.0, -1.0, 0.8], dtype='f4')
        self.light_direction += (target_light_dir - self.light_direction) * self.light_smoothing

    def _render_instanced(self, mvp):
        all_objs = self.sphere_manager.get_all_visible_objects()
        if not all_objs: return
        inst_data = np.array([(*s.position, *s.color, s.alpha, *s.rotation, s.current_radius) for s in all_objs], dtype='f4')
        self.instance_vbo.write(inst_data)
        if self.sphere_texture:
            self.sphere_texture.use(location=0)
        self.sphere_program['mvp'].write(mvp)
        self.sphere_program['u_light_direction'].write(self.light_direction.tobytes())
        self.sphere_vao.render(instances=len(all_objs))

    def _toggle_fullscreen(self):
        self.fullscreen = not self.fullscreen
        pygame.display.quit()
        self._init_pygame()
        self.ctx.viewport = (0, 0, *self.screen_size)
        self.camera.update_aspect_ratio(*self.screen_size)
        print(f"ディスプレイモードを切り替えました: {'フルスクリーン' if self.fullscreen else 'ウィンドウ'}")

    def cleanup(self):
        print("クリーンアップ処理を実行中...")
        self.running = False
        if self.mediapipe_thread.is_alive():
            self.mediapipe_thread.join(timeout=1)
        self.cap.release()
        cv2.destroyAllWindows()
        pygame.quit()
        print("クリーンアップ完了。")

if __name__ == '__main__':
    try:
        app = App(Config('config.json'))
        app.run()
    except Exception as e:
        print(f"アプリケーションの実行中に致命的なエラーが発生しました: {e}")
        import traceback
        traceback.print_exc()
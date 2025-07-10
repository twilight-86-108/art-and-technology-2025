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
        self.color = color if color is not None else np.array([1.0, 1.0, 1.0], dtype=np.float32)
        self.rotation = np.array([0.0, 0.0, 0.0], dtype=np.float32)
        self.rotation_speed = np.array([random.uniform(-0.6, 0.6) for _ in range(3)], dtype=np.float32)
        self.alpha = 1.0
        self.visible = True

    def update(self, dt, current_time):
        self.rotation += self.rotation_speed * dt * 0.15

class DisappearingSphere(BaseSphere):
    """触れると消える球体のクラス"""
    def __init__(self, position, radius, color=None):
        super().__init__(position, radius, color)
        self.state = 'visible'
        self.disappear_time = 0.0
        self.respawn_delay = random.uniform(2.0, 4.0)
        self.fade_speed = 4.0
        self.collision_radius = radius * 6.0
        self.float_offset = random.uniform(0, 2 * math.pi)

    def update(self, dt, current_time):
        super().update(dt, current_time)
        float_y = math.sin(current_time * 0.5 + self.float_offset) * 0.01
        self.position[1] = self.original_position[1] + float_y

        if self.state == 'disappearing':
            self.alpha -= self.fade_speed * dt
            if self.alpha <= 0.0:
                self.alpha, self.state, self.disappear_time = 0.0, 'invisible', current_time
        elif self.state == 'invisible':
            if current_time - self.disappear_time >= self.respawn_delay:
                self.state = 'appearing'
        elif self.state == 'appearing':
            self.alpha += self.fade_speed * dt
            if self.alpha >= 1.0:
                self.alpha, self.state = 1.0, 'visible'
        else:
            self.alpha = 0.9 + 0.1 * math.sin(current_time * 1.0)

class HandIndicatorSphere(BaseSphere):
    """手の位置を示すインジケーター球体"""
    def __init__(self, position, radius, color=None):
        super().__init__(position, radius, color)
        self.alpha = 0.8
        self.visible = False

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
            color = self._hsv_to_rgb(random.random(), 0.8, 1.0)
            self.spheres.append(DisappearingSphere(pos, size, color))
        print(f"{len(self.spheres)}個の球体を3D空間内にランダム配置しました。")

    def _create_hand_indicators(self):
        indicator_color = np.array([1.0, 1.0, 0.0], dtype=np.float32)
        indicator_size = self.config.get('hand_tracking', 'hand_indicator_size', default=0.08)
        for _ in range(10): # 両手x5本の指
            indicator = HandIndicatorSphere([0,0,0], indicator_size, indicator_color)
            self.hand_indicators.append(indicator)
        print(f"{len(self.hand_indicators)}個の指先インジケーターを生成しました (サイズ: {indicator_size})。")

    def update(self, dt, fingertip_positions):
        current_time = time.time()
        for sphere in self.spheres:
            sphere.update(dt, current_time)
        
        if fingertip_positions:
            for tip_pos in fingertip_positions:
                for sphere in self.spheres:
                    if sphere.state == 'visible' and np.linalg.norm(tip_pos - sphere.position) < sphere.collision_radius:
                        sphere.state = 'disappearing'
                        self.touch_count += 1
                        if self.particle_manager:
                            self.particle_manager.emit(sphere.position, sphere.color)
    
    def update_indicators(self, fingertip_positions):
        num_tips = len(fingertip_positions)
        for i, indicator in enumerate(self.hand_indicators):
            if i < num_tips:
                indicator.update(0, 0, target_position=fingertip_positions[i])
            else:
                indicator.update(0, 0, target_position=None)

    def get_all_visible_objects(self):
        return [s for s in self.spheres if s.alpha > 0.01] + [s for s in self.hand_indicators if s.visible]

    def reset(self):
        for sphere in self.spheres:
            sphere.state = 'appearing'
            sphere.alpha = 0.0
        self.touch_count = 0
        print("全ての球体をリセットしました。")

# =============================================================================
# 3. ユーティリティとヘルパークラス
# =============================================================================
class Camera:
    """3Dカメラと行列計算を管理するクラス (移動機能は削除)"""
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
    """デバッグ情報を画面に表示するクラス"""
    def __init__(self):
        self.font = cv2.FONT_HERSHEY_SIMPLEX
        self.last_update = 0
        self.update_interval = 0.25
        self.cached_texts = {}
        self.show_debug = True

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

    def update_and_draw(self, frame, fps, sphere_manager, tracker):
        help_text = "F11=Fullscreen | D=Debug | T=Reset"
        if not self.show_debug:
            cv2.putText(frame, help_text, (10, frame.shape[0] - 15), self.font, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
            return frame
        current_time = time.time()
        if current_time - self.last_update >= self.update_interval:
            self.cached_texts['fps'] = f"FPS: {fps:.1f}"
            visible = len([s for s in sphere_manager.spheres if s.state == 'visible'])
            self.cached_texts['spheres'] = f"Stars: {visible}/{len(sphere_manager.spheres)} | Touched: {sphere_manager.touch_count}"
            self.cached_texts['face_dir'] = f"Face: {tracker.get_face_direction()}"
            left_status = "OK" if tracker.left_hand_detected else "---"
            right_status = "OK" if tracker.right_hand_detected else "---"
            self.cached_texts['hands'] = f"Hands: L[{left_status}] R[{right_status}]"
            self.last_update = current_time
        y = 10; padding = 10
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

        image_points = np.array([
            (face_landmarks.landmark[1].x * self.frame_shape[1], face_landmarks.landmark[1].y * self.frame_shape[0]),
            (face_landmarks.landmark[152].x * self.frame_shape[1], face_landmarks.landmark[152].y * self.frame_shape[0]),
            (face_landmarks.landmark[33].x * self.frame_shape[1], face_landmarks.landmark[33].y * self.frame_shape[0]),
            (face_landmarks.landmark[263].x * self.frame_shape[1], face_landmarks.landmark[263].y * self.frame_shape[0]),
            (face_landmarks.landmark[61].x * self.frame_shape[1], face_landmarks.landmark[61].y * self.frame_shape[0]),
            (face_landmarks.landmark[291].x * self.frame_shape[1], face_landmarks.landmark[291].y * self.frame_shape[0])
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
        self.plane_z = config.get('art', 'plane_z')
        
        self.z_scale = self.config.get('hand_tracking', 'z_scale', default=30.0)
        self.z_offset = self.config.get('hand_tracking', 'z_offset', default=-5.0)

        self.fingertip_positions = []
        self.left_hand_detected = False
        self.right_hand_detected = False
        self.face_detector = FaceDirectionDetector(config, frame_shape)
        self.fingertip_indices = [4, 8, 12, 16, 20]
        print(f"3Dワールド空間のサイズを初期化: Width={self.world_width:.2f}, Height={self.world_height:.2f}")

    def update_from_landmarks(self, left_hand, right_hand, face_landmarks):
        self.left_hand_detected = left_hand is not None
        self.right_hand_detected = right_hand is not None
        self.fingertip_positions = []
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
        x = (landmark.x - 0.5) * self.world_width
        y = (0.5 - landmark.y) * self.world_height
        z = (landmark.z * self.z_scale) + self.z_offset
        return np.array([x, y, z], dtype=np.float32)

    def get_current_fingertip_positions(self):
        return self.fingertip_positions
        
    def get_face_direction(self):
        return self.face_detector.get_direction()

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

        # [pos(3), vel(3), life(1), color(3)]
        self.data = np.zeros((self.particle_count, 10), dtype='f4')
        self.vbo = self.ctx.buffer(reserve=self.data.nbytes)
        
        self.program = None
        self.vao = None
        self.next_particle_idx = 0
        print(f"{self.particle_count}個のパーティクルプールを生成しました。")

    def set_program(self, program):
        self.program = program
        # ★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★
        # ★ 修正箇所：VAOの定義を修正
        # ★ 最適化で消される'in_vel'をVAOの定義から削除し、代わりにパディング(12x)を追加
        # ★ また、パーティクルはインスタンス描画ではないため、'/i'を削除
        # ★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★
        self.vao = self.ctx.vertex_array(
            self.program,
            [(self.vbo, '3f 12x 1f 3f', 'in_pos', 'in_life', 'in_color')]
        )

    def emit(self, position, color):
        start = self.next_particle_idx
        end = start + self.emit_count
        
        if end > self.particle_count:
            remaining = end - self.particle_count
            indices = np.concatenate([np.arange(start, self.particle_count), np.arange(0, remaining)])
            self.next_particle_idx = remaining
        else:
            indices = np.arange(start, end)
            self.next_particle_idx = end

        velocities = np.random.normal(0.0, 1.0, (self.emit_count, 3)).astype('f4')
        norms = np.linalg.norm(velocities, axis=1, keepdims=True)
        # ゼロ除算を避ける
        norms[norms == 0] = 1.0
        velocities = velocities / norms * self.speed
        
        self.data[indices, 0:3] = position
        self.data[indices, 3:6] = velocities
        self.data[indices, 6] = self.lifespan
        self.data[indices, 7:10] = color

    def update(self, dt):
        active_mask = self.data[:, 6] > 0
        if not np.any(active_mask):
            return

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
        
        self._init_moderngl()
        
        self.particle_manager = ParticleManager(config, self.ctx)
        self.particle_manager.set_program(self.particle_program)

        self.sphere_manager = SphereManager(config, world_width, world_height, self.particle_manager)
        self.tracker = BodyTracker(config, world_width, world_height, (self.camera_height, self.camera_width))
        self.camera = Camera(config, self.screen_size[0], self.screen_size[1], world_width, world_height)
        
        self.starfield = Starfield(config, self.ctx, self.camera)
        self.starfield.set_program(self.star_program)

        self.debug_info = DebugInfo()
        
        self.light_direction = np.array([0.5, 1.0, 0.8], dtype='f4')
        self.light_smoothing = self.config.get('lighting', 'smoothing', default=0.05)

        self.landmarks_queue = queue.Queue(maxsize=2)
        self.video_queue = queue.Queue(maxsize=1)
        self.mediapipe_thread = threading.Thread(target=self._mediapipe_worker, daemon=True)
        
        self.clock = pygame.time.Clock()
        self.last_time = time.time()
        self.fps_history = deque(maxlen=15)
        
    def _init_camera_capture(self):
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened(): raise RuntimeError("カメラが見つかりません。")
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280); self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        self.camera_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.camera_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        print(f"カメラ解像度: {self.camera_width}x{self.camera_height}")

    def _init_pygame(self):
        pygame.init()
        display_index = self.config.get('display', 'display_index', default=0)
        if self.fullscreen:
            try: self.screen_size = pygame.display.get_desktop_sizes()[display_index]
            except IndexError:
                print(f"[警告] ディスプレイ {display_index} が見つかりません。プライマリを使用します。"); display_index = 0
                self.screen_size = pygame.display.get_desktop_sizes()[display_index]
            flags = pygame.OPENGL | pygame.DOUBLEBUF | pygame.FULLSCREEN
        else:
            self.screen_size = (self.config.get('display', 'default_width'), self.config.get('display', 'default_height'))
            flags = pygame.OPENGL | pygame.DOUBLEBUF
        pygame.display.set_mode(self.screen_size, flags, display=display_index)
        pygame.display.set_caption("Interactive Cosmic Art")
        print(f"Pygameウィンドウサイズ: {self.screen_size[0]}x{self.screen_size[1]}")

    def _init_moderngl(self):
        self.ctx = moderngl.create_context(); self.ctx.enable(moderngl.DEPTH_TEST | moderngl.BLEND | moderngl.CULL_FACE | moderngl.PROGRAM_POINT_SIZE)
        self.ctx.blend_func = moderngl.SRC_ALPHA, moderngl.ONE_MINUS_SRC_ALPHA
        self.ctx.viewport = (0, 0, *self.screen_size); self._create_shaders(); self._init_buffers()

    def _create_shaders(self):
        # 球体用シェーダー
        vertex_shader = """
        #version 330
        in vec3 position; in vec3 sphere_pos; in vec3 object_color; in float alpha; in vec3 rotation; in float radius;
        uniform mat4 mvp;
        out vec3 v_normal; out vec3 v_world_pos; out vec3 v_color; out float v_alpha;
        mat3 rotationMatrix(vec3 angles){
            vec3 c = cos(angles); vec3 s = sin(angles);
            return mat3(c.y*c.z, -c.y*s.z, s.y, c.x*s.z+s.x*s.y*c.z, c.x*c.z-s.x*s.y*s.z, -s.x*c.y, s.x*s.z-c.x*s.y*c.z, s.x*c.z+c.x*s.y*s.z, c.x*c.y);
        }
        void main() {
            mat3 rot_m = rotationMatrix(rotation);
            vec3 scaled_pos = position * radius;
            vec3 w_pos = (rot_m * scaled_pos) + sphere_pos;
            gl_Position = mvp * vec4(w_pos, 1.0);
            v_normal = normalize(rot_m * position);
            v_world_pos = w_pos;
            v_color = object_color;
            v_alpha = alpha;
        }"""
        fragment_shader = """
        #version 330
        in vec3 v_normal; in vec3 v_world_pos; in vec3 v_color; in float v_alpha;
        uniform vec3 u_light_direction;
        out vec4 fragColor;
        void main() {
            vec3 light_dir = normalize(u_light_direction);
            float ambient = 0.3; float diffuse = max(0.0, dot(v_normal, light_dir)) * 0.7;
            vec3 view_dir = normalize(-v_world_pos);
            vec3 reflect_dir = reflect(-light_dir, v_normal);
            float spec = pow(max(dot(view_dir, reflect_dir), 0.0), 16.0) * 0.2;
            vec3 final_color = v_color * (ambient + diffuse) + vec3(spec);
            fragColor = vec4(final_color, v_alpha);
        }"""
        self.sphere_program = self.ctx.program(vertex_shader=vertex_shader, fragment_shader=fragment_shader)

        # 星空用のシェーダー
        star_vertex_shader = """
        #version 330
        in vec3 in_vert;
        in float in_brightness;
        uniform mat4 mvp;
        uniform float max_size;
        out float v_brightness;
        void main() {
            gl_Position = mvp * vec4(in_vert, 1.0);
            gl_PointSize = in_brightness * max_size;
            v_brightness = in_brightness;
        }
        """
        star_fragment_shader = """
        #version 330
        in float v_brightness;
        out vec4 fragColor;
        void main() {
            fragColor = vec4(vec3(v_brightness), 1.0);
        }
        """
        self.star_program = self.ctx.program(vertex_shader=star_vertex_shader, fragment_shader=star_fragment_shader)

        # パーティクル用のシェーダー
        # ★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★
        # ★ 修正箇所：パーティクル用シェーダーから 'in_vel' を削除
        # ★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★
        particle_vertex_shader = """
        #version 330
        in vec3 in_pos;
        in float in_life;
        in vec3 in_color;
        uniform mat4 mvp;
        out vec3 v_color;
        out float v_alpha;
        void main() {
            if (in_life > 0.0) {
                gl_Position = mvp * vec4(in_pos, 1.0);
                gl_PointSize = 3.0 * in_life; // 寿命に応じて小さくなる
                v_color = in_color;
                v_alpha = in_life; // 寿命に応じて薄くなる
            } else {
                gl_Position = vec4(-2.0, -2.0, -2.0, 1.0); // 画面外に飛ばす
            }
        }
        """
        particle_fragment_shader = """
        #version 330
        in vec3 v_color;
        in float v_alpha;
        out vec4 fragColor;
        void main() {
            fragColor = vec4(v_color, v_alpha);
        }
        """
        self.particle_program = self.ctx.program(vertex_shader=particle_vertex_shader, fragment_shader=particle_fragment_shader)

    def _init_buffers(self):
        # 球体用バッファ
        res=self.config.get('art','sphere_resolution');v,i=self._generate_sphere_mesh(1.,res)
        self.sphere_vbo=self.ctx.buffer(v.tobytes());self.sphere_ibo=self.ctx.buffer(i.tobytes())
        inst_c=self.config.get('art','sphere_count')+10;self.instance_vbo=self.ctx.buffer(reserve=inst_c*11*4)
        self.sphere_vao=self.ctx.vertex_array(self.sphere_program,[(self.sphere_vbo,'3f','position'),(self.instance_vbo,'3f 3f 1f 3f 1f/i','sphere_pos','object_color','alpha','rotation','radius')],self.sphere_ibo)
        
    def _generate_sphere_mesh(self,r,res):
        v,ind=[],[];
        for i in range(res+1):
            lat=np.pi*(-.5+float(i)/res)
            for j in range(res+1):lon=2*np.pi*float(j)/res;v.append([r*np.cos(lat)*np.cos(lon),r*np.cos(lat)*np.sin(lon),r*np.sin(lat)])
        for i in range(res):
            for j in range(res):v1=i*(res+1)+j;v2=v1+res+1;ind.extend([v1,v2,v1+1,v2,v2+1,v1+1])
        return np.array(v,dtype='f4'),np.array(ind,dtype='i4')

    def _mediapipe_worker(self):
        mp_holistic = mp.solutions.holistic
        holistic = mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5, refine_face_landmarks=True)
        mp_drawing = mp.solutions.drawing_utils
        
        while self.running:
            ret,frame=self.cap.read()
            if not ret:continue
            frame=cv2.flip(frame,1);rgb=cv2.cvtColor(frame,cv2.COLOR_BGR2RGB);rgb.flags.writeable=False
            results=holistic.process(rgb);rgb.flags.writeable=True
            
            mp_drawing.draw_landmarks(frame, results.face_landmarks, mp_holistic.FACEMESH_CONTOURS, landmark_drawing_spec=None, connection_drawing_spec=mp_drawing.DrawingSpec(color=(80,110,10), thickness=1, circle_radius=1))
            mp_drawing.draw_landmarks(frame, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
            mp_drawing.draw_landmarks(frame, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS)

            left_hand = results.right_hand_landmarks
            right_hand = results.left_hand_landmarks
            
            try:
                self.landmarks_queue.put_nowait({'left': left_hand, 'right': right_hand, 'face': results.face_landmarks})
                fps=self.fps_history[-1] if self.fps_history else 0
                self.video_queue.put_nowait(self.debug_info.update_and_draw(frame,fps,self.sphere_manager, self.tracker))
            except queue.Full:
                try:self.landmarks_queue.get_nowait();self.video_queue.get_nowait()
                except queue.Empty:pass
        holistic.close()

    def run(self):
        self.mediapipe_thread.start();cv2.namedWindow('Motion Capture View',cv2.WINDOW_NORMAL)
        cv2.resizeWindow('Motion Capture View',self.camera_width//2,self.camera_height//2)
        latest_landmarks=None
        while self.running:
            for e in pygame.event.get():
                if e.type==pygame.QUIT or(e.type==pygame.KEYDOWN and e.key==pygame.K_ESCAPE):self.running=False
                if e.type==pygame.KEYDOWN:
                    if e.key==pygame.K_F11:self._toggle_fullscreen()
                    if e.key==pygame.K_d:self.debug_info.toggle()
                    if e.key==pygame.K_t:self.sphere_manager.reset()
            try:cv2.imshow('Motion Capture View',self.video_queue.get_nowait())
            except queue.Empty:pass
            if cv2.waitKey(1)&0xFF==27:self.running=False
            c_t=time.time();dt=c_t-self.last_time;self.last_time=c_t
            if dt>0:self.fps_history.append(1./dt)
            try:latest_landmarks=self.landmarks_queue.get_nowait()
            except queue.Empty:pass
            
            if latest_landmarks:
                self.tracker.update_from_landmarks(
                    latest_landmarks['left'],
                    latest_landmarks['right'],
                    latest_landmarks['face']
                )
            
            # Update logic
            self._update_lighting(self.tracker.get_face_direction())
            fingertip_positions = self.tracker.get_current_fingertip_positions()
            self.sphere_manager.update(dt, fingertip_positions)
            self.sphere_manager.update_indicators(fingertip_positions)
            self.starfield.update(dt)
            self.particle_manager.update(dt)

            # Render
            self.ctx.clear(.01,.01,.03,1.)
            mvp=self.camera.get_mvp_matrix()
            
            self.ctx.disable(moderngl.DEPTH_TEST)
            self.starfield.render(mvp)
            self.particle_manager.render(mvp)
            self.ctx.enable(moderngl.DEPTH_TEST)

            self._render_instanced(mvp)
            
            pygame.display.flip()
            self.clock.tick(60)
        self.cleanup()

    def _update_lighting(self, direction):
        target_light_dir = np.array([0.5, 1.0, 0.8], dtype='f4') # Default
        if direction == "Looking Left":
            target_light_dir = np.array([-1.0, 0.5, 0.8], dtype='f4')
        elif direction == "Looking Right":
            target_light_dir = np.array([1.0, 0.5, 0.8], dtype='f4')
        elif direction == "Looking Up":
            target_light_dir = np.array([0.0, 1.0, 0.8], dtype='f4')
        elif direction == "Looking Down":
            target_light_dir = np.array([0.0, -1.0, 0.8], dtype='f4')
        
        # Lerp for smooth transition
        self.light_direction += (target_light_dir - self.light_direction) * self.light_smoothing

    def _render_instanced(self,mvp):
        all_objs=self.sphere_manager.get_all_visible_objects()
        if not all_objs:return
        inst_data=np.array([(*s.position,*s.color,s.alpha,*s.rotation,s.radius) for s in all_objs],dtype='f4')
        self.instance_vbo.write(inst_data)
        self.sphere_program['mvp'].write(mvp)
        self.sphere_program['u_light_direction'].write(self.light_direction.tobytes())
        self.sphere_vao.render(instances=len(all_objs))

    def _toggle_fullscreen(self):
        self.fullscreen=not self.fullscreen;pygame.display.quit();self._init_pygame()
        self.ctx.viewport=(0,0,*self.screen_size);self.camera.update_aspect_ratio(*self.screen_size)
        print(f"ディスプレイモードを切り替えました: {'フルスクリーン' if self.fullscreen else 'ウィンドウ'}")

    def cleanup(self):
        print("クリーンアップ処理を実行中...");self.running=False
        if self.mediapipe_thread.is_alive():self.mediapipe_thread.join(timeout=1)
        self.cap.release();cv2.destroyAllWindows();pygame.quit();print("クリーンアップ完了。")

if __name__=='__main__':
    try:
        app=App(Config('config.json'))
        app.run()
    except Exception as e:
        print(f"アプリケーションの実行中に致命的なエラーが発生しました: {e}")
        import traceback;traceback.print_exc()
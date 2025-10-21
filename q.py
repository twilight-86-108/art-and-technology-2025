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
import types

try:
    import imageio.v2 as imageio
except ImportError:
    print("[エラー] imageioライブラリが見つかりません。'pip install imageio imageio-ffmpeg' を実行してください。")
    sys.exit(1)

# =============================================================================
# 1. 設定管理クラス (変更なし)
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
    
    def get_dict(self, key, default=None):
        """通常の辞書のgetメソッドと同じ動作"""
        return self._data.get(key, default)
    
    def set(self, *keys, value):
        """ネストしたキーで設定値を設定する（実行時のみ）"""
        if len(keys) == 0:
            return
        
        current = self._data
        for key in keys[:-1]:
            if key not in current:
                current[key] = {}
            current = current[key]
        current[keys[-1]] = value

# =============================================================================
# 2. 音響システム
# =============================================================================
class AudioSystem:
    """球体消失時の音響効果を管理するクラス"""
    def __init__(self, config):
        self.config = config
        self.enabled = True
        self.volume = 0.7
        self.sounds = {}
        
        try:
            pygame.mixer.init(frequency=22050, size=-16, channels=2, buffer=512)
            self._load_sounds()
            print("✓ 音響システムを初期化しました")
        except pygame.error as e:
            print(f"[警告] 音響システムの初期化に失敗: {e}")
            self.enabled = False

    def _load_sounds(self):
        """音声ファイルを生成または読み込み"""
        # プロシージャル音声生成（ファイルが無い場合の代替）
        self.sounds['pop'] = self._generate_pop_sound()
        self.sounds['chime'] = self._generate_chime_sound()
        self.sounds['explosion'] = self._generate_explosion_sound()
        self.sounds['chain'] = self._generate_chain_sound()
        
        # 外部ファイルがあれば読み込み（オプション）
        sound_files = {
            'pop': 'audio/pop.wav',
            'chime': 'audio/chime.wav',
            'explosion': 'audio/explosion.wav',
            'chain': 'audio/chain.wav'
        }
        
        for name, path in sound_files.items():
            if os.path.exists(path):
                try:
                    self.sounds[name] = pygame.mixer.Sound(path)
                    print(f"✓ 外部音声ファイル読み込み: {name}")
                except pygame.error:
                    pass  # プロシージャル音声を使用

    def _generate_pop_sound(self, duration=0.1, frequency=800):
        """ポップ音をプロシージャルに生成"""
        sample_rate = 22050
        frames = int(duration * sample_rate)
        
        # エンベロープと周波数変調
        envelope = np.exp(-np.linspace(0, 5, frames))
        frequency_mod = frequency * (1 + 0.3 * np.exp(-np.linspace(0, 3, frames)))
        
        # 波形生成
        wave = np.sin(2 * np.pi * frequency_mod * np.linspace(0, duration, frames))
        wave = wave * envelope * 0.3
        
        # ステレオ変換（C-contiguous配列として作成）
        stereo_wave = np.zeros((frames, 2), dtype=np.float32)
        stereo_wave[:, 0] = wave  # 左チャンネル
        stereo_wave[:, 1] = wave  # 右チャンネル
        
        # int16に変換してC-contiguousを保証
        stereo_wave = np.ascontiguousarray((stereo_wave * 32767).astype(np.int16))
        
        return pygame.sndarray.make_sound(stereo_wave)

    def _generate_chime_sound(self, duration=0.3):
        """チャイム音をプロシージャルに生成"""
        sample_rate = 22050
        frames = int(duration * sample_rate)
        
        # ハーモニック構造
        fundamentals = [523, 659, 784]  # C5, E5, G5
        wave = np.zeros(frames)
        
        for i, freq in enumerate(fundamentals):
            envelope = np.exp(-np.linspace(0, 2, frames))
            harmonic = np.sin(2 * np.pi * freq * np.linspace(0, duration, frames))
            wave += harmonic * envelope * (0.4 - i * 0.1)
        
        wave = wave * 0.2
        
        # ステレオ変換（C-contiguous配列として作成）
        stereo_wave = np.zeros((frames, 2), dtype=np.float32)
        stereo_wave[:, 0] = wave  # 左チャンネル
        stereo_wave[:, 1] = wave  # 右チャンネル
        
        # int16に変換してC-contiguousを保証
        stereo_wave = np.ascontiguousarray((stereo_wave * 32767).astype(np.int16))
        
        return pygame.sndarray.make_sound(stereo_wave)

    def _generate_explosion_sound(self, duration=0.4):
        """爆発音をプロシージャルに生成"""
        sample_rate = 22050
        frames = int(duration * sample_rate)
        
        # ノイズベース
        noise = np.random.random(frames) * 2 - 1
        
        # ローパスフィルター効果
        envelope = np.exp(-np.linspace(0, 4, frames))
        filtered_noise = noise * envelope
        
        # 低周波成分追加
        rumble = np.sin(2 * np.pi * 60 * np.linspace(0, duration, frames)) * envelope * 0.3
        
        wave = (filtered_noise + rumble) * 0.15
        
        # ステレオ変換（C-contiguous配列として作成）
        stereo_wave = np.zeros((frames, 2), dtype=np.float32)
        stereo_wave[:, 0] = wave  # 左チャンネル
        stereo_wave[:, 1] = wave  # 右チャンネル
        
        # int16に変換してC-contiguousを保証
        stereo_wave = np.ascontiguousarray((stereo_wave * 32767).astype(np.int16))
        
        return pygame.sndarray.make_sound(stereo_wave)

    def _generate_chain_sound(self, duration=0.2):
        """チェーンリアクション音をプロシージャルに生成"""
        sample_rate = 22050
        frames = int(duration * sample_rate)
        
        # 周波数スイープ
        start_freq, end_freq = 1200, 400
        frequency = np.linspace(start_freq, end_freq, frames)
        
        envelope = np.exp(-np.linspace(0, 3, frames))
        wave = np.sin(2 * np.pi * frequency * np.linspace(0, duration, frames) / sample_rate)
        wave = wave * envelope * 0.25
        
        # ステレオ変換（C-contiguous配列として作成）
        stereo_wave = np.zeros((frames, 2), dtype=np.float32)
        stereo_wave[:, 0] = wave  # 左チャンネル
        stereo_wave[:, 1] = wave  # 右チャンネル
        
        # int16に変換してC-contiguousを保証
        stereo_wave = np.ascontiguousarray((stereo_wave * 32767).astype(np.int16))
        
        return pygame.sndarray.make_sound(stereo_wave)

    def play_sphere_pop(self, position=None, sphere_color=None):
        """球体消失音を再生"""
        if not self.enabled:
            return
            
        sound = self.sounds['pop']
        
        # 色に基づく音程調整
        if sphere_color is not None:
            brightness = np.mean(sphere_color)
            pitch_factor = 0.8 + brightness * 0.4  # 明るい色ほど高音
            sound.set_volume(self.volume * pitch_factor)
        
        sound.play()

    def play_chain_reaction(self, depth=1):
        """チェーンリアクション音を再生"""
        if not self.enabled:
            return
            
        sound = self.sounds['chain']
        # 深度に応じて音量調整
        volume = max(0.1, self.volume * (1.0 - depth * 0.1))
        sound.set_volume(volume)
        sound.play()

    def play_explosion(self, intensity=1.0):
        """爆発音を再生"""
        if not self.enabled:
            return
            
        sound = self.sounds['explosion']
        sound.set_volume(self.volume * min(1.0, intensity))
        sound.play()

    def play_gesture_success(self):
        """ジェスチャー成功音を再生"""
        if not self.enabled:
            return
            
        self.sounds['chime'].play()

    def set_volume(self, volume):
        """マスターボリューム設定"""
        self.volume = max(0.0, min(1.0, volume))

    def toggle_enabled(self):
        """音響システムのON/OFF切り替え"""
        self.enabled = not self.enabled
        status = "ON" if self.enabled else "OFF"
        print(f"音響システム: {status}")

# =============================================================================
# 3. 強化された顔向き検出システム
# =============================================================================
class EnhancedFaceDirectionDetector:
    """数値ベースの高精度顔向き検出システム"""
    def __init__(self, config, frame_shape):
        self.config = config
        self.frame_shape = frame_shape
        
        # 数値データ
        self.yaw_angle = 0.0      # -90°〜+90° (左右)
        self.pitch_angle = 0.0    # -90°〜+90° (上下)
        self.roll_angle = 0.0     # -180°〜+180° (回転)
        self.confidence = 0.0     # 0.0〜1.0
        
        # 従来の文字列表現も維持
        self.direction_text = "Center"
        
        # 3D方向ベクトル
        self.direction_vector = np.array([0.0, 0.0, -1.0], dtype=np.float32)
        
        # 顔の3Dモデルポイント（標準的な顔のランドマーク）
        self.model_points = np.array([
            (0.0, 0.0, 0.0),           # 鼻先
            (0.0, -330.0, -65.0),      # 顎
            (-225.0, 170.0, -135.0),   # 左目尻
            (225.0, 170.0, -135.0),    # 右目尻
            (-150.0, -150.0, -125.0),  # 左口角
            (150.0, -150.0, -125.0)    # 右口角
        ], dtype=np.float64)
        
        # カメラ内部パラメータ
        focal_length = self.frame_shape[1]
        center = (self.frame_shape[1] / 2, self.frame_shape[0] / 2)
        self.camera_matrix = np.array([
            [focal_length, 0, center[0]],
            [0, focal_length, center[1]],
            [0, 0, 1]
        ], dtype=np.float64)
        
        self.dist_coeffs = np.zeros((4, 1))
        
        # 閾値設定
        self.yaw_threshold = self.config.get('face_detection', 'yaw_threshold', default=20.0)
        self.pitch_threshold = self.config.get('face_detection', 'pitch_threshold', default=15.0)
        
        # スムージング用
        self.smoothing_factor = 0.3
        self.last_valid_angles = [0.0, 0.0, 0.0]
        
        print("✓ 強化された顔向き検出システムを初期化")

    def update(self, face_landmarks):
        """顔のランドマークから詳細な向き情報を計算"""
        if not face_landmarks:
            self.confidence = 0.0
            self.direction_text = "No Face"
            return
        
        try:
            # 2Dランドマークポイントを抽出
            landmark_indices = [1, 152, 33, 263, 61, 291]  # 鼻先、顎、目尻、口角
            image_points = np.array([
                (face_landmarks.landmark[i].x * self.frame_shape[1],
                 face_landmarks.landmark[i].y * self.frame_shape[0])
                for i in landmark_indices
            ], dtype=np.float64)
            
            # PnPソルバーで3D姿勢を推定
            success, rotation_vector, translation_vector = cv2.solvePnP(
                self.model_points, image_points, 
                self.camera_matrix, self.dist_coeffs,
                flags=cv2.SOLVEPNP_ITERATIVE
            )
            
            if success:
                # 回転行列に変換
                rotation_matrix, _ = cv2.Rodrigues(rotation_vector)
                
                # オイラー角を計算
                angles = self._rotation_matrix_to_euler_angles(rotation_matrix)
                
                # スムージング適用
                smoothed_angles = [
                    self.last_valid_angles[i] * (1 - self.smoothing_factor) + 
                    angles[i] * self.smoothing_factor
                    for i in range(3)
                ]
                
                self.yaw_angle = smoothed_angles[1]    # Y軸回転
                self.pitch_angle = smoothed_angles[0]  # X軸回転  
                self.roll_angle = smoothed_angles[2]   # Z軸回転
                
                self.last_valid_angles = smoothed_angles
                
                # 3D方向ベクトルを計算
                self.direction_vector = self._calculate_direction_vector(rotation_matrix)
                
                # 信頼度を計算（ランドマークの分散から推定）
                self.confidence = self._calculate_confidence(face_landmarks)
                
                # 従来の文字列表現を更新
                self._update_direction_text()
                
            else:
                self.confidence = 0.0
                self.direction_text = "Calculation Failed"
                
        except Exception as e:
            self.confidence = 0.0
            self.direction_text = "Error"
            print(f"[警告] 顔向き計算エラー: {e}")

    def _rotation_matrix_to_euler_angles(self, R):
        """回転行列からオイラー角を計算"""
        sy = math.sqrt(R[0,0] * R[0,0] + R[1,0] * R[1,0])
        
        singular = sy < 1e-6
        
        if not singular:
            x = math.atan2(R[2,1], R[2,2])
            y = math.atan2(-R[2,0], sy)
            z = math.atan2(R[1,0], R[0,0])
        else:
            x = math.atan2(-R[1,2], R[1,1])
            y = math.atan2(-R[2,0], sy)
            z = 0
        
        return [math.degrees(x), math.degrees(y), math.degrees(z)]

    def _calculate_direction_vector(self, rotation_matrix):
        """回転行列から3D方向ベクトルを計算"""
        # Z軸の負方向（前方）を回転
        forward = np.array([0, 0, -1], dtype=np.float32)
        direction = rotation_matrix @ forward
        return direction.astype(np.float32)

    def _calculate_confidence(self, face_landmarks):
        """ランドマークの安定性から信頼度を計算"""
        # 主要ランドマークの相対位置の安定性を評価
        key_points = [1, 33, 263, 61, 291]  # 鼻、目、口
        positions = np.array([
            [face_landmarks.landmark[i].x, face_landmarks.landmark[i].y]
            for i in key_points
        ])
        
        # 分散を計算（低いほど安定）
        variance = np.var(positions, axis=0).mean()
        confidence = max(0.0, min(1.0, 1.0 - variance * 10))
        
        return confidence

    def _update_direction_text(self):
        """数値角度から文字列表現を生成"""
        if abs(self.yaw_angle) < self.yaw_threshold and abs(self.pitch_angle) < self.pitch_threshold:
            self.direction_text = "Center"
        elif abs(self.yaw_angle) > abs(self.pitch_angle):
            self.direction_text = "Looking Right" if self.yaw_angle > 0 else "Looking Left"
        else:
            self.direction_text = "Looking Up" if self.pitch_angle < 0 else "Looking Down"

    # アクセサメソッド
    def get_angles(self):
        """現在の角度を取得"""
        return {
            'yaw': self.yaw_angle,
            'pitch': self.pitch_angle, 
            'roll': self.roll_angle,
            'confidence': self.confidence
        }

    def get_direction_vector(self):
        """3D方向ベクトルを取得"""
        return self.direction_vector.copy()

    def get_normalized_direction(self):
        """正規化された方向（-1〜1）を取得"""
        return {
            'horizontal': np.clip(self.yaw_angle / 90.0, -1.0, 1.0),
            'vertical': np.clip(self.pitch_angle / 90.0, -1.0, 1.0),
            'confidence': self.confidence
        }

    def get_direction(self):
        """従来の文字列表現（後方互換性）"""
        return self.direction_text

# =============================================================================
# 4. 修正版拡張ジェスチャーシステム
# =============================================================================
class EnhancedGestureSystem:
    """複数のジェスチャーを認識・管理するシステム（修正版）"""
    def __init__(self, config):
        self.config = config
        self.gesture_states = {}
        self.gesture_timers = {}
        
        # ジェスチャー履歴（スムージング用）
        self.gesture_history = {
            'palm_open_left': deque(maxlen=5),
            'palm_open_right': deque(maxlen=5),
            'finger_count': deque(maxlen=5),
        }
        
        # クールダウン管理
        self.cooldowns = {
            'palm_summon_left': 0.0,
            'palm_summon_right': 0.0,
            'finger_effect': 0.0,
        }
        
        print("✓ 拡張ジェスチャーシステムを初期化")

    def update(self, left_hand, right_hand):
        """全ジェスチャーを更新（修正版）"""
        current_time = time.time()
        
        # 既存の重力井戸ジェスチャー
        gravity_result = self._detect_gravity_well(left_hand, right_hand)
        
        # 新ジェスチャーの検出
        palm_result = self._detect_palm_gestures(left_hand, right_hand)
        finger_result = self._detect_finger_count(left_hand, right_hand)
        rotation_result = self._detect_hand_rotation(left_hand, right_hand)
        distance_result = self._detect_hand_distance(left_hand, right_hand)
        
        # ### 修正: 統一された形式で結果を保存 ###
        self.gesture_states = {
            'gravity_well': gravity_result,
            'palm_open': palm_result,
            'finger_count': finger_result,
            'hand_rotation': rotation_result,
            'hand_distance': distance_result,
        }
        
        # タイマー更新（activeキーが存在する場合のみ）
        for gesture, state in self.gesture_states.items():
            if isinstance(state, dict) and 'active' in state and state['active']:
                if gesture not in self.gesture_timers:
                    self.gesture_timers[gesture] = current_time
            else:
                self.gesture_timers.pop(gesture, None)

    def _detect_gravity_well(self, left_hand, right_hand):
        """既存の重力井戸ジェスチャー（両手つまみ）"""
        if not left_hand or not right_hand:
            return {'active': False, 'position': None, 'strength': 0.0}
            
        try:
            # 親指と人差し指の位置
            l_thumb = self._landmark_to_array(left_hand.landmark[4])
            l_index = self._landmark_to_array(left_hand.landmark[8])
            r_thumb = self._landmark_to_array(right_hand.landmark[4])
            r_index = self._landmark_to_array(right_hand.landmark[8])
            
            # つまみ距離
            l_pinch_dist = np.linalg.norm(l_thumb - l_index)
            r_pinch_dist = np.linalg.norm(r_thumb - r_index)
            
            threshold = 0.1
            
            if l_pinch_dist < threshold and r_pinch_dist < threshold:
                center = (l_thumb + l_index + r_thumb + r_index) / 4.0
                strength = 1.0 - (l_pinch_dist + r_pinch_dist) / (2 * threshold)
                return {'active': True, 'position': center, 'strength': strength}
                
        except Exception:
            pass
            
        return {'active': False, 'position': None, 'strength': 0.0}

    def _detect_palm_gestures(self, left_hand, right_hand):
        """手のひらの開閉を検出（修正版）"""
        results = {
            'left': {'open': False, 'openness': 0.0, 'position': None, 'ready_to_summon': False},
            'right': {'open': False, 'openness': 0.0, 'position': None, 'ready_to_summon': False}
        }
        
        current_time = time.time()
        
        for hand_name, hand_data in [('left', left_hand), ('right', right_hand)]:
            if not hand_data:
                continue
                
            try:
                # 手のひら中心と指先の距離を計算
                palm_center = self._landmark_to_array(hand_data.landmark[0])  # 手首
                fingertips = [
                    self._landmark_to_array(hand_data.landmark[i]) 
                    for i in [4, 8, 12, 16, 20]  # 全指先
                ]
                
                # 手のひらの開き具合を計算
                distances = [np.linalg.norm(tip - palm_center) for tip in fingertips]
                avg_distance = np.mean(distances)
                
                # 正規化（経験的な値）
                openness = np.clip(avg_distance / 0.25, 0.0, 1.0)
                is_open = openness > 0.7
                
                # 手の位置（3D空間用）
                hand_position = self._landmark_to_world_pos(hand_data.landmark[9])  # 中指基部
                
                # 召喚準備状態の判定
                cooldown_key = f'palm_summon_{hand_name}'
                cooldown_time = self.cooldowns.get(cooldown_key, 0.0)
                ready_to_summon = is_open and openness > 0.8 and (current_time - cooldown_time) > 1.0
                
                results[hand_name] = {
                    'open': is_open,
                    'openness': openness,
                    'position': hand_position,
                    'ready_to_summon': ready_to_summon
                }
                
            except Exception as e:
                print(f"[警告] 手のひら検出エラー ({hand_name}): {e}")
                pass
                
        return results

    def _detect_finger_count(self, left_hand, right_hand):
        """指の数を検出（10本指対応版）"""
        results = {
            'left': 0, 
            'right': 0, 
            'total': 0, 
            'effect_ready': False,
            'effect_type': 'none',
            'hand_balance': 'none',  # 新規追加
            'combination_type': 'none'  # 新規追加
        }
    
        current_time = time.time()
    
        for hand_name, hand_data in [('left', left_hand), ('right', right_hand)]:
            if not hand_data:
                continue
            
            try:
                finger_count = 0
                landmarks = hand_data.landmark
            
                # 親指（より厳密な判定）
                thumb_tip = landmarks[4]
                thumb_ip = landmarks[3]
                thumb_mcp = landmarks[2]
            
                if hand_name == 'left':
                    is_thumb_extended = (thumb_tip.x > thumb_ip.x + 0.03 and 
                                   thumb_tip.y < thumb_mcp.y)
                else:
                    is_thumb_extended = (thumb_tip.x < thumb_ip.x - 0.03 and 
                                   thumb_tip.y < thumb_mcp.y)
            
                if is_thumb_extended:
                    finger_count += 1
            
                # 他の指（より厳密な判定）
                finger_joints = [(8, 6, 5), (12, 10, 9), (16, 14, 13), (20, 18, 17)]
            
                for tip_idx, pip_idx, mcp_idx in finger_joints:
                    tip = landmarks[tip_idx]
                    pip = landmarks[pip_idx]
                    mcp = landmarks[mcp_idx]
                
                    is_extended = (tip.y < pip.y - 0.02 and
                             pip.y < mcp.y + 0.01)
                
                    if is_extended:
                        finger_count += 1
            
                results[hand_name] = finger_count
            
            except Exception as e:
                print(f"[警告] 指カウントエラー ({hand_name}): {e}")
                pass
    
        results['total'] = results['left'] + results['right']
    
        # 新機能: 手のバランス分析
        results['hand_balance'] = self._analyze_hand_balance(results['left'], results['right'])
        results['combination_type'] = self._analyze_combination_type(results['left'], results['right'])
    
        # エフェクト準備状態の判定（クールダウン延長）
        total_fingers = results['total']
        if total_fingers > 0 and (current_time - self.cooldowns.get('finger_effect', 0.0)) > 1.5:
            results['effect_ready'] = True
        
            # 拡張エフェクトタイプの決定
            effect_types = {
                1: 'precision_pop',
                2: 'small_explosion', 
                3: 'chain_reaction',
                4: 'time_slow',
                5: 'screen_reset',
                6: 'color_shift',        # 新規
                7: 'gravity_reverse',    # 新規
                8: 'sphere_split',       # 新規
                9: 'space_warp',         # 新規
                10: 'genesis_effect'     # 新規
            }
            results['effect_type'] = effect_types.get(total_fingers, 'none')
    
        return results
    
    # 新規追加メソッド
    def _analyze_hand_balance(self, left_count, right_count):
        """両手のバランスを分析"""
        if left_count == 0 and right_count == 0:
            return 'none'
        elif left_count == 0:
            return 'right_only'
        elif right_count == 0:
            return 'left_only'
        elif left_count == right_count:
            return 'balanced'
        elif left_count > right_count:
            return 'left_heavy'
        else:
            return 'right_heavy'

    def _analyze_combination_type(self, left_count, right_count):
        """指の組み合わせパターンを分析"""
        total = left_count + right_count
    
        if total == 0:
            return 'none'
        elif left_count == right_count and left_count > 0:
            return f'symmetric_{left_count}x{right_count}'
        elif left_count == 0 or right_count == 0:
            return f'single_hand_{max(left_count, right_count)}'
        else:
            return f'asymmetric_{left_count}x{right_count}'

    def _detect_hand_rotation(self, left_hand, right_hand):
        """手の回転を検出（簡易版）"""
        return {'active': False, 'angle': 0.0, 'direction': 'none'}

    def _detect_hand_distance(self, left_hand, right_hand):
        """両手の距離を検出（簡易版）"""
        return {'active': False, 'distance': 0.0, 'type': 'none'}

    def _landmark_to_array(self, landmark):
        """ランドマークをnumpy配列に変換"""
        return np.array([landmark.x, landmark.y, landmark.z], dtype=np.float32)

    def _landmark_to_world_pos(self, landmark):
        """ランドマークを3D世界座標に変換（簡易版）"""
        # この関数は後でBodyTrackerの変換関数を使用するように修正予定
        x = (landmark.x - 0.5) * 20.0  # 暫定的なスケール
        y = (0.5 - landmark.y) * 20.0
        z = landmark.z * 10.0
        return np.array([x, y, z], dtype=np.float32)

    def get_gesture_state(self, gesture_name):
        """指定ジェスチャーの状態を取得"""
        return self.gesture_states.get(gesture_name, {'active': False})

    def is_gesture_active(self, gesture_name):
        """ジェスチャーがアクティブかを確認"""
        state = self.get_gesture_state(gesture_name)
        return state.get('active', False)

    def trigger_cooldown(self, cooldown_name, duration=1.0):
        """クールダウンを開始"""
        self.cooldowns[cooldown_name] = time.time()

# =============================================================================
# 5. 元のコードから必要なクラス群を再利用
# =============================================================================

PI = math.pi

# IBL Preprocessor クラス (最終完成・安定版)
class IBLPreprocessor:
    def __init__(self, ctx, app, hdr_texture):
        self.ctx = ctx
        self.app = app
        self.hdr_texture = hdr_texture
        self.env_cubemap = None
        self.irradiance_map = None
        self.prefilter_map = None
        self.brdf_lut = None

        self._setup_geometry()
        self._load_shaders()

    def _setup_geometry(self):
        cube_vertices = np.array([
            -1.0,  1.0, -1.0, -1.0, -1.0, -1.0,  1.0, -1.0, -1.0,  1.0, -1.0, -1.0,  1.0,  1.0, -1.0, -1.0,  1.0, -1.0,
            -1.0, -1.0,  1.0, -1.0, -1.0, -1.0, -1.0,  1.0, -1.0, -1.0,  1.0, -1.0, -1.0,  1.0,  1.0, -1.0, -1.0,  1.0,
             1.0, -1.0, -1.0,  1.0, -1.0,  1.0,  1.0,  1.0,  1.0,  1.0,  1.0,  1.0,  1.0,  1.0, -1.0,  1.0, -1.0, -1.0,
            -1.0, -1.0,  1.0, -1.0,  1.0,  1.0,  1.0,  1.0,  1.0,  1.0,  1.0,  1.0,  1.0, -1.0,  1.0, -1.0, -1.0,  1.0,
            -1.0,  1.0, -1.0,  1.0,  1.0, -1.0,  1.0,  1.0,  1.0,  1.0,  1.0,  1.0, -1.0,  1.0,  1.0, -1.0,  1.0, -1.0,
            -1.0, -1.0, -1.0, -1.0, -1.0,  1.0,  1.0, -1.0, -1.0,  1.0, -1.0, -1.0, -1.0, -1.0,  1.0,  1.0, -1.0,  1.0
        ], dtype='f4')
        self.cube_vbo = self.ctx.buffer(cube_vertices)

        quad_vertices = np.array([
            -1.0,  1.0, 0.0, 1.0, -1.0, -1.0, 0.0, 0.0, 1.0, -1.0, 1.0, 0.0, 1.0, 1.0, 1.0, 1.0,
        ], dtype='f4')
        self.quad_vbo = self.ctx.buffer(quad_vertices)

    def _load_shaders(self):
        self.equirect_to_cube_prog = self.ctx.program(
            vertex_shader=self.app._load_shader_source('cube.vert'),
            fragment_shader=self.app._load_shader_source('equirect_to_cube.frag'))
        self.irradiance_prog = self.ctx.program(
            vertex_shader=self.app._load_shader_source('cube.vert'),
            fragment_shader=self.app._load_shader_source('irradiance_convolution.frag'))
        self.brdf_prog = self.ctx.program(
            vertex_shader=self.app._load_shader_source('fullscreen_quad.vert'),
            fragment_shader=self.app._load_shader_source('brdf_integration.frag'))

    def preprocess(self):
        print("IBL: Pre-processing environment map...")
        original_viewport = self.ctx.viewport
        
        self._equirectangular_to_cubemap()
        self._create_irradiance_map()
        self._create_prefilter_map() 
        self._create_brdf_lut()
        
        self.hdr_texture.release()

        self.ctx.viewport = original_viewport
        print("IBL: Pre-processing complete.")

    def _render_to_cube_face(self, fbo, vao, program, view_matrix):
        program['view'].write(view_matrix.tobytes())
        fbo.clear()
        vao.render()

    def _equirectangular_to_cubemap(self):
        env_map_size = 512
        self.env_cubemap = self.ctx.texture_cube(size=(env_map_size, env_map_size), components=3, dtype='f4')
        
        capture_texture = self.ctx.texture((env_map_size, env_map_size), 3, dtype='f4')
        capture_fbo = self.ctx.framebuffer(
            color_attachments=[capture_texture],
            depth_attachment=self.ctx.depth_renderbuffer((env_map_size, env_map_size))
        )

        projection = self.app.camera._perspective(90.0, 1.0, 0.1, 10.0)
        views = [ self.app.camera._look_at(np.array([0,0,0], dtype=np.float32), np.array(t, dtype=np.float32), np.array(u, dtype=np.float32))
            for t, u in [([1,0,0],[0,-1,0]), ([-1,0,0],[0,-1,0]), ([0,1,0],[0,0,1]), ([0,-1,0],[0,0,-1]), ([0,0,1],[0,-1,0]), ([0,0,-1],[0,-1,0])]
        ]

        self.equirect_to_cube_prog['projection'].write(projection.tobytes())
        self.hdr_texture.use(0)
        self.equirect_to_cube_prog['equirectangularMap'].value = 0
        vao = self.ctx.vertex_array(self.equirect_to_cube_prog, [(self.cube_vbo, '3f', 'aPos')])
        
        self.ctx.viewport = (0, 0, env_map_size, env_map_size)
        capture_fbo.use()
        for i in range(6):
            self._render_to_cube_face(capture_fbo, vao, self.equirect_to_cube_prog, views[i])
            data = capture_fbo.read(components=3, dtype='f4')
            self.env_cubemap.write(face=i, data=data)

        self.env_cubemap.build_mipmaps()
        capture_fbo.release()
        capture_texture.release()
        vao.release()

    def _create_irradiance_map(self):
        irradiance_size = 32
        self.irradiance_map = self.ctx.texture_cube((irradiance_size, irradiance_size), 3, dtype='f4')
        capture_texture = self.ctx.texture((irradiance_size, irradiance_size), 3, dtype='f4')
        capture_fbo = self.ctx.framebuffer(
            color_attachments=[capture_texture],
            depth_attachment=self.ctx.depth_renderbuffer((irradiance_size, irradiance_size))
        )
        
        projection = self.app.camera._perspective(90.0, 1.0, 0.1, 10.0)
        views = [ self.app.camera._look_at(np.array([0,0,0], dtype=np.float32), np.array(t, dtype=np.float32), np.array(u, dtype=np.float32))
            for t, u in [([1,0,0],[0,-1,0]), ([-1,0,0],[0,-1,0]), ([0,1,0],[0,0,1]), ([0,-1,0],[0,0,-1]), ([0,0,1],[0,-1,0]), ([0,0,-1],[0,-1,0])]
        ]
        
        self.irradiance_prog['projection'].write(projection.tobytes())
        self.env_cubemap.use(0)
        self.irradiance_prog['environmentMap'].value = 0
        vao = self.ctx.vertex_array(self.irradiance_prog, [(self.cube_vbo, '3f', 'aPos')])

        self.ctx.viewport = (0, 0, irradiance_size, irradiance_size)
        capture_fbo.use()
        for i in range(6):
            self._render_to_cube_face(capture_fbo, vao, self.irradiance_prog, views[i])
            data = capture_fbo.read(components=3, dtype='f4')
            self.irradiance_map.write(face=i, data=data)
        
        capture_fbo.release()
        capture_texture.release()
        vao.release()

    def _create_prefilter_map(self):
        self.prefilter_map = self.env_cubemap
        self.prefilter_map.filter = (moderngl.LINEAR_MIPMAP_LINEAR, moderngl.LINEAR)
        self.prefilter_map.build_mipmaps()

    def _create_brdf_lut(self):
        brdf_size = 512
        self.brdf_lut = self.ctx.texture((brdf_size, brdf_size), 2, dtype='f4')
        capture_fbo = self.ctx.framebuffer(color_attachments=[self.brdf_lut])
        vao = self.ctx.vertex_array(self.brdf_prog, [(self.quad_vbo, '2f 2f', 'aPos', 'aTexCoords')])
        
        self.ctx.viewport = (0, 0, brdf_size, brdf_size)
        capture_fbo.use()
        capture_fbo.clear()
        vao.render(mode=moderngl.TRIANGLE_STRIP)
        
        capture_fbo.release()
        vao.release()

class BaseSphere:
    """全ての球体の基底クラス"""
    def __init__(self, position, radius, color=None, texture_id=0, normal_map_id=0):
        self.position = np.array(position, dtype=np.float32)
        self.original_position = self.position.copy()
        self.radius = radius
        self.current_radius = radius
        self.color = color if color is not None else np.array([1.0, 1.0, 1.0], dtype=np.float32)
        self.rotation = np.array([random.uniform(0, 2 * PI), random.uniform(0, 2 * PI), random.uniform(0, 2 * PI)], dtype=np.float32)
        self.rotation_speed = np.array([random.uniform(-0.1, 0.1) for _ in range(3)], dtype=np.float32)
        self.alpha = 1.0
        self.texture_id = texture_id
        self.normal_map_id = normal_map_id
        self.dynamic_roughness = 0.5
        self.dynamic_metallic = 0.1
        self.dynamic_emission = 0.0
        self.dynamic_color_multiplier = 1.0
        self.velocity = np.array([0.0, 0.0, 0.0], dtype=np.float32)

    def update(self, dt, current_time):
        self.rotation += self.rotation_speed * dt
        self.position += self.velocity * dt
        self.velocity *= 0.98

class DisappearingSphere(BaseSphere):
    """消失機能付き球体（音響対応版）"""
    def __init__(self, position, radius, color, config, texture_id, normal_map_id=0):
        super().__init__(position, radius, color, texture_id, normal_map_id)
        self.config = config
        self.state = 'appearing'
        self.state_time = 0.0
        self.respawn_delay = random.uniform(3.0, 6.0)
        self.fade_speed = 4.0
        self.hover_scale = self.config.get('interaction', 'hover_scale', default=1.5)
        self.touch_cooldown = self.config.get('interaction', 'touch_cooldown', default=0.1)
        self.is_in_chain_reaction = False
        self.alpha = 0.0
        self.float_offset = random.uniform(0, 2 * PI)
        self.original_color = color.copy()

    def update(self, dt, current_time):
        """球体の更新処理（重力反転効果の時間制限追加）"""
        super().update(dt, current_time)
        self.state_time += dt

        # 🔧 修正点7: 重力反転効果の自動解除
        if hasattr(self, 'gravity_reverse_time'):
            time_since_reverse = current_time - self.gravity_reverse_time
            if time_since_reverse > 3.0:  # 3秒後に効果を減衰
                self.velocity[1] *= 0.95  # 上向き速度を徐々に減衰
                self.dynamic_emission *= 0.98  # 発光効果も減衰
            
                # 5秒後に完全に効果を解除
                if time_since_reverse > 5.0:
                    delattr(self, 'gravity_reverse_time')
                    self.dynamic_emission = 0.0

        # 浮遊効果の計算
        float_y = math.sin(current_time * 0.5 + self.float_offset) * 0.02
    
        # 🔧 修正点8: original_positionの更新を制限
        # 重力反転中でない場合のみ通常の浮遊処理を行う
        if not hasattr(self, 'gravity_reverse_time'):
            self.original_position[1] += (self.position[1] - self.original_position[1]) * 0.1
            self.position[1] = self.original_position[1] + float_y
        else:
            # 重力反転中は浮遊効果を最小限にする
            self.position[1] += float_y * 0.1

        # 状態に応じた処理（既存コードと同じ）
        if self.state == 'disappearing':
            self.alpha -= self.fade_speed * dt
            if self.alpha <= 0:
                self.state = 'invisible'
                self.alpha = 0
                self.state_time = 0
                self.is_in_chain_reaction = False
        elif self.state == 'invisible':
            if self.state_time > self.respawn_delay:
                self.state = 'appearing'
                # 🔧 修正点9: リスポーン時の完全リセット
                self.position = self.original_position.copy()
                self.velocity = np.zeros(3, dtype=np.float32)
                if hasattr(self, 'gravity_reverse_time'):
                    delattr(self, 'gravity_reverse_time')
                self.state_time = 0
        elif self.state == 'appearing':
            self.alpha += self.fade_speed * dt
            if self.alpha >= 1.0:
                self.state = 'visible'
                self.alpha = 1.0
                self.state_time = 0
        elif self.state == 'hover':
            target_radius = self.radius * self.hover_scale
            self.current_radius += (target_radius - self.current_radius) * 15.0 * dt
        elif self.state == 'visible':
            target_radius = self.radius
            self.current_radius += (target_radius - self.current_radius) * 10.0 * dt
    
    def handle_touch(self, audio_system=None):
        """球体タッチ処理（音響システム対応）"""
        if self.state in ['visible', 'hover'] and self.state_time > self.touch_cooldown:
            self.state = 'disappearing'
            self.state_time = 0
            
            # 音響効果を再生
            if audio_system:
                audio_system.play_sphere_pop(self.position, self.color)
            
            return True
        return False

class HandIndicator:
    """手のインジケータ専用クラス"""
    def __init__(self, config):
        self.config_all = config
        self.config = config.get_dict('hand_indicators')
        self.base_color = np.array(self.config.get('base_color'), dtype=np.float32)
        self.pulse_speed = self.config.get('pulse_speed')
        self.pulse_intensity = self.config.get('pulse_intensity')
        self.trail_length = self.config.get('trail_length')
        self.trail_fade_speed = self.config.get('trail_fade_speed')
        self.glow_radius_multiplier = 2.5

        self.position = np.zeros(3, dtype=np.float32)
        self.radius = self.config_all.get('hand_tracking', 'hand_indicator_size')
        self.visible = False
        self.trail = deque(maxlen=self.trail_length)

    def update(self, dt, current_time, target_position):
        self.visible = target_position is not None
        if self.visible:
            self.position += (np.array(target_position, dtype=np.float32) - self.position) * 0.5
            self.trail.append({'pos': self.position.copy(), 'alpha': 1.0})

        for t in self.trail:
            t['alpha'] *= self.trail_fade_speed

        while self.trail and self.trail[0]['alpha'] < 0.01:
            self.trail.popleft()

    def get_render_data(self, current_time):
        if not self.visible:
            return []

        data = []
        pulse = (math.sin(current_time * self.pulse_speed) * 0.5 + 0.5) * self.pulse_intensity
        main_color = self.base_color + pulse
        data.append({
            'pos': self.position, 'color': main_color, 'alpha': 1.0,
            'radius': self.radius * (1.0 + pulse * 0.5), 'type': 0.0 # Main
        })
        data.append({
            'pos': self.position, 'color': main_color, 'alpha': 0.4 * (1.0 + pulse),
            'radius': self.radius * self.glow_radius_multiplier, 'type': 2.0 # Glow
        })
        for t in self.trail:
            data.append({
                'pos': t['pos'], 'color': self.base_color, 'alpha': t['alpha'] * 0.5,
                'radius': self.radius * t['alpha'] * 0.8, 'type': 1.0 # Trail
            })
        return data

class DynamicMaterialSystem:
    def __init__(self, config):
        self.config = config
        self.material_config = config.get_dict('dynamic_materials', {})
        self.enabled = self.material_config.get('enable_material_animation', True)
        self.roughness_variation = self.material_config.get('roughness_variation', 0.2)
        self.metallic_variation = self.material_config.get('metallic_variation', 0.3)
        self.emission_variation = self.material_config.get('emission_variation', 0.5)
        self.animation_speed = self.material_config.get('animation_speed', 0.8)
        self.interaction_response = self.material_config.get('interaction_response', True)
        self.sphere_timers = {}

    def initialize_sphere(self, sphere_id):
        if not self.enabled: return
        self.sphere_timers[sphere_id] = {'offset': random.uniform(0, 2 * math.pi), 'speed_multiplier': random.uniform(0.8, 1.2), 'interaction_intensity': 0.0, 'last_touch_time': 0.0}

    def update_sphere_material(self, sphere, current_time, interaction_level=0.0):
        if not self.enabled: return
        sphere_id = id(sphere)
        if sphere_id not in self.sphere_timers: self.initialize_sphere(sphere_id)
        timer_data = self.sphere_timers[sphere_id]
        if interaction_level > 0.1:
            timer_data['interaction_intensity'] = min(1.0, timer_data['interaction_intensity'] + interaction_level)
            timer_data['last_touch_time'] = current_time
        else:
            timer_data['interaction_intensity'] *= math.exp(-(current_time - timer_data['last_touch_time']) * 2.0)
        phase = current_time * self.animation_speed * timer_data['speed_multiplier'] + timer_data['offset']
        base_wave, fast_wave = math.sin(phase), math.sin(phase * 3.0) * 0.3
        interaction_boost = timer_data['interaction_intensity'] if self.interaction_response else 0.0
        sphere.dynamic_roughness = np.clip(0.3 + (base_wave * self.roughness_variation) + (interaction_boost * 0.4), 0.0, 1.0)
        sphere.dynamic_metallic = np.clip(0.1 + (fast_wave * self.metallic_variation) + (interaction_boost * 0.6), 0.0, 1.0)
        sphere.dynamic_emission = np.clip(abs(base_wave * self.emission_variation) + (interaction_boost * 0.8), 0.0, 2.0)
        sphere.dynamic_color_multiplier = 1.0 + sphere.dynamic_emission * 0.5

# =============================================================================
# 6. 拡張SphereManager（新ジェスチャー対応）
# =============================================================================
class SphereManager:
    """シーン内の全ての球体を管理するクラス（新ジェスチャー対応）"""
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
        
        self.dynamic_material_system = DynamicMaterialSystem(config)

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
        count = self.config.get('art', 'sphere_count', default=250)
        min_size = self.config.get('art', 'sphere_min_size', default=0.1)
        max_size = self.config.get('art', 'sphere_max_size', default=0.2)
        world_depth = self.config.get('hand_tracking', 'world_depth', default=15.0)
        half_w, half_h, half_d = self.world_width / 2.0, self.world_height / 2.0, world_depth / 2.0
        texture_paths = self.config.get('art', 'texture_paths', default=[])
        num_textures = len(texture_paths)
        normal_map_paths = self.config.get('art', 'normal_map_paths', default=[])
        num_normals = len(normal_map_paths)
        for _ in range(count):
            pos = [random.uniform(-half_w, half_w), random.uniform(-half_h, half_h), random.uniform(-half_d, half_d)]
            size = random.uniform(min_size, max_size)
            color = self._hsv_to_rgb(random.random(), 0.1, 1.0)
            texture_id = random.randint(0, num_textures - 1) if num_textures > 0 else 0
            norm_id = random.randint(0, num_normals - 1) if num_normals > 0 else 0
            sphere = DisappearingSphere(pos, size, color, self.config, texture_id, norm_id)
            self.spheres.append(sphere)
            self.dynamic_material_system.initialize_sphere(id(sphere))
        print(f"{len(self.spheres)}個の球体を3D空間内にランダム配置しました。")

    def _create_hand_indicators(self):
        count = self.config.get('hand_tracking', 'hand_indicator_count')
        self.hand_indicators = [HandIndicator(self.config) for _ in range(count)]

    def _trigger_chain_reaction(self, origin_sphere, depth, audio_system=None):
        """チェーンリアクション（音響対応）"""
        if depth >= self.chain_reaction_max_depth:
            return

        for sphere in self.spheres:
            if sphere.state in ['visible', 'hover'] and not sphere.is_in_chain_reaction:
                dist = np.linalg.norm(sphere.position - origin_sphere.position)
                if dist < self.chain_reaction_radius:
                    sphere.is_in_chain_reaction = True
                    delay = self.chain_reaction_delay * (depth + 1) * (dist / self.chain_reaction_radius)
                    self.chain_reaction_queue.append((sphere, time.time() + delay, depth + 1, audio_system))

    # =============================================================================
    # 新ジェスチャー効果の実装
    # =============================================================================
    
    def summon_spheres_from_palm(self, hand_position, openness, audio_system=None):
        """手のひら召喚：手の位置に新しい球体を生成"""
        spawn_count = int(openness * 8) + 2  # 2-10個の球体を生成
        
        # 最大球体数の制限をチェック
        max_spheres = self.config.get('art', 'sphere_count', default=250)
        current_count = len(self.spheres)
        if current_count + spawn_count > max_spheres:
            spawn_count = max_spheres - current_count
            if spawn_count <= 0:
                print("球体数が上限に達しています。古い球体が消失してから再試行してください。")
                return

        print(f"手のひら召喚発動！ {spawn_count}個の球体を生成")
        
        for i in range(spawn_count):
            # 手の周りにランダムに配置
            offset = np.random.normal(0, 1.5, 3)  # 手の周囲1.5単位の範囲
            new_position = hand_position + offset
            
            # 画面外に出ないように制限
            new_position[0] = np.clip(new_position[0], -self.world_width/2, self.world_width/2)
            new_position[1] = np.clip(new_position[1], -self.world_height/2, self.world_height/2)
            new_position[2] = np.clip(new_position[2], -7.5, 7.5)
            
            # 球体のサイズと色を決定
            size = random.uniform(0.15, 0.35)
            # 手のひらの開き具合で色を決定（開くほど明るく）
            hue = random.random()
            saturation = 0.3 + openness * 0.4
            brightness = 0.8 + openness * 0.2
            color = self._hsv_to_rgb(hue, saturation, brightness)
            
            # 新しい球体を作成
            texture_paths = self.config.get('art', 'texture_paths', default=[])
            normal_map_paths = self.config.get('art', 'normal_map_paths', default=[])
            texture_id = random.randint(0, len(texture_paths) - 1) if texture_paths else 0
            normal_id = random.randint(0, len(normal_map_paths) - 1) if normal_map_paths else 0
            
            new_sphere = DisappearingSphere(new_position, size, color, self.config, texture_id, normal_id)
            new_sphere.state = 'appearing'
            new_sphere.alpha = 0.0
            
            # 軽い初期速度を与える
            new_sphere.velocity = np.random.normal(0, 0.5, 3)
            
            self.spheres.append(new_sphere)
            self.dynamic_material_system.initialize_sphere(id(new_sphere))
            
            # パーティクル効果
            if self.particle_manager:
                self.particle_manager.emit(new_position, color * 1.5, count=30)
        
        # 音響効果
        if audio_system:
            audio_system.play_gesture_success()

    def apply_finger_effect(self, finger_count, hand_balance='none', combination_type='none', audio_system=None):
        """指の数による効果を適用（10本指対応版）"""
        visible_spheres = [s for s in self.spheres if s.state in ['visible', 'hover']]
    
        if not visible_spheres:
            return
        
        print(f"指数制御発動！ {finger_count}本の指で'{self._get_effect_name(finger_count)}'効果")
        print(f"手のバランス: {hand_balance}, 組み合わせ: {combination_type}")
    
        # 既存の1-5本の効果
        if finger_count == 1:
            self._precision_pop(visible_spheres, audio_system)
        elif finger_count == 2:
            self._small_explosion(visible_spheres, audio_system)
        elif finger_count == 3:
            self._chain_reaction_effect(visible_spheres, audio_system)
        elif finger_count == 4:
            self._time_slow_effect(visible_spheres, audio_system)
        elif finger_count == 5:
            self._screen_reset_effect(visible_spheres, audio_system)
    
        # 新規の6-10本の効果
        elif finger_count == 6:
            self._color_shift_effect(visible_spheres, hand_balance, audio_system)
        elif finger_count == 7:
            self._gravity_reverse_effect(visible_spheres, combination_type, audio_system)
        elif finger_count == 8:
            self._sphere_split_effect(visible_spheres, hand_balance, audio_system)
        elif finger_count == 9:
            self._space_warp_effect(visible_spheres, combination_type, audio_system)
        elif finger_count >= 10:
            self._genesis_effect(visible_spheres, hand_balance, audio_system)

    def _get_effect_name(self, finger_count):
        """効果名を取得（拡張版）"""
        names = {
            1: "精密消去",
            2: "小爆発", 
            3: "チェーン反応",
            4: "時間減速",
            5: "全画面リセット",
            6: "色彩変化",      # 新規
            7: "重力反転",      # 新規
            8: "球体分裂",      # 新規
            9: "時空歪曲",      # 新規
            10: "創世効果"      # 新規
        }
        return names.get(finger_count, "未知の効果")

    def _precision_pop(self, visible_spheres, audio_system=None):
        """精密消去：手の位置から最も近い球体を消去"""
        if not visible_spheres:
            return
        
        # 手の位置を取得（指先の平均位置を使用）
        hand_position = self._get_current_hand_center()
        if hand_position is None:
            # 手が検出されない場合は中央基準にフォールバック
            hand_position = np.array([0, 0, 0], dtype=np.float32)
        
        # 手から一定範囲内の球体のみを対象
        max_distance = 3.0  # 効果範囲を3.0単位に制限
        nearby_spheres = [
            s for s in visible_spheres 
            if np.linalg.norm(s.position - hand_position) <= max_distance
        ]
        
        if not nearby_spheres:
            print("手の近くに球体がありません")
            return
        
        # 手から最も近い球体を探す
        closest_sphere = min(nearby_spheres, key=lambda s: np.linalg.norm(s.position - hand_position))
        
        if closest_sphere.handle_touch(audio_system):
            self.touch_count += 1
            if self.particle_manager:
                self.particle_manager.emit(closest_sphere.position, closest_sphere.color * 2.0, count=100)
            print(f"精密消去: 手から{np.linalg.norm(closest_sphere.position - hand_position):.1f}単位の球体を消去")

    def _small_explosion(self, visible_spheres, audio_system=None):
        """小爆発：手の周囲の球体を2-4個消去"""
        # 手の位置を取得
        hand_position = self._get_current_hand_center()
        if hand_position is None:
            hand_position = np.array([0, 0, 0], dtype=np.float32)
        
        # 手から一定範囲内の球体のみを対象
        explosion_radius = 4.0  # 爆発範囲を4.0単位に制限
        nearby_spheres = [
            s for s in visible_spheres 
            if np.linalg.norm(s.position - hand_position) <= explosion_radius
        ]
        
        if not nearby_spheres:
            print("手の近くに球体がありません")
            return
        
        # 範囲内から2-4個をランダム選択
        target_count = min(random.randint(2, 4), len(nearby_spheres))
        targets = random.sample(nearby_spheres, target_count)
        
        print(f"小爆発: 手から{explosion_radius}単位範囲内の{len(targets)}個を消去")
        
        for sphere in targets:
            if sphere.handle_touch(audio_system):
                self.touch_count += 1
                if self.particle_manager:
                    self.particle_manager.emit(sphere.position, sphere.color * 1.8, count=80)
        
        # 爆発音
        if audio_system:
            audio_system.play_explosion(0.6)

    def _chain_reaction_effect(self, visible_spheres, audio_system=None):
        """チェーンリアクション：ランダムな起点から連鎖開始"""
        if not visible_spheres:
            return
            
        origin_sphere = random.choice(visible_spheres)
        if origin_sphere.handle_touch(audio_system):
            self.touch_count += 1
            if self.particle_manager:
                self.particle_manager.emit(origin_sphere.position, origin_sphere.color * 2.0, count=150)
            self._trigger_chain_reaction(origin_sphere, 1, audio_system)

    def _time_slow_effect(self, visible_spheres, audio_system=None):
        """時間減速：全ての球体の動きを一時的に遅くする"""
        for sphere in visible_spheres:
            # 回転速度を50%に減速
            sphere.rotation_speed *= 0.5
            # 速度も減速
            sphere.velocity *= 0.3
            # 特殊な色効果（青みがかった色に）
            # sphere.color = sphere.color * np.array([0.7, 0.8, 1.2], dtype=np.float32)
            # sphere.dynamic_emission = 0.3
        
        # 音響効果
        if audio_system:
            audio_system.play_gesture_success()
        
        print("時間減速効果発動！球体の動きが遅くなりました")

    def _screen_reset_effect(self, visible_spheres, audio_system=None):
        """全画面リセット：虹色エフェクトと共に全球体をリセット"""
        print("全画面リセット！虹色エフェクト発動")
        
        # 全球体に虹色エフェクト
        for i, sphere in enumerate(visible_spheres):
            # 虹色グラデーション
            hue = (i / len(visible_spheres)) % 1.0
            rainbow_color = self._hsv_to_rgb(hue, 0.9, 1.0)
            
            # 大量のパーティクル放出
            if self.particle_manager:
                self.particle_manager.emit(sphere.position, rainbow_color, count=200)
        
        # 全球体をリセット
        self.reset()
        
        # 特殊音響効果
        if audio_system:
            audio_system.play_explosion(1.0)
            audio_system.play_gesture_success()

    def _color_shift_effect(self, visible_spheres, hand_balance, audio_system=None):
        """6本指: 色彩変化エフェクト"""
        print("色彩変化発動！球体が虹色に変化")
    
        for i, sphere in enumerate(visible_spheres):
            # 手のバランスによって色の変化パターンを変更
            if hand_balance == 'balanced':
                # 対称的な虹色
                hue = (i / len(visible_spheres)) % 1.0
            elif hand_balance == 'left_heavy':
                # 左寄りの色（寒色系）
                hue = 0.5 + (i / len(visible_spheres)) * 0.3
            elif hand_balance == 'right_heavy':
                # 右寄りの色（暖色系）
                hue = (i / len(visible_spheres)) * 0.3
            else:
                # デフォルト虹色
                hue = (i / len(visible_spheres)) % 1.0
        
            new_color = self._hsv_to_rgb(hue, 0.8, 1.0)
            sphere.color = new_color
            sphere.dynamic_emission = 0.5
        
            # パーティクル効果
            if self.particle_manager:
                self.particle_manager.emit(sphere.position, new_color * 1.5, count=50)
    
        if audio_system:
            audio_system.play_gesture_success()

    def _gravity_reverse_effect(self, visible_spheres, combination_type, audio_system=None):
        """7本指: 重力反転エフェクト（修正版）"""
        print("重力反転！球体が上昇します（一時的効果）")
    
        # 組み合わせパターンによって上昇力を変更
        if 'symmetric' in combination_type:
            upward_force = 8.0
        elif 'single_hand' in combination_type:
            upward_force = 5.0
        else:
            upward_force = 6.0
    
        for sphere in visible_spheres:
            # 上向きの速度を付与（制限付き）
            sphere.velocity[1] = min(upward_force, sphere.velocity[1] + upward_force)  # 🔧 最大速度制限
        
            # 横方向の速度は維持しつつ、縦方向のみ変更
            sphere.velocity[0] *= 0.95  # 横方向の速度を少し減衰
            sphere.velocity[2] *= 0.95  # 奥行き方向の速度を少し減衰
        
            # エフェクト用の色変化（一時的）
            sphere.dynamic_emission = 0.3
        
            # 🔧 修正点6: 重力反転効果の持続時間制限
            sphere.gravity_reverse_time = time.time()  # 効果開始時刻を記録
        
            if self.particle_manager:
                upward_particles = sphere.position + np.array([0, 1.0, 0])
                self.particle_manager.emit(upward_particles, sphere.color * 1.2, count=30)
    
        if audio_system:
            audio_system.play_explosion(0.4)

    def _sphere_split_effect(self, visible_spheres, hand_balance, audio_system=None):
        """8本指: 球体分裂エフェクト"""
        print("球体分裂！各球体が2つに分裂")
    
        new_spheres = []
        max_spheres = self.config.get('art', 'sphere_count', default=250)
    
        for sphere in visible_spheres[:max_spheres//2]:  # 最大数を超えないよう制限
            if len(self.spheres) + len(new_spheres) >= max_spheres:
                break
            
            # 元の球体のサイズを小さくする
            sphere.radius *= 0.8
            sphere.current_radius *= 0.8
        
            # 新しい球体を作成
            offset_distance = sphere.radius * 2.5
        
            if hand_balance == 'balanced':
                # 左右に分裂
                offset = np.array([offset_distance, 0, 0])
            elif hand_balance == 'left_heavy':
                # 左上に分裂
                offset = np.array([-offset_distance, offset_distance, 0])
            elif hand_balance == 'right_heavy':
                # 右上に分裂
                offset = np.array([offset_distance, offset_distance, 0])
            else:
                # ランダム方向に分裂
                angle = random.uniform(0, 2 * np.pi)
                offset = np.array([np.cos(angle), np.sin(angle), 0]) * offset_distance
        
            new_position = sphere.position + offset
            new_sphere = DisappearingSphere(
                new_position, sphere.radius, sphere.color.copy(), 
                self.config, sphere.texture_id, sphere.normal_map_id
            )
            new_sphere.state = 'appearing'
            new_sphere.alpha = 0.0
            new_sphere.velocity = offset * 0.3  # 分裂方向に初期速度
        
            new_spheres.append(new_sphere)
        
            # パーティクル効果
            if self.particle_manager:
                self.particle_manager.emit(sphere.position, sphere.color * 2.0, count=100)
    
        # 新しい球体をリストに追加
        self.spheres.extend(new_spheres)
        for sphere in new_spheres:
            self.dynamic_material_system.initialize_sphere(id(sphere))
    
        print(f"{len(new_spheres)}個の新しい球体が分裂により生成されました")
    
        if audio_system:
            audio_system.play_explosion(0.7)

    def _space_warp_effect(self, visible_spheres, combination_type, audio_system=None):
        """9本指: 時空歪曲エフェクト"""
        print("時空歪曲！球体が螺旋状に移動")
    
        # 中心点を計算
        if visible_spheres:
            center = np.mean([s.position for s in visible_spheres], axis=0)
        else:
            center = np.array([0, 0, 0])
    
        # 組み合わせによって回転の方向と速度を決定
        if 'symmetric' in combination_type:
            rotation_speed = 2.0
            spiral_direction = 1  # 時計回り
        elif 'left_heavy' in combination_type:
            rotation_speed = 3.0
            spiral_direction = -1  # 反時計回り
        else:
            rotation_speed = 2.5
            spiral_direction = 1
    
        current_time = time.time()
    
        for i, sphere in enumerate(visible_spheres):
            # 中心からの距離と角度を計算
            relative_pos = sphere.position - center
            distance = np.linalg.norm(relative_pos[:2])  # XY平面での距離
        
            if distance > 0.1:  # 中心に近すぎる場合は除外
                # 螺旋運動の計算
                angle_offset = (i / len(visible_spheres)) * 2 * np.pi
                spiral_angle = current_time * rotation_speed * spiral_direction + angle_offset
            
                # 新しい位置を計算
                new_x = center[0] + distance * np.cos(spiral_angle) * 1.1
                new_y = center[1] + distance * np.sin(spiral_angle) * 1.1
                new_z = sphere.position[2] + np.sin(spiral_angle * 2) * 0.5
            
                # 滑らかに移動させる
                target_pos = np.array([new_x, new_y, new_z])
                sphere.velocity += (target_pos - sphere.position) * 0.3
            
                # 回転速度も増加
                sphere.rotation_speed *= 1.5
            
                # 視覚効果
                sphere.dynamic_emission = 0.4
                sphere.dynamic_color_multiplier = 1.3
    
        if audio_system:
            audio_system.play_gesture_success()

    def _genesis_effect(self, visible_spheres, hand_balance, audio_system=None):
        """10本指: 創世エフェクト"""
        print("創世効果！新たな宇宙の創造")
    
        # 既存の球体をリセット
        self.reset()
    
        # 大量の新球体を生成
        genesis_count = 100  # 通常より多く生成
        max_spheres = self.config.get('art', 'sphere_count', default=250)
        genesis_count = min(genesis_count, max_spheres - len(self.spheres))
    
        # 手のバランスによって生成パターンを変更
        if hand_balance == 'balanced':
            # 中央から放射状に生成
            center = np.array([0, 0, 0])
            pattern = 'radial'
        elif hand_balance == 'left_heavy':
            # 左側から右側へ流れるように生成
            center = np.array([-self.world_width/3, 0, 0])
            pattern = 'flow_right'
        elif hand_balance == 'right_heavy':
            # 右側から左側へ流れるように生成
            center = np.array([self.world_width/3, 0, 0])
            pattern = 'flow_left'
        else:
            # 複数の点から生成
            center = np.array([0, 0, 0])
            pattern = 'multi_point'
    
        for i in range(genesis_count):
            if pattern == 'radial':
                # 放射状配置
                angle = (i / genesis_count) * 2 * np.pi
                radius = (i % 20) * 0.5
                pos = center + np.array([
                    np.cos(angle) * radius,
                    np.sin(angle) * radius,
                    random.uniform(-2, 2)
                ])
            elif pattern == 'flow_right':
                # 右向きの流れ
                pos = center + np.array([
                    i * 0.3,
                    random.uniform(-self.world_height/2, self.world_height/2),
                    random.uniform(-5, 5)
                ])
            elif pattern == 'flow_left':
                # 左向きの流れ
                pos = center + np.array([
                    -i * 0.3,
                    random.uniform(-self.world_height/2, self.world_height/2),
                    random.uniform(-5, 5)
                ])
            else:
                # マルチポイント
                cluster_center = np.array([
                    random.uniform(-self.world_width/2, self.world_width/2),
                    random.uniform(-self.world_height/2, self.world_height/2),
                    random.uniform(-5, 5)
                ])
                pos = cluster_center + np.random.normal(0, 1, 3)
        
            # 球体の属性
            size = random.uniform(0.1, 0.3)
            hue = (i / genesis_count) % 1.0
            color = self._hsv_to_rgb(hue, 0.7, 1.0)
        
            # テクスチャID
            texture_paths = self.config.get('art', 'texture_paths', default=[])
            normal_map_paths = self.config.get('art', 'normal_map_paths', default=[])
            texture_id = random.randint(0, len(texture_paths) - 1) if texture_paths else 0
            normal_id = random.randint(0, len(normal_map_paths) - 1) if normal_map_paths else 0
        
            # 新球体作成
            new_sphere = DisappearingSphere(pos, size, color, self.config, texture_id, normal_id)
            new_sphere.state = 'appearing'
            new_sphere.alpha = 0.0
            new_sphere.dynamic_emission = 1.0
        
            # 初期速度（創世の爆発的拡散）
            explosion_velocity = (pos - center) * 0.1
            new_sphere.velocity = explosion_velocity
        
            self.spheres.append(new_sphere)
            self.dynamic_material_system.initialize_sphere(id(new_sphere))
        
            # パーティクル効果
            if self.particle_manager and i % 5 == 0:  # 5個おきにパーティクル
                self.particle_manager.emit(pos, color * 3.0, count=200)
    
        print(f"創世完了！{genesis_count}個の新しい星が誕生しました")
    
        # 特殊音響効果
        if audio_system:
            audio_system.play_explosion(1.0)
            audio_system.play_gesture_success()

    def update(self, dt, current_time, fingertip_positions, gravity_well, gesture_states=None, audio_system=None):
        """メイン更新ループ（新ジェスチャー対応）"""
        
        # 指先位置を保存（効果計算で使用）
        self._current_fingertips = fingertip_positions
        
        # チェーンリアクションの処理
        while self.chain_reaction_queue and self.chain_reaction_queue[0][1] <= current_time:
            sphere, _, depth, chain_audio = self.chain_reaction_queue.popleft()
            if sphere.handle_touch(chain_audio):
                self.particle_manager.emit(sphere.position, sphere.color, count=50)
                self._trigger_chain_reaction(sphere, depth, chain_audio)
                if chain_audio:
                    chain_audio.play_chain_reaction(depth)

        # 新ジェスチャーの処理
        if gesture_states and audio_system:
            self._handle_new_gestures(gesture_states, audio_system)

        # 既存の球体更新処理
        for sphere in self.spheres:
            sphere.update(dt, current_time)

            if gravity_well.is_active and sphere.state != 'disappearing':
                gravity_well.apply_force_to_sphere(sphere, dt)

            is_hovering = False
            if sphere.state not in ['disappearing', 'invisible']:
                for tip_pos in fingertip_positions:
                    dist = np.linalg.norm(tip_pos - sphere.position)

                    if dist < sphere.current_radius:
                        if sphere.handle_touch(audio_system):
                            self.touch_count += 1
                            self.particle_manager.emit(sphere.position, sphere.color, count=150)
                            self._trigger_chain_reaction(sphere, 1, audio_system)
                        is_hovering = False
                        break

                    elif dist < self.hover_distance:
                        is_hovering = True

            if sphere.state not in ['disappearing', 'invisible', 'appearing']:
                sphere.state = 'hover' if is_hovering else 'visible'

            self.dynamic_material_system.update_sphere_material(sphere, current_time)

        for i, indicator in enumerate(self.hand_indicators):
            target_pos = fingertip_positions[i] if i < len(fingertip_positions) else None
            indicator.update(dt, current_time, target_pos)

    def _handle_new_gestures(self, gesture_states, audio_system):
        """新ジェスチャーの処理（拡張版）"""
        # 手のひら召喚の処理（既存）
        palm_state = gesture_states.get('palm_open', {})
        for hand_name in ['left', 'right']:
            hand_data = palm_state.get(hand_name, {})
            if hand_data.get('ready_to_summon', False):
                position = hand_data.get('position')
                openness = hand_data.get('openness', 0.0)
                if position is not None:
                    self.summon_spheres_from_palm(position, openness, audio_system)
                    if hasattr(audio_system, 'gesture_system'):
                        audio_system.gesture_system.trigger_cooldown(f'palm_summon_{hand_name}')

        # 拡張された指数制御の処理
        finger_state = gesture_states.get('finger_count', {})
        if finger_state.get('effect_ready', False):
            finger_count = finger_state.get('total', 0)
            hand_balance = finger_state.get('hand_balance', 'none')
            combination_type = finger_state.get('combination_type', 'none')
        
            if finger_count > 0:
                self.apply_finger_effect(finger_count, hand_balance, combination_type, audio_system)
                if hasattr(audio_system, 'gesture_system'):
                    # より高い指数の場合はクールダウンを長くする
                    cooldown_time = 1.0 + (finger_count - 1) * 0.2
                    audio_system.gesture_system.trigger_cooldown('finger_effect', cooldown_time)

    def _get_current_hand_center(self):
        """現在の手の中心位置を取得"""
        try:
            # 指先位置から手の中心を推定
            if hasattr(self, '_current_fingertips') and self._current_fingertips:
                # 指先の平均位置を手の中心とする
                positions = np.array(self._current_fingertips)
                return np.mean(positions, axis=0)
            return None
        except:
            return None

    def reset(self):
        """全ての球体を完全にリセット（修正版）"""
        for sphere in self.spheres:
            # 基本状態のリセット
            sphere.state = 'appearing'
            sphere.alpha = 0.0
            sphere.is_in_chain_reaction = False
        
            # 物理状態の完全リセット
            sphere.velocity = np.zeros(3, dtype=np.float32)
            sphere.position = sphere.original_position.copy()  # 🔧 修正点1: 位置を元に戻す
        
            # 回転状態のリセット
            sphere.rotation_speed = np.array([random.uniform(-0.1, 0.1) for _ in range(3)], dtype=np.float32)
            sphere.rotation = np.array([random.uniform(0, 2 * PI), random.uniform(0, 2 * PI), random.uniform(0, 2 * PI)], dtype=np.float32)
        
            # サイズのリセット
            sphere.current_radius = sphere.radius  # 🔧 修正点2: サイズを元に戻す
        
            # 動的材質プロパティのリセット
            sphere.dynamic_roughness = 0.5       # 🔧 修正点3: 材質を初期値に戻す
            sphere.dynamic_metallic = 0.1
            sphere.dynamic_emission = 0.0
            sphere.dynamic_color_multiplier = 1.0
        
            # 色のリセット（重力反転などで変更された色を元に戻す）
            # 元の色が保存されていない場合の対処
            if not hasattr(sphere, 'original_color'):
                # HSVで新しい色を生成（元のロジックと同じ）
                hue = random.random()
                saturation = 0.1
                brightness = 1.0
                sphere.original_color = self._hsv_to_rgb(hue, saturation, brightness)
        
            sphere.color = sphere.original_color.copy()  # 🔧 修正点4: 色を元に戻す
    
        # システム状態のリセット
        self.touch_count = 0
        self.chain_reaction_queue.clear()
    
        print("全ての球体を完全にリセットしました。")

    def force_reset_all_physics(self):
        """物理状態を強制的にリセット（緊急用）"""
        print("物理状態を強制リセット中...")
    
        for sphere in self.spheres:
            # 位置の強制リセット
            sphere.position = sphere.original_position.copy()
        
            # 物理状態の強制クリア
            sphere.velocity = np.zeros(3, dtype=np.float32)
        
            # 重力反転効果の強制解除
            if hasattr(sphere, 'gravity_reverse_time'):
                delattr(sphere, 'gravity_reverse_time')
        
            # 動的プロパティの強制リセット
            sphere.dynamic_emission = 0.0
            sphere.dynamic_roughness = 0.5
            sphere.dynamic_metallic = 0.1
            sphere.dynamic_color_multiplier = 1.0
        
            # 色の強制リセット
            if hasattr(sphere, 'original_color'):
                sphere.color = sphere.original_color.copy()
    
        print("物理状態の強制リセット完了")

# =============================================================================
# 7. 修正版BodyTrackerクラス
# =============================================================================
class BodyTracker:
    def __init__(self, config, world_width, world_height, frame_shape):
        self.config, self.world_width, self.world_height = config, world_width, world_height
        self.z_scale = self.config.get('hand_tracking', 'z_scale', default=30.0)
        self.z_offset = self.config.get('hand_tracking', 'z_offset', default=0.5)
        self.fingertip_positions = []
        self.left_hand_detected, self.right_hand_detected = False, False
        
        # 新しいシステムに置き換え
        self.face_detector = EnhancedFaceDirectionDetector(config, frame_shape)
        self.gesture_system = EnhancedGestureSystem(config)
        
        self.fingertip_indices = self.config.get('mediapipe', 'fingertip_indices', default=[4, 8, 12, 16, 20])
        self.thumb_tip_idx = self.config.get('mediapipe', 'thumb_tip')
        self.index_tip_idx = self.config.get('mediapipe', 'index_tip')
        
        # 従来の重力ジェスチャー変数（後方互換性）
        self.gravity_gesture_active = False
        self.gravity_gesture_center = None
        
        self.left_hand_landmarks = None
        self.right_hand_landmarks = None
        print(f"3Dワールド空間のサイズを初期化: Width={self.world_width:.2f}, Height={self.world_height:.2f}")

    def update_from_landmarks(self, left_hand, right_hand, face_landmarks):
        self.left_hand_landmarks = left_hand
        self.right_hand_landmarks = right_hand
        self.left_hand_detected, self.right_hand_detected = left_hand is not None, right_hand is not None
        self.fingertip_positions.clear()
        
        tip_landmarks = []
        if self.left_hand_landmarks:
            tip_landmarks.extend([self.left_hand_landmarks.landmark[i] for i in self.fingertip_indices])
        if self.right_hand_landmarks:
            tip_landmarks.extend([self.right_hand_landmarks.landmark[i] for i in self.fingertip_indices])

        self.fingertip_positions = [self._landmark_to_world(lm) for lm in tip_landmarks]
        
        # 強化された顔検出システムを更新
        self.face_detector.update(face_landmarks)
        
        # ジェスチャーシステムを更新
        self.gesture_system.update(left_hand, right_hand)
        
        # 従来の重力ジェスチャーとの互換性
        gravity_state = self.gesture_system.get_gesture_state('gravity_well')
        self.gravity_gesture_active = gravity_state['active']
        self.gravity_gesture_center = gravity_state['position']

    def _landmark_to_world(self, landmark):
        if not landmark: return None
        x = (landmark.x - 0.5) * self.world_width
        y = (0.5 - landmark.y) * self.world_height
        z = (landmark.z * self.z_scale) + self.z_offset
        return np.array([x, y, z], dtype=np.float32)

    # 新しいアクセスメソッド
    def get_face_angles(self):
        """詳細な顔の角度情報を取得"""
        return self.face_detector.get_angles()

    def get_face_direction_vector(self):
        """3D顔向きベクトルを取得"""
        return self.face_detector.get_direction_vector()

    def get_normalized_face_direction(self):
        """正規化された顔向きを取得"""
        return self.face_detector.get_normalized_direction()

    def get_gesture_states(self):
        """全ジェスチャーの状態を取得"""
        return self.gesture_system.gesture_states

    def is_gesture_active(self, gesture_name):
        """指定ジェスチャーがアクティブかを確認"""
        return self.gesture_system.is_gesture_active(gesture_name)

    # 従来のメソッド（後方互換性）
    def get_current_fingertip_positions(self): return self.fingertip_positions
    def get_face_direction(self): return self.face_detector.get_direction()
    def is_any_hand_detected(self): return self.left_hand_detected or self.right_hand_detected
    def get_average_fingertip_z(self):
        if not self.fingertip_positions: return None
        all_z = [pos[2] for pos in self.fingertip_positions]
        return np.mean(all_z) if all_z else None

# =============================================================================
# 8. その他の必要なクラス（元コードから継承）
# =============================================================================

class Camera:
    def __init__(self, config, screen_width, screen_height, world_width, world_height):
        self.config, self.world_width, self.world_height = config, world_width, world_height
        self.eye = np.array(self.config.get('camera', 'eye'), dtype=np.float32)
        self.target = np.array(self.config.get('camera', 'target'), dtype=np.float32)
        self.up = np.array(self.config.get('camera', 'up'), dtype=np.float32)
        self.near, self.far = self.config.get('camera', 'near_plane'), self.config.get('camera', 'far_plane')
        self.update_aspect_ratio(screen_width, screen_height)

    def update_aspect_ratio(self, screen_width, screen_height):
        self.aspect_ratio = screen_width / screen_height

    def get_view_matrix(self):
        return self._look_at(self.eye, self.target, self.up)

    def get_projection_matrix(self):
        proj_type = self.config.get('camera', 'projection_type', default='perspective')
        if proj_type == 'orthographic':
            return self._orthographic(-self.world_width / 2.0, self.world_width / 2.0,
                                      -self.world_height / 2.0, self.world_height / 2.0, self.near, self.far)
        else:
            return self._perspective(self.config.get('camera', 'fovy'), self.aspect_ratio, self.near, self.far)

    def _look_at(self, eye, target, up):
        f = target - eye; f_norm = np.linalg.norm(f)
        if f_norm == 0: return np.eye(4, dtype='f4')
        f /= f_norm; s = np.cross(f, up); s_norm = np.linalg.norm(s)
        if s_norm == 0: return np.eye(4, dtype='f4')
        s /= s_norm; u = np.cross(s, f); m = np.eye(4, dtype='f4')
        m[0, :3], m[1, :3], m[2, :3] = s, u, -f
        m[:3, 3] = -s @ eye, -u @ eye, f @ eye
        return m.astype('f4')

    def _perspective(self, fovy, aspect, near, far):
        f = 1.0 / np.tan(np.radians(fovy) / 2.0); m = np.zeros((4, 4), dtype='f4')
        m[0, 0], m[1, 1] = f / aspect, f
        m[2, 2], m[3, 2] = (far + near) / (near - far), -1.0
        m[2, 3] = (2.0 * far * near) / (near - far)
        return m

    def _orthographic(self, l, r, b, t, n, f):
        m = np.eye(4, dtype='f4')
        m[0,0], m[1,1], m[2,2] = 2/(r-l), 2/(t-b), -2/(f-n)
        m[0,3], m[1,3], m[2,3] = -(r+l)/(r-l), -(t+b)/(t-b), -(f+n)/(f-n)
        return m

class DebugInfo:
    def __init__(self, config):
        self.config, self.font = config, cv2.FONT_HERSHEY_SIMPLEX
        self.last_update, self.update_interval = 0, 0.25
        self.cached_texts, self.show_debug = {}, True
        self.show_depth_indicator = self.config.get('ui', 'show_depth_indicator', default=True)

    def _draw_text_with_bg(self, frame, text, pos, scale, color, thick=1):
        x, y = pos; (w, h), base = cv2.getTextSize(text, self.font, scale, thick)
        p = 5; start, end = (x - p, y), (x + w + p, y + h + base + p)
        x1, y1, x2, y2 = max(0, start[0]), max(0, start[1]), min(frame.shape[1], end[0]), min(frame.shape[0], end[1])
        if x1 >= x2 or y1 >= y2: return 0
        sub = frame[y1:y2, x1:x2]
        frame[y1:y2, x1:x2] = cv2.addWeighted(sub, 0.5, np.full(sub.shape, (0,0,0), np.uint8), 0.5, 1.0)
        cv2.putText(frame, text, (x, y + h + int(base/2)), self.font, scale, color, thick, cv2.LINE_AA)
        return h + base + p * 2

    def _draw_depth_indicator(self, frame, tracker):
        if not self.show_depth_indicator or not (tracker.left_hand_landmarks or tracker.right_hand_landmarks): return
        h, w, _ = frame.shape; bar_w, bar_h = 15, h // 2
        bar_x, bar_y = w - bar_w - 20, h // 4
        z_offset = tracker.config.get('hand_tracking', 'z_offset', default=0.5)
        z_min, z_max = z_offset, z_offset - tracker.config.get('hand_tracking', 'world_depth', default=15.0)
        cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h), (255, 255, 255), 1)
        
        all_z = []
        if tracker.left_hand_landmarks: all_z.extend([lm.z for lm in tracker.left_hand_landmarks.landmark])
        if tracker.right_hand_landmarks: all_z.extend([lm.z for lm in tracker.right_hand_landmarks.landmark])
        if not all_z: return
        
        avg_z_raw = np.mean(all_z)
        avg_z_world = (avg_z_raw * tracker.config.get('hand_tracking', 'z_scale')) + z_offset

        if avg_z_world is not None:
            norm_z = np.clip((avg_z_world - z_min) / (z_max - z_min), 0, 1)
            marker_y = int(bar_y + norm_z * bar_h)
            cv2.line(frame, (bar_x - 5, marker_y), (bar_x + bar_w + 5, marker_y), (0, 255, 255), 2)
            cv2.putText(frame, f"{avg_z_world:.1f}", (bar_x - 50, marker_y + 5), self.font, 0.5, (0, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(frame, "Depth", (bar_x - 15, bar_y - 10), self.font, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

    def update_and_draw(self, frame, fps, sphere_manager, tracker):
        help_text = "F11=Fullscreen | D=Debug | T=Reset | M=Audio"
        self._draw_depth_indicator(frame, tracker)
        if not self.show_debug:
            cv2.putText(frame, help_text, (10, frame.shape[0] - 15), self.font, 0.6, (255,255,255), 1, cv2.LINE_AA)
            return frame
        if time.time() - self.last_update >= self.update_interval:
            visible = len([s for s in sphere_manager.spheres if s.state in ['visible', 'hover']])
            
            # 新ジェスチャー情報を取得
            gesture_states = tracker.get_gesture_states()
            palm_state = gesture_states.get('palm_open', {})
            finger_state = gesture_states.get('finger_count', {})
            
            # 手のひら状態
            palm_info = []
            for hand in ['left', 'right']:
                hand_data = palm_state.get(hand, {})
                if hand_data.get('open', False):
                    openness = hand_data.get('openness', 0.0)
                    palm_info.append(f"{hand[0].upper()}:{openness:.1f}")
            palm_text = f"Palm: {'/'.join(palm_info) if palm_info else 'Closed'}"
            
            # 指の数
            finger_text = f"Fingers: L{finger_state.get('left', 0)} R{finger_state.get('right', 0)} (Total:{finger_state.get('total', 0)})"
            
            # 顔の角度情報
            face_angles = tracker.get_face_angles()
            face_text = f"Face: Y{face_angles['yaw']:.1f}° P{face_angles['pitch']:.1f}° (C:{face_angles['confidence']:.2f})"
            
            self.cached_texts = {
                'fps': f"FPS: {fps:.1f}",
                'spheres': f"Stars: {visible}/{len(sphere_manager.spheres)} | Touched: {sphere_manager.touch_count}",
                'face': face_text,
                'gestures': f"{palm_text} | {finger_text}",
                'hands': f"Hands: L[{'OK' if tracker.left_hand_landmarks else '---'}] R[{'OK' if tracker.right_hand_landmarks else '---'}]",
                'gravity': f"Gravity Well: {'ACTIVE' if tracker.gravity_gesture_active else 'OFF'}",
                'finger_text': f"Fingers: L{finger_state.get('left', 0)} R{finger_state.get('right', 0)} Total:{finger_state.get('total', 0)} [{finger_state.get('hand_balance', 'none')}]",
                'combo_text': f"Combo: {finger_state.get('combination_type', 'none')}"
            }
            self.last_update = time.time()
        y, p = 10, 10
        lines = [ (self.cached_texts.get(k), s, c, t) for k,s,c,t in [
            ('fps', 0.7, (0,255,128), 2), ('spheres', 0.6, (255,255,180), 1),
            ('face', 0.5, (180,255,255), 1), ('gestures', 0.5, (255,180,255), 1),
            ('hands', 0.6, (255,180,255), 1), ('gravity', 0.6, (255, 255, 0), 1)
            ]]
        for text, scale, color, thick in lines:
            if text: y += self._draw_text_with_bg(frame, text, (p, y), scale, color, thick)
        cv2.putText(frame, help_text, (10, frame.shape[0] - 15), self.font, 0.6, (255,255,255), 1, cv2.LINE_AA)
        return frame

    def toggle(self):
        self.show_debug = not self.show_debug
        print(f"デバッグ情報表示: {'ON' if self.show_debug else 'OFF'}")

class GravityWell:
    def __init__(self, config):
        self.config = config.get_dict('advanced_physics')
        self.strength = self.config.get('gravity_well_strength')
        self.radius = self.config.get('gravity_well_radius')
        self.position = None
        self.is_active = False
        self.current_strength = 0.0

    def activate(self, position):
        self.position = np.array(position, dtype=np.float32)
        self.is_active = True

    def deactivate(self):
        self.is_active = False

    def update(self, dt):
        target_strength = self.strength if self.is_active else 0.0
        self.current_strength += (target_strength - self.current_strength) * dt * 5.0

    def apply_force_to_sphere(self, sphere, dt):
        if self.current_strength < 0.01: return

        diff = self.position - sphere.position
        dist_sq = np.dot(diff, diff)

        if dist_sq < self.radius * self.radius and dist_sq > 0.01:
            dist = np.sqrt(dist_sq)
            force_mag = self.current_strength / (dist_sq + 1.0)
            force = (diff / dist) * force_mag
            sphere.velocity += force * dt

class Starfield:
    def __init__(self, config, ctx, camera):
        self.config, self.ctx, self.camera = config, ctx, camera
        self.star_count = self.config.get('starfield', 'star_count', default=1000)
        self.speed = self.config.get('starfield', 'star_speed', default=0.5)
        self.max_size = self.config.get('starfield', 'star_max_size', default=2.0)
        self.ww, self.wh = self.camera.world_width * 1.5, self.camera.world_height * 1.5
        self.depth_range = (self.camera.near, self.camera.far)
        self.stars = np.zeros((self.star_count, 4), dtype='f4')
        self.stars[:, 0] = np.random.uniform(-self.ww, self.ww, self.star_count)
        self.stars[:, 1] = np.random.uniform(-self.wh, self.wh, self.star_count)
        self.stars[:, 2] = np.random.uniform(*self.depth_range, self.star_count)
        self.stars[:, 3] = np.random.uniform(0.2, 1.0, self.star_count)
        self.vbo = None
        self.program, self.vao = None, None
        print(f"{self.star_count}個の星を背景に生成しました。")

    def set_program(self, program):
        self.program = program
        self.vbo = self.ctx.buffer(self.stars.tobytes())
        self.vao = self.ctx.vertex_array(self.program, [(self.vbo, '3f 1f', 'in_vert', 'in_brightness')])

    def update(self, dt):
        self.stars[:, 2] += self.speed * dt
        reset_indices = self.stars[:, 2] > self.camera.far
        if np.any(reset_indices):
            self.stars[reset_indices, 2] = self.camera.near
        if self.vbo:
            self.vbo.write(self.stars.tobytes())

    def render(self, mvp):
        if self.vao and self.program:
            self.program['mvp'].write(mvp)
            self.program['max_size'].value = self.max_size
            self.ctx.point_size = self.max_size
            self.vao.render(mode=moderngl.POINTS)

class ParticleManager:
    def __init__(self, config, ctx):
        self.config, self.ctx = config, ctx
        self.particle_count = self.config.get('particles', 'count', default=10000)
        self.lifespan = self.config.get('particles', 'lifespan', default=1.5)
        self.speed = self.config.get('particles', 'speed', default=4.0)
        self.gravity = np.array([0, self.config.get('particles', 'gravity', default=-2.0), 0], dtype='f4')
        self.emit_count = self.config.get('particles', 'emit_count', default=50)
        self.data = np.zeros((self.particle_count, 10), dtype='f4')
        self.data[:, 6] = -1.0 # Initialize as inactive
        self.vbo1 = self.ctx.buffer(self.data)
        self.vbo2 = self.ctx.buffer(reserve=self.data.nbytes)
        self.program, self.transform_program = None, None
        self.render_vao, self.transform_vao = None, None
        self.next_particle_idx = 0
        self.enabled = True
        print(f"{self.particle_count}個のパーティクルプールをGPUベースで生成しました。")

    def set_programs(self, render_program, transform_program):
        self.render_program = render_program
        self.transform_program = transform_program
        if self.transform_program is None or self.render_program is None:
            self.enabled = False
            return
        try:
            self.transform_vao = self.ctx.vertex_array(
                self.transform_program, 
                [(self.vbo1, '3f 3f 1f 3f', 'in_pos', 'in_vel', 'in_life', 'in_color')]
            )
            self.render_vao = self.ctx.vertex_array(
                self.render_program,
                [(self.vbo1, '3f 4x4 1f 3f', 'in_pos', 'in_life', 'in_color')]
            )
            self.enabled = True
            print("✓ パーティクルVAOを正常に作成しました")
        except Exception as e:
            print(f"[警告] パーティクルVAOの作成に失敗しました: {e}")
            self.enabled = False

    def emit(self, position, color, count=None):
        if not self.enabled: return
        emit_count = count if count else self.emit_count
        start = self.next_particle_idx
        end = start + emit_count
        
        velocities = np.random.normal(0.0, 1.0, (emit_count, 3)).astype('f4')
        norms = np.linalg.norm(velocities, axis=1, keepdims=True)
        velocities = (velocities / (norms + 1e-8)) * self.speed * (np.random.rand(emit_count, 1) * 0.5 + 0.5)

        new_data = np.zeros((emit_count, 10), dtype='f4')
        new_data[:, 0:3] = position
        new_data[:, 3:6] = velocities
        new_data[:, 6] = self.lifespan * (np.random.rand(emit_count) * 0.5 + 0.5)
        new_data[:, 7:10] = np.clip(color * 1.5, 0.0, 1.0)
        
        if end > self.particle_count:
            part1_count = self.particle_count - start
            self.vbo1.write(new_data[:part1_count].tobytes(), offset=start * 40)
            part2_count = end - self.particle_count
            self.vbo1.write(new_data[part1_count:].tobytes(), offset=0)
        else:
            self.vbo1.write(new_data.tobytes(), offset=start * 40)

        self.next_particle_idx = end % self.particle_count

    def update(self, dt):
        if not self.enabled or self.transform_program is None or self.transform_vao is None: return
        try:
            self.transform_program['u_gravity'].write(self.gravity)
            self.transform_program['u_dt'].value = dt
            self.ctx.enable_only(moderngl.RASTERIZER_DISCARD)
            self.transform_vao.transform(self.vbo2, moderngl.POINTS)
            self.ctx.disable(moderngl.RASTERIZER_DISCARD)
            self.vbo1, self.vbo2 = self.vbo2, self.vbo1
        except Exception as e:
            print(f"[警告] パーティクル更新でエラー: {e}")
            self.enabled = False

    def render(self, mvp):
        if not self.enabled or self.render_vao is None or self.render_program is None: return
        try:
            self.render_program['mvp'].write(mvp)
            self.render_vao.render(mode=moderngl.POINTS)
        except Exception as e:
            print(f"[警告] パーティクル描画でエラー: {e}")
            self.enabled = False

# =============================================================================
# 9. メインアプリケーションクラス（音響・新ジェスチャー対応版）
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
        
        # 音響システムを最初に初期化
        self.audio_system = AudioSystem(config)
        
        self.particle_manager = ParticleManager(config, self.ctx)
        self.gravity_well = GravityWell(config)
        self.sphere_manager = SphereManager(config, world_width, world_height, self.particle_manager)
        self.tracker = BodyTracker(config, world_width, world_height, (self.camera_height, self.camera_width))
        self.camera = Camera(config, self.screen_size[0], self.screen_size[1], world_width, world_height)
        self.starfield = Starfield(config, self.ctx, self.camera)
        
        self.debug_info = DebugInfo(config)
        self.light_direction = np.array([0.5, 1.0, 0.8], dtype='f4')
        self.light_smoothing = self.config.get('lighting', 'smoothing')

        self._load_all_shaders()
        self._load_all_textures()
        self._init_all_buffers()
        self._init_post_processing()

        self.landmarks_queue = queue.Queue(maxsize=2)
        self.video_queue = queue.Queue(maxsize=1)
        self.mediapipe_thread = threading.Thread(target=self._mediapipe_worker, daemon=True)
        self.clock = pygame.time.Clock()
        self.last_time = time.time()
        self.fps_history = deque(maxlen=60)

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
        idx = self.config.get('display', 'display_index', default=0)
        try:
            displays = pygame.display.get_desktop_sizes()
            if idx >= len(displays): print(f"[警告] ディスプレイ {idx} が見つかりません。"); idx = 0
        except pygame.error:
            print("[警告] ディスプレイサイズ取得失敗。"); displays = [(self.config.get('display', 'default_width'), self.config.get('display', 'default_height'))]; idx = 0
        self.screen_size = displays[idx] if self.fullscreen else (self.config.get('display', 'default_width'), self.config.get('display', 'default_height'))
        flags = pygame.OPENGL | pygame.DOUBLEBUF | (pygame.FULLSCREEN if self.fullscreen else 0)
        pygame.display.set_mode(self.screen_size, flags, display=idx)
        pygame.display.set_caption("Interactive Cosmic Art")
        print(f"Pygameウィンドウサイズ: {self.screen_size[0]}x{self.screen_size[1]}")

    def _init_moderngl(self):
        self.ctx = moderngl.create_context()
        self.ctx.enable(moderngl.DEPTH_TEST | moderngl.CULL_FACE)
        self.ctx.blend_func = moderngl.SRC_ALPHA, moderngl.ONE_MINUS_SRC_ALPHA
        self.ctx.viewport = (0, 0, *self.screen_size)

    def _calculate_world_size(self):
        w, h = self.screen_size
        if self.config.get('camera', 'projection_type') == 'orthographic':
            wh = self.config.get('camera', 'orthographic_height')
            ww = wh * (w / h)
        else:
            fovy = self.config.get('camera', 'fovy')
            dist = abs(self.config.get('camera', 'eye')[2] - self.config.get('camera', 'target')[2])
            wh = 2 * dist * math.tan(math.radians(fovy) / 2)
            ww = wh * (w / h)
        return ww, wh

    def _load_shader_source(self, shader_name):
        shader_path = os.path.join("shaders", shader_name)
        if not os.path.exists(shader_path):
            raise FileNotFoundError(f"Shader file '{shader_path}' not found.")
        with open(shader_path, "r", encoding="utf-8") as f:
            return f.read()

    def _load_all_shaders(self):
        self.star_program = self.ctx.program(
            vertex_shader=self._load_shader_source('star.vert'),
            fragment_shader=self._load_shader_source('star.frag')
        )
        particle_vs = self._load_shader_source('particle.vert')
        self.particle_render_program = self.ctx.program(vertex_shader=particle_vs, fragment_shader=self._load_shader_source('particle.frag'))
        
        transform_vs_source = self._load_shader_source('particle_transform.vert')
        self.particle_transform_program = self.ctx.program(
            vertex_shader=transform_vs_source,
            varyings=['out_pos', 'out_vel', 'out_life', 'out_color']
        )
        self.particle_manager.set_programs(self.particle_render_program, self.particle_transform_program)
        
        self.indicator_program = self.ctx.program(
            vertex_shader=self._load_shader_source('indicator.vert'),
            fragment_shader=self._load_shader_source('indicator.frag')
        )
        
        sphere_vert_source = self._load_shader_source('enhanced_sphere.vert')
        self.sphere_program = self.ctx.program(
            vertex_shader=sphere_vert_source,
            fragment_shader=self._load_shader_source('enhanced_sphere.frag')
        )
        
        post_quad_vs = self._load_shader_source('post_quad.vert')
        self.post_bright_program = self.ctx.program(vertex_shader=post_quad_vs, fragment_shader=self._load_shader_source('post_bright.frag'))
        self.post_blur_program = self.ctx.program(vertex_shader=post_quad_vs, fragment_shader=self._load_shader_source('post_blur.frag'))
        self.post_composite_program = self.ctx.program(vertex_shader=post_quad_vs, fragment_shader=self._load_shader_source('post_composite.frag'))
        print("✓ 全てのシェーダーを正常に読み込みました。")

    def _load_all_textures(self):
        albedo_paths = self.config.get('art', 'texture_paths')
        self.albedo_texture_array = self._create_texture_array(albedo_paths)
        normal_paths = self.config.get('art', 'normal_map_paths')
        self.normal_texture_array = self._create_texture_array(normal_paths, is_normal=True)
        
        env_map_path = self.config.get('art', 'env_map_path')
        try:
            hdr_image = imageio.imread(env_map_path)
            hdr_image_f4 = hdr_image.astype('f4')
            hdr_texture = self.ctx.texture(hdr_image.shape[1::-1], 3, hdr_image_f4.tobytes(), dtype='f4')
            
            self.ibl_preprocessor = IBLPreprocessor(self.ctx, self, hdr_texture)
            self.ibl_preprocessor.preprocess()
            
            hdr_texture.release()

        except FileNotFoundError:
            print(f"[警告] 環境マップ '{env_map_path}' が見つかりません。IBLは無効になります。")
            self.ibl_preprocessor = None

    def _create_texture_array(self, paths, is_normal=False):
        images = []
        try:
            for path in paths:
                img = Image.open(path).convert('RGBA')
                images.append(img)
            if not images: return None
            base_size = images[0].size
            data = b''.join([img.resize(base_size).tobytes() for img in images])
            texture_array = self.ctx.texture_array((*base_size, len(images)), 4, data)
            texture_array.filter = (moderngl.LINEAR_MIPMAP_LINEAR, moderngl.LINEAR)
            texture_array.build_mipmaps()
            return texture_array
        except Exception as e:
            print(f"[エラー] テクスチャ配列の作成中にエラー: {e}")
            return None

    def _init_all_buffers(self):
        res = self.config.get('art', 'sphere_resolution')
        verts, inds, tangents, bitangents = self._generate_sphere_mesh(1.0, res)
        self.sphere_vbo = self.ctx.buffer(verts)
        self.sphere_ibo = self.ctx.buffer(inds)
        self.sphere_tangent_vbo = self.ctx.buffer(tangents)
        self.sphere_bitangent_vbo = self.ctx.buffer(bitangents)

        max_spheres = self.config.get('art', 'sphere_count')
        self.sphere_instance_vbo = self.ctx.buffer(reserve=max_spheres * 17 * 4)
        self.sphere_vao = self.ctx.vertex_array(
            self.sphere_program,
            [
                (self.sphere_vbo, '3f 2f 3f', 'position', 'in_uv', 'in_normal'),
                (self.sphere_tangent_vbo, '3f', 'in_tangent'),
                (self.sphere_bitangent_vbo, '3f', 'in_bitangent'),
                (self.sphere_instance_vbo, '3f 3f 1f 3f 1f 1f 1f 1f 1f 1f 1f /i',
                 'sphere_pos', 'object_color', 'alpha', 'rotation', 'radius',
                 'dynamic_roughness', 'dynamic_metallic', 'dynamic_emission', 'dynamic_color_multiplier',
                 'texture_id', 'normal_map_id')
            ],
            self.sphere_ibo
        )
        
        max_indicators = self.config.get('hand_tracking', 'hand_indicator_count') * (self.config.get('hand_indicators', 'trail_length') + 2)
        self.indicator_instance_vbo = self.ctx.buffer(reserve=max_indicators * 9 * 4)
        self.indicator_vao = self.ctx.vertex_array(
            self.indicator_program,
            [
                (self.sphere_vbo, '3f 2f 3x4', 'position', 'in_uv'),
                (self.indicator_instance_vbo, '3f 3f 1f 1f 1f /i', 
                 'instance_pos', 'instance_color', 'instance_alpha', 'instance_radius', 'instance_type')
            ],
            self.sphere_ibo
        )

        self.starfield.set_program(self.star_program)
        
        quad_vertices = np.array([-1, -1, 0, 0, 1, -1, 1, 0, -1, 1, 0, 1, 1, 1, 1, 1], dtype='f4')
        self.post_quad_vbo = self.ctx.buffer(quad_vertices)
        self.post_quad_vao = self.ctx.vertex_array(self.post_bright_program, [(self.post_quad_vbo, '2f 2f', 'in_vert', 'in_uv')])

    def _init_post_processing(self):
        self.enable_bloom = self.config.get('post_processing', 'enable_bloom')
        if not self.enable_bloom: return
            
        self.scene_texture = self.ctx.texture(self.screen_size, 4, dtype='f4')
        self.scene_depth = self.ctx.depth_texture(self.screen_size)
        self.scene_fbo = self.ctx.framebuffer(self.scene_texture, self.scene_depth)
        self.bright_texture = self.ctx.texture(self.screen_size, 4, dtype='f4')
        self.bright_fbo = self.ctx.framebuffer(self.bright_texture)
        self.blur_textures = [self.ctx.texture(self.screen_size, 4, dtype='f4'), self.ctx.texture(self.screen_size, 4, dtype='f4')]
        self.blur_fbos = [self.ctx.framebuffer(self.blur_textures[0]), self.ctx.framebuffer(self.blur_textures[1])]
        print("ポストプロセッシングの準備ができました。")
        
    def _generate_sphere_mesh(self, r, res):
        verts, inds, tangents, bitangents = [], [], [], []
        for i_lat in range(res + 1):
            lat, u = PI * (-0.5 + i_lat / res), i_lat / res
            for i_lon in range(res + 1):
                lon, v_ = 2 * PI * i_lon / res, i_lon / res
                x, y, z = r*np.cos(lat)*np.cos(lon), r*np.cos(lat)*np.sin(lon), r*np.sin(lat)
                nx, ny, nz = x/r, y/r, z/r
                verts.extend([x, y, z, v_, u, nx, ny, nz])
                tx, ty, tz = -np.sin(lon), np.cos(lon), 0
                tangents.extend([tx, ty, tz])
                bt = np.cross([nx,ny,nz], [tx,ty,tz])
                bitangents.extend(bt)

        for i_lat in range(res):
            for i_lon in range(res):
                v1 = i_lat * (res + 1) + i_lon; v2 = v1 + res + 1
                inds.extend([v1, v2, v1 + 1, v2, v2 + 1, v1 + 1])
        
        return np.array(verts, dtype='f4'), np.array(inds, dtype='i4'), np.array(tangents, dtype='f4'), np.array(bitangents, dtype='f4')

    def _mediapipe_worker(self):
        mp_h = mp.solutions.holistic
        mp_drawing = mp.solutions.drawing_utils
        holistic = mp_h.Holistic(
            model_complexity=self.config.get('mediapipe', 'model_complexity'),
            min_detection_confidence=self.config.get('mediapipe', 'min_detection_confidence'),
            min_tracking_confidence=self.config.get('mediapipe', 'min_tracking_confidence'),
            refine_face_landmarks=True
        )
        while self.running:
            ret, frame = self.cap.read()
            if not ret: continue
            frame = cv2.flip(frame, 1)
            results = holistic.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            
            mp_drawing.draw_landmarks(frame, results.face_landmarks, mp_h.FACEMESH_CONTOURS, landmark_drawing_spec=None, connection_drawing_spec=mp_drawing.DrawingSpec(color=(80,110,10), thickness=1, circle_radius=1))
            mp_drawing.draw_landmarks(frame, results.left_hand_landmarks, mp_h.HAND_CONNECTIONS)
            mp_drawing.draw_landmarks(frame, results.right_hand_landmarks, mp_h.HAND_CONNECTIONS)

            try:
                self.landmarks_queue.put_nowait({
                    'left': results.left_hand_landmarks, 
                    'right': results.right_hand_landmarks, 
                    'face': results.face_landmarks
                })
                fps = self.fps_history[-1] if self.fps_history else 0
                self.video_queue.put_nowait(self.debug_info.update_and_draw(frame, fps, self.sphere_manager, self.tracker))
            except queue.Full:
                try: 
                    self.landmarks_queue.get_nowait()
                    self.video_queue.get_nowait()
                except queue.Empty: pass
        holistic.close()

    def run(self):
        self.mediapipe_thread.start()
        cv2.namedWindow('Motion Capture View', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('Motion Capture View', self.camera_width // 3, self.camera_height // 3)
        landmarks = None
        
        print("コントロール:")
        print("F11: フルスクリーン切替")
        print("D: デバッグ情報表示切替")
        print("T: 球体リセット")
        print("M: 音響ON/OFF")
        print("+/-: 音量調整")
        print("新ジェスチャー:")
        print("手のひらを大きく開く → 球体召喚")
        print("指の数で効果変化 (1-5本)")
        print("両手でつまむ → 重力井戸")
        print("拡張ジェスチャー（6-10本指）:")
        print("6本指:  色彩変化（虹色グラデーション）")
        print("7本指:  重力反転（球体が上昇）")
        print("8本指:  球体分裂（数が倍増）")
        print("9本指:  時空歪曲（螺旋運動）")
        print("10本指:  創世効果（新宇宙創造）")
        print("両手組み合わせ:")
        print("左右均等: 対称効果")
        print("片手集中: 一方向効果")
        print("左手優位: 寒色・反時計回り")
        print("右手優位: 暖色・時計回り")
        
        while self.running:
            current_time = time.time()
            dt = min(0.05, current_time - self.last_time)
            self.last_time = current_time
            if dt > 0: self.fps_history.append(1.0 / dt)

            for e in pygame.event.get():
                if e.type == pygame.QUIT or (e.type == pygame.KEYDOWN and e.key == pygame.K_ESCAPE):
                    self.running = False
                if e.type == pygame.KEYDOWN:
                    if e.key == pygame.K_F11: self._toggle_fullscreen()
                    if e.key == pygame.K_d: self.debug_info.toggle()
                    if e.key == pygame.K_t: self.sphere_manager.reset()
                    if e.key == pygame.K_m: self.audio_system.toggle_enabled()
                    if e.key == pygame.K_PLUS or e.key == pygame.K_EQUALS:
                        current_vol = self.audio_system.volume
                        self.audio_system.set_volume(min(1.0, current_vol + 0.1))
                        print(f"音量: {self.audio_system.volume:.1f}")
                    if e.key == pygame.K_MINUS:
                        current_vol = self.audio_system.volume
                        self.audio_system.set_volume(max(0.0, current_vol - 0.1))
                        print(f"音量: {self.audio_system.volume:.1f}")
                    if e.key == pygame.K_r:  # Rキーで強制リセット
                        self.sphere_manager.force_reset_all_physics()
                        print("🔧 Rキーによる強制リセットを実行")
            
            try: cv2.imshow('Motion Capture View', self.video_queue.get_nowait())
            except queue.Empty: pass
            if cv2.waitKey(1) & 0xFF == 27: self.running = False

            try: landmarks = self.landmarks_queue.get_nowait()
            except queue.Empty: pass
            
            if landmarks:
                self.tracker.update_from_landmarks(landmarks['left'], landmarks['right'], landmarks['face'])

            self._update(dt, current_time)
            self._render(current_time)
            
            self.clock.tick(120)
        
        self.cleanup()

    def _update(self, dt, current_time):
        # 強化された顔向きベースのライティング
        # self._update_lighting_enhanced(self.tracker.face_detector)  # この行をコメントアウト
    
        # 代わりに固定ライティングを使用（一時的な修正）
        self.light_direction = np.array([0.3, 1.0, 0.6], dtype='f4')
        self.light_direction = self.light_direction / np.linalg.norm(self.light_direction)
    
        if self.tracker.gravity_gesture_active:
            self.gravity_well.activate(self.tracker.gravity_gesture_center)
        else:
            self.gravity_well.deactivate()
        self.gravity_well.update(dt)

        fingertips = self.tracker.get_current_fingertip_positions()
        gesture_states = self.tracker.get_gesture_states()
    
        # 新ジェスチャー対応の更新
        self.sphere_manager.update(dt, current_time, fingertips, self.gravity_well, gesture_states, self.audio_system)
        self.starfield.update(dt)
        self.particle_manager.update(dt)

    def _render(self, current_time):
        self.ctx.viewport = (0, 0, *self.screen_size)

        if self.enable_bloom:
            self.scene_fbo.use()
        
        self.ctx.clear(0.005, 0.005, 0.01, 1.0)
        self.ctx.enable(moderngl.DEPTH_TEST)
        
        view_matrix = self.camera.get_view_matrix()
        proj_matrix = self.camera.get_projection_matrix()
        mvp = proj_matrix @ view_matrix

        # Starfield Rendering
        self.ctx.disable(moderngl.DEPTH_TEST)
        self.star_program['mvp'].write(mvp.astype('f4'))
        self.starfield.render(mvp)
        self.ctx.enable(moderngl.DEPTH_TEST)

        # Particle Rendering
        self.ctx.enable(moderngl.BLEND)
        self.ctx.blend_func = moderngl.SRC_ALPHA, moderngl.ONE
        self.particle_manager.render(mvp.astype('f4'))
        
        # Main Spheres Rendering (IBL)
        self.ctx.blend_func = moderngl.SRC_ALPHA, moderngl.ONE_MINUS_SRC_ALPHA
        
        if self.albedo_texture_array: self.albedo_texture_array.use(location=0)
        if self.normal_texture_array: self.normal_texture_array.use(location=1)
        
        if self.ibl_preprocessor:
            self.ibl_preprocessor.irradiance_map.use(location=2)
            self.ibl_preprocessor.prefilter_map.use(location=3)
            self.ibl_preprocessor.brdf_lut.use(location=4)

        self.sphere_program['u_albedo_maps'].value = 0
        self.sphere_program['u_normal_maps'].value = 1
        
        if self.ibl_preprocessor:
            self.sphere_program['u_irradianceMap'].value = 2
            self.sphere_program['u_prefilterMap'].value = 3
            self.sphere_program['u_brdfLUT'].value = 4
        
        self.sphere_program['mvp'].write(mvp.astype('f4'))
        self.sphere_program['u_cam_pos'].write(self.camera.eye)
        self.sphere_program['u_light_direction'].write(self.light_direction)
        self._render_spheres()

        # Hand Indicators Rendering
        self.indicator_program['mvp'].write(mvp.astype('f4'))
        self.indicator_program['u_time'].value = current_time
        self._render_indicators(current_time)
        
        # Post-processing
        if self.enable_bloom:
            self._run_post_processing()
        
        pygame.display.flip()

    def _render_spheres(self):
        visible_spheres = [s for s in self.sphere_manager.spheres if s.alpha > 0.01]
        if not visible_spheres: return
    
        # バッファ容量をチェック
        max_spheres = self.config.get('art', 'sphere_count', default=250)
        if len(visible_spheres) > max_spheres:
            visible_spheres = visible_spheres[:max_spheres]  # 最大数に制限
    
        data = np.array([
            (*s.position, *s.color, s.alpha, *s.rotation, s.current_radius,
            s.dynamic_roughness, s.dynamic_metallic, s.dynamic_emission, s.dynamic_color_multiplier,
            s.texture_id, s.normal_map_id)
            for s in visible_spheres
        ], dtype='f4')
    
        self.sphere_instance_vbo.write(data.tobytes())
        self.sphere_vao.render(instances=len(visible_spheres))

    def _render_indicators(self, current_time):
        all_indicator_data = []
        for indicator in self.sphere_manager.hand_indicators:
            all_indicator_data.extend(indicator.get_render_data(current_time))
        if not all_indicator_data: return

        data = np.array([
            (*d['pos'], *d['color'], d['alpha'], d['radius'], d['type'])
            for d in all_indicator_data
        ], dtype='f4')
        
        self.indicator_instance_vbo.write(data.tobytes())
        self.indicator_vao.render(instances=len(all_indicator_data))

    def _run_post_processing(self):
        self.ctx.disable(moderngl.DEPTH_TEST)
        self.ctx.enable(moderngl.BLEND)
        self.ctx.blend_func = moderngl.SRC_ALPHA, moderngl.ONE_MINUS_SRC_ALPHA

        self.bright_fbo.use()
        self.bright_fbo.clear(0.0, 0.0, 0.0, 1.0)
        self.scene_texture.use(location=0)
        self.post_bright_program['u_texture'].value = 0
        self.post_bright_program['u_threshold'].value = self.config.get('post_processing', 'bloom_threshold')
        self.post_quad_vao.render(mode=moderngl.TRIANGLE_STRIP)

        blur_passes = self.config.get('post_processing', 'bloom_blur_passes')
        horizontal = True
        for i in range(blur_passes * 2):
            self.blur_fbos[i % 2].use()
            tex_to_blur = self.bright_texture if i == 0 else self.blur_textures[(i - 1) % 2]
            tex_to_blur.use(location=0)
            self.post_blur_program['u_texture'].value = 0
            self.post_blur_program['u_horizontal'].value = horizontal
            self.post_blur_program['u_texel_size'].value = (1 / self.screen_size[0], 1 / self.screen_size[1])
            self.post_quad_vao.render(mode=moderngl.TRIANGLE_STRIP)
            horizontal = not horizontal

        self.ctx.screen.use()
        self.scene_texture.use(location=0)
        self.blur_textures[(blur_passes * 2 - 1) % 2].use(location=1)
        self.post_composite_program['u_scene_texture'].value = 0
        self.post_composite_program['u_bloom_texture'].value = 1
        self.post_composite_program['u_bloom_intensity'].value = self.config.get('post_processing', 'bloom_intensity')
        self.post_quad_vao.render(mode=moderngl.TRIANGLE_STRIP)

    def _update_lighting_enhanced(self, face_detector):
        """強化された顔向きベースのライティング"""
        angles = face_detector.get_normalized_direction()
        
        # より滑らかで直感的なライティング
        target_direction = np.array([
            angles['horizontal'] * 0.8,  # 左右
            1.0 - abs(angles['vertical']) * 0.3,  # 上下
            0.8 + angles['vertical'] * 0.2  # 奥行き
        ], dtype='f4')
        
        # 信頼度に基づく補間
        blend_factor = angles['confidence'] * self.light_smoothing * 2.0
        self.light_direction += (target_direction - self.light_direction) * blend_factor
        
        # 正規化
        norm = np.linalg.norm(self.light_direction)
        if norm > 0:
            self.light_direction = self.light_direction / norm
    
    def _toggle_fullscreen(self):
        self.fullscreen = not self.fullscreen
        pygame.display.quit()
        self._init_pygame()
        self.ctx.viewport = (0, 0, *self.screen_size)
        self.camera.update_aspect_ratio(*self.screen_size)
        self._init_post_processing()
        print(f"ディスプレイモードを切り替え: {'フルスクリーン' if self.fullscreen else 'ウィンドウ'}")

    def cleanup(self):
        self.running = False
        if self.mediapipe_thread.is_alive(): self.mediapipe_thread.join(timeout=1)
        self.cap.release()
        cv2.destroyAllWindows()
        pygame.quit()
        print("クリーンアップ完了。")

# =============================================================================
# メイン実行部
# =============================================================================
if __name__ == '__main__':
    try:
        print("=" * 60 + "\nInteractive Cosmic Art v4 Enhanced\n" + "=" * 60)
        print("新機能: 音響システム + 新ジェスチャー")
        print("初期化中...")
        config = Config('config.json')
        app = App(config)
        app.run()
    except Exception as e:
        print(f"アプリケーションの実行中に致命的なエラーが発生しました: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

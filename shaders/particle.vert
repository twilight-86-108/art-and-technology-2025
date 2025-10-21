// =============================================================================
// shaders/particle.vert
// =============================================================================
#version 330

// #define TRANSFORM_PASS // この行はPython側で追加されます

// 入力: 現在のパーティクルの状態
in vec3 in_pos;
in vec3 in_vel;
in float in_life;
in vec3 in_color;

// ユニフォーム: 全パーティクルで共通の値
uniform mat4 mvp;
uniform vec3 u_gravity;
uniform float u_dt;

// --- 出力変数の定義 ---
#ifdef TRANSFORM_PASS
    // Transform Feedback用の出力
    out vec3 out_pos;
    out vec3 out_vel;
    out float out_life;
    out vec3 out_color;
#else
    // フラグメントシェーダーへの出力 (描画用)
    out vec3 v_color;
    out float v_alpha;
#endif

void main() {
    // --- 物理計算 (両方のパスで共通) ---
    vec3 new_vel = in_vel + u_gravity * u_dt;
    vec3 new_pos = in_pos + new_vel * u_dt;
    float new_life = in_life - u_dt;

    #ifdef TRANSFORM_PASS
        // Transform Feedbackパス：計算結果を次のバッファに書き込む
        if (new_life > 0.0) {
            out_pos = new_pos;
            out_vel = new_vel;
            out_life = new_life;
            out_color = in_color;
        } else {
            // 寿命が尽きたパーティクルはそのまま（再利用される）
            out_pos = vec3(0.0);
            out_vel = vec3(0.0);
            out_life = -1.0;
            out_color = vec3(0.0);
        }
    #else
        // レンダリングパス：画面上の位置を計算
        if (in_life > 0.0) {
            gl_Position = mvp * vec4(in_pos, 1.0);
            gl_PointSize = 8.0 * in_life; // 少し大きく
            v_color = in_color;
            v_alpha = in_life;
        } else {
            // 描画しない
            gl_Position = vec4(-2.0, -2.0, -2.0, 1.0);
        }
    #endif
}
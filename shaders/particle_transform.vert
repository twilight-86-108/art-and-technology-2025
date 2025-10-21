#version 330 core

// 入力: 現在のパーティクルの状態
in vec3 in_pos;
in vec3 in_vel;
in float in_life;
in vec3 in_color;

// Uniforms: Pythonから受け取る定数
uniform float u_dt;
uniform vec3 u_gravity;

// 出力: 更新後のパーティクルの状態 (Varyings)
out vec3 out_pos;
out vec3 out_vel;
out float out_life;
out vec3 out_color;

void main() {
    // 寿命が尽きていれば何もしない
    if (in_life <= 0.0) {
        out_pos = in_pos;
        out_vel = in_vel;
        out_life = -1.0; // 寿命を負にして非アクティブ状態を維持
        out_color = in_color;
        return;
    }

    // 物理演算
    vec3 new_vel = in_vel + u_gravity * u_dt;
    vec3 new_pos = in_pos + new_vel * u_dt;
    float new_life = in_life - u_dt;

    // 計算結果を出力変数に格納
    out_pos = new_pos;
    out_vel = new_vel;
    out_life = new_life;
    out_color = in_color; // 色は変更しない
}
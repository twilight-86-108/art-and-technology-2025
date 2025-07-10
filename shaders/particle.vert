#version 330

// ★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★
// ★ 修正箇所：未使用の in_vel と in_unused を削除
// ★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★
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
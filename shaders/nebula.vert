// =============================================================================
// shaders/nebula.vert
// =============================================================================
#version 330

in vec3 in_position;
in vec2 in_uv;

out vec2 v_uv;
out vec3 v_world_pos;

void main() {
    gl_Position = vec4(in_position, 1.0);
    v_uv = in_uv;
    // フルスクリーンクアッドなので、ワールド座標は適当に設定
    v_world_pos = in_position;
}
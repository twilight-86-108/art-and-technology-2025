// =============================================================================
// shaders/indicator.vert
// =============================================================================
#version 330

uniform mat4 mvp;
uniform float u_time;

in vec3 position;
in vec2 in_uv;

in vec3 instance_pos;
in vec3 instance_color;
in float instance_alpha;
in float instance_radius;
in float instance_type; // 0: main, 1: trail, 2: glow

out float v_alpha;
out vec3 v_color;
out float v_type;
out vec2 v_uv;

void main() {
    gl_Position = mvp * vec4(instance_pos + position * instance_radius, 1.0);
    v_alpha = instance_alpha;
    v_color = instance_color;
    v_type = instance_type;
    v_uv = in_uv;
}
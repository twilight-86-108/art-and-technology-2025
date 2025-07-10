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

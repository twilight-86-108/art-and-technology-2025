#version 330

in vec3 v_color;
in float v_alpha;

out vec4 fragColor;

void main() {
    fragColor = vec4(v_color, v_alpha);
}
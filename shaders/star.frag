#version 330

in float v_brightness;
out vec4 fragColor;

void main() {
    fragColor = vec4(vec3(v_brightness), 1.0);
}

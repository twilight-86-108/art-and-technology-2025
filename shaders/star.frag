// =============================================================================
// shaders/star.frag
// =============================================================================
#version 330

in float v_brightness;
out vec4 fragColor;

void main() {
    float r = 0.6 + v_brightness * 0.4;
    float g = 0.8 + v_brightness * 0.2;
    float b = 1.0;
    fragColor = vec4(r, g, b, v_brightness);
}
// =============================================================================
// shaders/indicator.frag
// =============================================================================
#version 330

in float v_alpha;
in vec3 v_color;
in float v_type;
in vec2 v_uv;

uniform float u_time;

out vec4 fragColor;

float noise(vec2 p) {
    return fract(sin(dot(p, vec2(12.9898, 78.233))) * 43758.5453);
}

void main() {
    vec3 final_color = v_color;
    float final_alpha = v_alpha;

    if (v_type < 0.5) { // Main sphere
        float pulse = (sin(u_time * 5.0) * 0.5 + 0.5) * 0.3;
        final_color += vec3(pulse);
    } else if (v_type < 1.5) { // Trail
        // Trail is simple
    } else { // Glow
        final_alpha *= 0.5;
    }

    // A simple fresnel-like effect for the core
    float fresnel = pow(1.0 - abs(v_uv.y * 2.0 - 1.0), 2.0);
    final_color += fresnel * 0.5;

    fragColor = vec4(final_color, final_alpha);
}
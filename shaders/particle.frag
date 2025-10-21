// =============================================================================
// shaders/particle.frag
// =============================================================================
#version 330

in vec3 v_color;
in float v_alpha;

out vec4 fragColor;

void main() {
    // ポイントスプライト用のテクスチャ座標
    vec2 coord = gl_PointCoord - vec2(0.5);
    float dist = length(coord);

    // 中心が明るく、縁がぼやけるような効果
    float mask = smoothstep(0.5, 0.0, dist);

    fragColor = vec4(v_color, v_alpha * mask);
}
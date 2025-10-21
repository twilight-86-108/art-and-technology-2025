// =============================================================================
// shaders/post_bright.frag
// =============================================================================
#version 330
// ブルームのために、画像の明るい部分だけを抽出する

in vec2 v_uv;
uniform sampler2D u_texture;
uniform float u_threshold;

out vec4 fragColor;

void main() {
    vec4 color = texture(u_texture, v_uv);
    float brightness = dot(color.rgb, vec3(0.2126, 0.7152, 0.0722));
    if (brightness > u_threshold) {
        fragColor = color;
    } else {
        fragColor = vec4(0.0, 0.0, 0.0, 1.0);
    }
}
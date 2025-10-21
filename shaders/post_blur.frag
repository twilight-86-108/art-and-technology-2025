// =============================================================================
// shaders/post_blur.frag
// =============================================================================
#version 330
// ガウシアンブラーをかける

in vec2 v_uv;
uniform sampler2D u_texture;
uniform bool u_horizontal;
uniform vec2 u_texel_size;

out vec4 fragColor;

// 9-tap ガウシアンブラー
float weight[5] = float[] (0.227027, 0.1945946, 0.1216216, 0.05405405, 0.01621621);

void main() {
    vec3 result = texture(u_texture, v_uv).rgb * weight[0];
    if (u_horizontal) {
        for(int i = 1; i < 5; ++i) {
            result += texture(u_texture, v_uv + vec2(u_texel_size.x * float(i), 0.0)).rgb * weight[i];
            result += texture(u_texture, v_uv - vec2(u_texel_size.x * float(i), 0.0)).rgb * weight[i];
        }
    } else {
        for(int i = 1; i < 5; ++i) {
            result += texture(u_texture, v_uv + vec2(0.0, u_texel_size.y * float(i))).rgb * weight[i];
            result += texture(u_texture, v_uv - vec2(0.0, u_texel_size.y * float(i))).rgb * weight[i];
        }
    }
    fragColor = vec4(result, 1.0);
}
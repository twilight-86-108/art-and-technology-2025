// =============================================================================
// shaders/post_composite.frag
// =============================================================================
#version 330
// 元のシーンとブルームテクスチャを合成する

in vec2 v_uv;
uniform sampler2D u_scene_texture;
uniform sampler2D u_bloom_texture;
uniform float u_bloom_intensity;

out vec4 fragColor;

void main() {
    vec3 scene_color = texture(u_scene_texture, v_uv).rgb;
    vec3 bloom_color = texture(u_bloom_texture, v_uv).rgb;
    
    // 加算合成
    vec3 final_color = scene_color + bloom_color * u_bloom_intensity;

    fragColor = vec4(final_color, 1.0);
}

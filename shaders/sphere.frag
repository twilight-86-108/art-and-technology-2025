#version 330

// 頂点シェーダーから受け取る変数
in vec3 v_normal;
in vec3 v_world_pos;
in vec3 v_color;
in float v_alpha;
in vec2 v_uv;

// ユニフォーム
uniform vec3 u_light_direction;
uniform sampler2D u_texture;

// 出力
out vec4 fragColor;

void main() {
    // テクスチャから基本色を取得
    vec3 tex_color = texture(u_texture, v_uv).rgb;

    // ライティング計算
    vec3 light_dir = normalize(u_light_direction);
    float ambient = 0.4; // 少し明るくしてテクスチャを見やすくする
    float diffuse = max(0.0, dot(v_normal, light_dir)) * 0.7;
    
    // スペキュラ（ハイライト）
    vec3 view_dir = normalize(-v_world_pos);
    vec3 reflect_dir = reflect(-light_dir, v_normal);
    float spec = pow(max(dot(view_dir, reflect_dir), 0.0), 32.0) * 0.4;
    
    // テクスチャ色、インスタンスごとの色(v_color)、ライティングを合成
    vec3 final_color = tex_color * v_color * (ambient + diffuse) + vec3(spec);
    
    fragColor = vec4(final_color, v_alpha);
}

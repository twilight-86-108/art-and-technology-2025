// 最終更新 2025/07/11 21:50
// =============================================================================
// shaders/enhanced_sphere.vert
// =============================================================================
#version 330

// Uniforms
uniform mat4 mvp;
uniform mat4 model_matrix; // ### 変更点: model行列を受け取る

// Per-vertex attributes
in vec3 position;
in vec2 in_uv;
in vec3 in_normal; // ### 変更点: 法線を受け取る
in vec3 in_tangent; // ### 新規追加: 接空間のTBNベクトル
in vec3 in_bitangent;

// Per-instance attributes
in vec3 sphere_pos;
in vec3 object_color;
in float alpha;
in vec3 rotation;
in float radius;
in float dynamic_roughness;
in float dynamic_metallic;
in float dynamic_emission;
in float dynamic_color_multiplier;
in float texture_id;
in float normal_map_id;

// Outputs
out vec3 v_world_pos;
out vec2 v_uv;
out mat3 v_tbn; // ### 変更点: TBN行列を渡す
out vec3 v_color;
out float v_alpha;
out float v_roughness;
out float v_metallic;
out float v_emission;
out float v_color_multiplier;
flat out int v_texture_id;
flat out int v_normal_map_id;

mat3 rotationMatrix(vec3 angles){
    vec3 c = cos(angles);
    vec3 s = sin(angles);
    return mat3(
        c.y*c.z, -c.y*s.z, s.y,
        c.x*s.z+s.x*s.y*c.z, c.x*c.z-s.x*s.y*s.z, -s.x*c.y,
        s.x*s.z-c.x*s.y*c.z, s.x*c.z+c.x*s.y*s.z, c.x*c.y
    );
}

void main() {
    mat3 rot_m = rotationMatrix(rotation);
    vec3 w_pos = (rot_m * (position * radius)) + sphere_pos;
    gl_Position = mvp * vec4(w_pos, 1.0);
    
    v_world_pos = w_pos;
    v_uv = in_uv;

    // TBN行列を計算してフラグメントシェーダーへ
    vec3 T = normalize(rot_m * in_tangent);
    vec3 B = normalize(rot_m * in_bitangent);
    vec3 N = normalize(rot_m * in_normal);
    v_tbn = mat3(T, B, N);

    // インスタンスごとのデータを渡す
    v_color = object_color;
    v_alpha = alpha;
    v_roughness = dynamic_roughness;
    v_metallic = dynamic_metallic;
    v_emission = dynamic_emission;
    v_color_multiplier = dynamic_color_multiplier;
    v_texture_id = int(texture_id);
    v_normal_map_id = int(normal_map_id);
}
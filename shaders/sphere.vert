#version 330

// 頂点ごとのデータ
in vec3 position;
in vec2 in_uv;

// インスタンスごとのデータ
in vec3 sphere_pos;
in vec3 object_color;
in float alpha;
in vec3 rotation;
in float radius;

// ユニフォーム
uniform mat4 mvp;

// フラグメントシェーダーへ渡す変数
out vec3 v_normal;
out vec3 v_world_pos;
out vec3 v_color;
out float v_alpha;
out vec2 v_uv;

// 回転行列を生成する関数
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
    
    vec3 scaled_pos = position * radius;
    vec3 w_pos = (rot_m * scaled_pos) + sphere_pos;
    
    gl_Position = mvp * vec4(w_pos, 1.0);
    
    v_normal = normalize(rot_m * position);
    v_world_pos = w_pos;
    v_color = object_color;
    v_alpha = alpha;
    v_uv = in_uv;
}

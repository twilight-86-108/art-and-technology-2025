// =============================================================================
// shaders/nebula.frag
// =============================================================================
#version 330

in vec2 v_uv;
in vec3 v_world_pos;

uniform float u_time;
uniform vec2 u_resolution;
uniform float u_density;
uniform float u_scale;
uniform float u_interaction_energy;
uniform vec3 u_nebula_color;
uniform mat4 u_disturbances; // 4x4 matrix で 4つの擾乱点 (pos.xyz, intensity)
uniform int u_disturbance_count;

out vec4 fragColor;

// 3Dノイズ関数
float noise3D(vec3 p) {
    return fract(sin(dot(p, vec3(12.9898, 78.233, 37.719))) * 43758.5453);
}

// フラクタルノイズ
float fractalNoise(vec3 p) {
    float value = 0.0;
    float amplitude = 1.0;
    float frequency = 1.0;
    
    for (int i = 0; i < 4; i++) {
        value += noise3D(p * frequency) * amplitude;
        amplitude *= 0.5;
        frequency *= 2.0;
    }
    
    return value;
}

// より複雑な3Dノイズ（パーリンノイズ風）
float perlinNoise3D(vec3 p) {
    vec3 i = floor(p);
    vec3 f = fract(p);
    f = f * f * (3.0 - 2.0 * f); // スムーズステップ
    
    float a = noise3D(i);
    float b = noise3D(i + vec3(1.0, 0.0, 0.0));
    float c = noise3D(i + vec3(0.0, 1.0, 0.0));
    float d = noise3D(i + vec3(1.0, 1.0, 0.0));
    float e = noise3D(i + vec3(0.0, 0.0, 1.0));
    float f_val = noise3D(i + vec3(1.0, 0.0, 1.0));
    float g = noise3D(i + vec3(0.0, 1.0, 1.0));
    float h = noise3D(i + vec3(1.0, 1.0, 1.0));
    
    float x1 = mix(a, b, f.x);
    float x2 = mix(c, d, f.x);
    float x3 = mix(e, f_val, f.x);
    float x4 = mix(g, h, f.x);
    
    float y1 = mix(x1, x2, f.y);
    float y2 = mix(x3, x4, f.y);
    
    return mix(y1, y2, f.z);
}

// 星雲の密度計算
float nebulaDensity(vec3 worldPos, float time) {
    // ベースとなる3D位置
    vec3 pos = worldPos * u_scale;
    
    // 時間による変化
    pos += vec3(time * 0.1, time * 0.05, time * 0.08);
    
    // 複数のノイズレイヤーを重ね合わせ
    float density = 0.0;
    
    // 大きなスケールの構造
    density += perlinNoise3D(pos * 0.5) * 0.8;
    
    // 中程度のスケールの細部
    density += perlinNoise3D(pos * 1.5) * 0.4;
    
    // 細かいテクスチャ
    density += fractalNoise(pos * 4.0) * 0.2;
    
    // インタラクションエネルギーの影響
    density += u_interaction_energy * 0.3;
    
    // 擾乱の影響
    for (int i = 0; i < u_disturbance_count; i++) {
        vec3 disturbance_pos = u_disturbances[i].xyz;
        float disturbance_intensity = u_disturbances[i].w;
        
        float dist = length(worldPos - disturbance_pos);
        float influence = exp(-dist * 2.0) * disturbance_intensity;
        
        // 擾乱による密度の変動
        vec3 disturbed_pos = pos + vec3(influence * 2.0);
        density += perlinNoise3D(disturbed_pos * 2.0) * influence * 0.5;
    }
    
    return clamp(density, 0.0, 1.0);
}

// 色の計算
vec3 calculateNebulaColor(float density, vec3 baseColor, vec3 worldPos) {
    // 基本色
    vec3 color = baseColor;
    
    // 密度に応じた色の変化
    vec3 dense_color = baseColor * 1.5; // 密な部分はより明るく
    vec3 sparse_color = baseColor * 0.3; // 薄い部分は暗く
    
    color = mix(sparse_color, dense_color, density);
    
    // 温度による色変化（ホットスポット）
    float temperature = fractalNoise(worldPos * 3.0 + u_time * 0.2);
    if (temperature > 0.7) {
        // ホットスポットでは暖色系に
        color += vec3(0.3, 0.1, 0.0) * (temperature - 0.7) * 3.0;
    }
    
    // 深度による色の変化
    float depth_factor = (worldPos.z + 1.0) * 0.5; // -1.0 ～ 1.0 を 0.0 ～ 1.0 に
    color *= 0.7 + depth_factor * 0.3; // 奥は少し暗く
    
    return color;
}

void main() {
    // UV座標を世界座標にマッピング
    vec2 uv = v_uv * 2.0 - 1.0; // -1.0 ～ 1.0
    vec3 worldPos = vec3(uv * u_resolution / max(u_resolution.x, u_resolution.y), 0.0);
    
    // 複数の深度での星雲を合成
    vec3 finalColor = vec3(0.0);
    float totalAlpha = 0.0;
    
    for (int layer = 0; layer < 3; layer++) {
        float layerDepth = float(layer) * 0.3 - 0.3; // -0.3, 0.0, 0.3
        vec3 layerPos = worldPos + vec3(0.0, 0.0, layerDepth);
        
        // この層の密度
        float density = nebulaDensity(layerPos, u_time);
        density *= u_density; // ユーザー設定の密度
        
        // この層の色
        vec3 layerColor = calculateNebulaColor(density, u_nebula_color, layerPos);
        
        // 層の透明度
        float layerAlpha = density * 0.4; // 基本透明度
        
        // アルファブレンディング
        finalColor += layerColor * layerAlpha * (1.0 - totalAlpha);
        totalAlpha += layerAlpha * (1.0 - totalAlpha);
        
        if (totalAlpha > 0.95) break; // 十分不透明になったら終了
    }
    
    // 星のような輝点を追加
    float starNoise = fractalNoise(worldPos * 20.0 + u_time * 0.1);
    if (starNoise > 0.85) {
        float starBrightness = (starNoise - 0.85) * 6.0;
        finalColor += vec3(starBrightness);
        totalAlpha = min(1.0, totalAlpha + starBrightness * 0.1);
    }
    
    // 最終的な透明度を調整（背景なのであまり濃くしない）
    totalAlpha *= 0.6;
    
    fragColor = vec4(finalColor, totalAlpha);
}
// =============================================================================
// shaders/enhanced_sphere.frag
// =============================================================================
#version 330 core

// Inputs
in vec3 v_world_pos;
in vec2 v_uv;
in mat3 v_tbn;
in vec3 v_color;
in float v_alpha;
in float v_roughness;
in float v_metallic;
in float v_emission;
in float v_color_multiplier;
flat in int v_texture_id;
flat in int v_normal_map_id;

// Uniforms
uniform vec3 u_cam_pos;
uniform vec3 u_light_direction;
uniform sampler2DArray u_albedo_maps;
uniform sampler2DArray u_normal_maps;
uniform samplerCube u_irradianceMap;
uniform samplerCube u_prefilterMap;
uniform sampler2D u_brdfLUT;

// Output
out vec4 fragColor;

const float PI = 3.14159265359;

// --- BRDF関数群 (前回と同じなので省略) ---
float D_GGX(vec3 N, vec3 H, float roughness) {
    float a = roughness * roughness; float a2 = a * a;
    float NdotH = max(dot(N, H), 0.0); float NdotH2 = NdotH * NdotH;
    float nom = a2;
    float denom = (NdotH2 * (a2 - 1.0) + 1.0);
    denom = PI * denom * denom;
    return nom / max(denom, 0.0001);
}
float G_SchlickGGX(float NdotV, float roughness) {
    float r = (roughness + 1.0); float k = (r * r) / 8.0;
    return NdotV / (NdotV * (1.0 - k) + k);
}
float G_Smith(vec3 N, vec3 V, vec3 L, float roughness) {
    float NdotV = max(dot(N, V), 0.0); float NdotL = max(dot(N, L), 0.0);
    return G_SchlickGGX(NdotV, roughness) * G_SchlickGGX(NdotL, roughness);
}
vec3 F_Schlick(float cosTheta, vec3 F0) {
    return F0 + (1.0 - F0) * pow(clamp(1.0 - cosTheta, 0.0, 1.0), 5.0);
}

void main() {
    vec3 albedo = texture(u_albedo_maps, vec3(v_uv, float(v_texture_id))).rgb * v_color * v_color_multiplier;
    float metallic = v_metallic;
    float roughness = v_roughness;
    vec3 emission = albedo * v_emission;

    vec3 N = normalize(v_tbn * (texture(u_normal_maps, vec3(v_uv, float(v_normal_map_id))).rgb * 2.0 - 1.0));
    vec3 V = normalize(u_cam_pos - v_world_pos);
    vec3 R = reflect(-V, N);

    vec3 F0 = mix(vec3(0.04), albedo, metallic);

    // === Direct Lighting ===
    vec3 Lo = vec3(0.0);
    vec3 L = normalize(u_light_direction);
    vec3 H = normalize(V + L);
    float NdotL = max(dot(N, L), 0.0);
    if(NdotL > 0.0) {
        float NDF = D_GGX(N, H, roughness);
        float G   = G_Smith(N, V, L, roughness);
        vec3  F   = F_Schlick(max(dot(H, V), 0.0), F0);
        vec3 kS = F;
        vec3 kD = vec3(1.0) - kS;
        kD *= 1.0 - metallic;
        vec3 numerator    = NDF * G * F;
        float denominator = 4.0 * max(dot(N, V), 0.0) * NdotL + 0.001;
        vec3 specular = numerator / denominator;
        Lo += (kD * albedo / PI + specular) * NdotL * vec3(1.0); // Light color is white
    }

    // === Indirect Lighting (IBL) ===
    vec3 F = F_Schlick(max(dot(N, V), 0.0), F0);
    vec3 kS = F;
    vec3 kD = vec3(1.0) - kS;
    kD *= 1.0 - metallic;

    vec3 irradiance = texture(u_irradianceMap, N).rgb;
    vec3 diffuse    = irradiance * albedo;
    
    const float MAX_REFLECTION_LOD = 4.0;
    vec3 prefilteredColor = textureLod(u_prefilterMap, R,  roughness * MAX_REFLECTION_LOD).rgb;    
    vec2 brdf  = texture(u_brdfLUT, vec2(max(dot(N, V), 0.0), roughness)).rg;
    vec3 specular = prefilteredColor * (F * brdf.x + brdf.y);

    vec3 ambient = kD * diffuse + specular;

    vec3 final_color = ambient + Lo + emission;
    
    // HDR to LDR
    final_color = final_color / (final_color + vec3(1.0));
    final_color = pow(final_color, vec3(1.0/2.2)); 

    fragColor = vec4(final_color, v_alpha);
}
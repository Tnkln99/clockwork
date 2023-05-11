#version 450

layout(location = 0) in vec3 pos;
layout(location = 1) in vec3 color;
layout(location = 2) in vec3 normal;
layout(location = 3) in vec2 uv;

layout(location = 0) out vec3 fragColor;
layout(location = 1) out vec3 fragPosWorld;
layout(location = 2) out vec3 fragNormalWorld;

layout(set = 0, binding = 0) uniform GlobalUbo{
    mat4 projectionMatrix;
    mat4 viewMatrix;
    vec4 ambientLightColor; // w is intesity
    vec3 lightPosition;
    vec4 lightColor;
} ubo;

layout(push_constant) uniform Push{
    mat4 modelMatrix; // projection * view * transform
    mat4 normalMatrix;
} push;


void main(){
    vec4 positionWorld = push.modelMatrix * vec4(pos, 1.0f);
    gl_Position = ubo.projectionMatrix * ubo.viewMatrix * positionWorld;
    fragNormalWorld = normalize(mat3(push.normalMatrix) * normal);
    fragPosWorld = positionWorld.xyz;
    fragColor = color;
}
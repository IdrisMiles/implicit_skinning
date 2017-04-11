#version 330
in vec3 vert;
in vec3 vertNormal;

uniform vec3 lightPos;
uniform vec3 uColour;

out vec4 fragColor;


void main()
{
   vec3 L = normalize(lightPos - vert);
   float NL = max(dot(normalize(vertNormal), L), 0.0);
   vec3 col = clamp(uColour * 0.2 + uColour * 0.8 * NL, 0.0, 1.0);
   fragColor = vec4(0.5f*(vertNormal+vec3(1.0f)), 1.0);
}

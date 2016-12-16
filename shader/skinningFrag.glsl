#version 330
in vec3 vert;
in vec3 vertNormal;
in vec3 boneColour;

uniform vec3 lightPos;
uniform vec3 uColour;

out vec4 fragColor;



void main()
{
   vec3 L = normalize(lightPos - vert);
   float NL = max(dot(normalize(vertNormal), L), 0.0);
   vec3 col = clamp(boneColour * 0.3 + boneColour * 0.7 * NL, 0.0, 1.0);
   fragColor = vec4(col, 1.0);
}

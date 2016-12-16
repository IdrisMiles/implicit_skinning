#version 330

in vec3 gVertColour;
out vec4 fragColor;


void main()
{
   fragColor = vec4(gVertColour, 1.0);
}

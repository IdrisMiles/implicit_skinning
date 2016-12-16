#version 330
in vec3 vertex;
in ivec4 BoneIDs;
in vec4 Weights;
in vec3 colour;

const int MAX_BONES = 100;

uniform mat4 projMatrix;
uniform mat4 mvMatrix;
uniform mat4 Bones[MAX_BONES];

out vec3 vertColour;

void main()
{
    vertColour = colour;

    mat4 BoneTransform = Bones[BoneIDs[0]] * Weights[0];
    BoneTransform += Bones[BoneIDs[1]] * Weights[1];
    BoneTransform += Bones[BoneIDs[2]] * Weights[2];
    BoneTransform += Bones[BoneIDs[3]] * Weights[3];

    gl_Position = projMatrix * mvMatrix * BoneTransform * vec4(vertex, 1.0);

}

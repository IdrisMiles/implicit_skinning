#version 330
layout(lines) in;
layout(triangle_strip, max_vertices = 18) out;

in vec3 vertColour[2];
out vec3 gVertColour;

void main()
{
    vec3 norm = cross(gl_in[0].gl_Position.xyz, gl_in[1].gl_Position.xyz);
    float projZ = dot(norm, vec3(0.0, 0.0, 1.0));
    vec3 projVec = norm - (projZ*vec3(0.0, 0.0, 1.0));
    vec4 offset = 0.5 * normalize(vec4(projVec, 0.0));

    vec4 offsetTR = 0.9f*vec4(1.0, 1.0, 0.0, 0.0);
    vec4 offsetBR = 0.9f*vec4(1.0, -1.0, 0.0, 0.0);
    vec4 offsetTL = 0.9f*vec4(-1.0, 1.0, 0.0, 0.0);
    vec4 offsetBL = 0.9f*vec4(-1.0, -1.0, 0.0, 0.0);

    // Top joint
    gl_Position = gl_in[0].gl_Position + offsetBL;    // bottom left
    gVertColour = vertColour[0];
    EmitVertex();

    gl_Position = gl_in[0].gl_Position + offsetTL;    // top left
    gVertColour = vertColour[0];
    EmitVertex();

    gl_Position = gl_in[0].gl_Position + offsetTR;    // top right
    gVertColour = vertColour[0];
    EmitVertex();

    gl_Position = gl_in[0].gl_Position +offsetBL;    // bottom left
    gVertColour = vertColour[0];
    EmitVertex();

    gl_Position = gl_in[0].gl_Position + offsetTR;    // top right
    gVertColour = vertColour[0];
    EmitVertex();

    gl_Position = gl_in[0].gl_Position + offsetBR;    // bottom right
    gVertColour = vertColour[0];
    EmitVertex();


    EndPrimitive();


    // Bone
    gl_Position = gl_in[1].gl_Position - 0.1f*offset;    // bottom left
    gVertColour = vertColour[1];
    EmitVertex();

    gl_Position = gl_in[0].gl_Position - offset;    // top left
    gVertColour = vertColour[0];
    EmitVertex();

    gl_Position = gl_in[0].gl_Position + offset;    // top right
    gVertColour = vertColour[0];
    EmitVertex();

    gl_Position = gl_in[1].gl_Position - 0.1f*offset;    // bottom left
    gVertColour = vertColour[1];
    EmitVertex();

    gl_Position = gl_in[0].gl_Position + offset;    // top right
    gVertColour = vertColour[0];
    EmitVertex();

    gl_Position = gl_in[1].gl_Position + 0.1f*offset;    // bottom right
    gVertColour = vertColour[1];
    EmitVertex();


    EndPrimitive();



    // Bottom joint
    gl_Position = gl_in[1].gl_Position + offsetBL;    // bottom left
    gVertColour = vertColour[1];
    EmitVertex();

    gl_Position = gl_in[1].gl_Position + offsetTL;    // top left
    gVertColour = vertColour[1];
    EmitVertex();

    gl_Position = gl_in[1].gl_Position + offsetTR;    // top right
    gVertColour = vertColour[1];
    EmitVertex();

    gl_Position = gl_in[1].gl_Position +offsetBL;    // bottom left
    gVertColour = vertColour[1];
    EmitVertex();

    gl_Position = gl_in[1].gl_Position + offsetTR;    // top right
    gVertColour = vertColour[1];
    EmitVertex();

    gl_Position = gl_in[1].gl_Position + offsetBR;    // bottom right
    gVertColour = vertColour[1];
    EmitVertex();


    EndPrimitive();
}

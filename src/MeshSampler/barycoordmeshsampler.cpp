#include "include/MeshSampler/barycoordmeshsampler.h"
#include <vector>
#include <algorithm>
#include <random>

struct AcceptSample
{
    AcceptSample() : accept(false) {}
    AcceptSample(const glm::vec3 &_samplePoint, const float _sampleDist) :
        accept(false),
        samplePoint(_samplePoint),
        sampleDist(_sampleDist)
    {}

    void operator()(const glm::vec3 &activePoint)
    {
        if(glm::distance(activePoint, samplePoint) < sampleDist)
        {
            accept = true;
        }
    }

    glm::vec3 samplePoint;
    float sampleDist;
    bool accept;
};

Mesh MeshSampler::BaryCoord::SampleMesh(const Mesh &_mesh, const int _numSamples)
{

    // iterate through triangles and get their area
    std::vector<float> triAreas;//(_mesh.m_meshTris.size());
    for(auto &tri : _mesh.m_meshTris)
    {
        glm::vec3 v1 = _mesh.m_meshVerts[tri[0]];
        glm::vec3 v2 = _mesh.m_meshVerts[tri[1]];
        glm::vec3 v3 = _mesh.m_meshVerts[tri[2]];

        glm::vec3 e1 = v2 - v1;
        glm::vec3 e2 = v3 - v1;

        float area = 0.5f * glm::length(glm::cross(e1, e2));

        triAreas.push_back(area);
    }


    // Get smallest triangle area
    float smallestArea = *std::min_element(triAreas.begin(), triAreas.end());


    // Get the proprtion of all triangle areas relative to smallest area
    std::vector<unsigned int> triAreaProportions;
    for(auto &triArea : triAreas)
    {
        unsigned int propirtion = ceil(triArea / smallestArea);
        triAreaProportions.push_back(propirtion);
    }


    // Generate Sample Probability Vector
    std::vector<unsigned int> sampleTriProbability;
    unsigned int triIndex=0;
    for(auto &triAreaPro : triAreaProportions)
    {
        for(unsigned int j=0; j<triAreaPro; j++)
        {
            sampleTriProbability.push_back(triIndex);
        }

        triIndex++;
    }



    // figure out our sampling criteria
    float totalArea = std::accumulate(triAreas.begin(), triAreas.end(), 0);
    float sampleArea = totalArea / _numSamples;
    float sampleDist = (sqrt(sampleArea));


    // initiliase random number generator
    std::default_random_engine prg;
    std::uniform_int_distribution<unsigned int> randDistTriIndex(0, sampleTriProbability.size());
    std::uniform_real_distribution<float> randDistBarycentric(0.0f, 1.0f);


    // Get sample points
    int currIteration;
    int maxIterations = 1000;
    Mesh samples;
    while(samples.m_meshVerts.size() < _numSamples && currIteration < maxIterations)
    {
        unsigned int sampleTriProbabilityIndex = randDistTriIndex(prg);
        triIndex = sampleTriProbability[sampleTriProbabilityIndex];

        glm::vec3 v1 = _mesh.m_meshVerts[_mesh.m_meshTris[triIndex][0]];
        glm::vec3 v2 = _mesh.m_meshVerts[_mesh.m_meshTris[triIndex][1]];
        glm::vec3 v3 = _mesh.m_meshVerts[_mesh.m_meshTris[triIndex][2]];

        float u = randDistBarycentric(prg);
        float v = randDistBarycentric(prg);
        if(u+v > 1.0)
        {
            v = 1.0f - u;
        }
        float w = 1.0f - (u+v);

        glm::vec3 samplePoint = (u*v1) + (v*v2) + (w*v3);

        // Check samplePoint meets sampling criteria before adding to sample list
        AcceptSample accept(samplePoint, sampleDist);
        accept = std::for_each(samples.m_meshVerts.begin(), samples.m_meshVerts.end(), accept);

        if(accept.accept)
        {
            samples.m_meshVerts.push_back(samplePoint);
            samples.m_meshNorms.push_back(_mesh.m_meshNorms[_mesh.m_meshTris[triIndex][0]]);
        }


        currIteration++;
    }




    return Mesh();
}

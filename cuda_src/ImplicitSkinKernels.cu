#include "ImplicitSkinKernels.h"
#include <cuda_runtime.h>
#include <stdio.h>


//------------------------------------------------------------------------------------------------
// CUDA Device Functions
//------------------------------------------------------------------------------------------------

//------------------------------------------------------------------------------------------------

__device__ glm::vec3 ProjectPointOnToPlane(const glm::vec3 &_point, const glm::vec3 &_planeOrigin, const glm::vec3 &_planeNormal)
{
    return (_point - (glm::dot(_point - _planeOrigin, _planeNormal) * _planeNormal));
}

//------------------------------------------------------------------------------------------------

__device__ void EvalGlobalField(float &_outputF,
                                const glm::vec3 &_samplePoint,
                                const int _numSamples,
                                const glm::mat4 *_textureSpace,
                                const glm::mat4 *_rigidTransforms,
                                const cudaTextureObject_t *_fieldFuncs,
                                const int _numFields,
                                const cudaTextureObject_t *_compOps,
                                const cudaTextureObject_t *_theta,
                                const int _numOps,
                                const ComposedFieldCuda *_compFields,
                                const int _numCompFields)
{
    float f[100];
    glm::vec3 df[100];
    int i=0;
    for(i=0; i<_numFields; i++)
    {
        glm::mat4 rigidTrans = _rigidTransforms[i];
        glm::mat4 textureSpace = _textureSpace[i];
        glm::vec3 transformedPoint = glm::vec3(glm::inverse(rigidTrans) * glm::vec4(_samplePoint, 1.0f));
        glm::vec3 texturePoint = glm::vec3(textureSpace * glm::vec4(transformedPoint, 1.0f));
        texturePoint = 1.015f*texturePoint;

        float4 val = tex3D<float4>(_fieldFuncs[i], texturePoint.x, texturePoint.y, texturePoint.z);
        f[i] = val.w;
        df[i] = glm::vec3(val.x, val.y, val.z);
    }


    float cf[100];
    float maxF = FLT_MIN;
    for(i=0; i<_numCompFields; i++)
    {
        int f1Id = _compFields[i].fieldFuncA;
        int f2Id = _compFields[i].fieldFuncB;
        int coId = _compFields[i].compOp;

        if(f2Id > -1)
        {
            glm::vec3 df1(df[f1Id].x, df[f1Id].y, df[f1Id].z);
            glm::vec3 df2(df[f2Id].x, df[f2Id].y, df[f2Id].z);
            float angle = glm::angle(df1, df2);
            float theta = tex1D<float>(_theta[coId], (angle*0.5f*M_1_PI));

            float4 val = tex3D<float4>(_compOps[coId], f[f1Id], f[f2Id], theta);
            cf[i] = val.w;
        }
        else
        {
            cf[i] = f[f1Id];
        }
        maxF = (cf[i]>maxF) ? cf[i] : maxF;
    }


    _outputF = maxF;

}

//------------------------------------------------------------------------------------------------


__device__ void EvalGradGlobalField(float &_outputF,
                                    glm::vec3 &_outputG,
                                    const glm::vec3 &_samplePoint,
                                    const int _numSamples,
                                    const glm::mat4 *_textureSpace,
                                    const glm::mat4 *_rigidTransforms,
                                    const cudaTextureObject_t *_fieldFuncs,
                                    const int _numFields,
                                    const cudaTextureObject_t *_compOps,
                                    const cudaTextureObject_t *_theta,
                                    const int _numOps,
                                    const ComposedFieldCuda *_compFields,
                                    const int _numCompFields)
{

    float h = 60.0f / 64.0f;

    float newf;
    EvalGlobalField(newf, _samplePoint, _numSamples, _textureSpace, _rigidTransforms, _fieldFuncs, _numFields, _compOps, _theta, _numOps, _compFields, _numCompFields);
    float x2;
    glm::vec3 sampleX = _samplePoint + glm::vec3(h, 0.0f, 0.0f);
    EvalGlobalField(x2, sampleX, _numSamples, _textureSpace, _rigidTransforms, _fieldFuncs, _numFields, _compOps, _theta, _numOps, _compFields, _numCompFields);
    float y2;
    glm::vec3 sampleY = _samplePoint + glm::vec3(0.0f, h, 0.0f);
    EvalGlobalField(y2, sampleY, _numSamples, _textureSpace, _rigidTransforms, _fieldFuncs, _numFields, _compOps, _theta, _numOps, _compFields, _numCompFields);
    float z2;
    glm::vec3 sampleZ = _samplePoint + glm::vec3(0.0f, 0.0f, h);
    EvalGlobalField(z2, sampleZ, _numSamples, _textureSpace, _rigidTransforms, _fieldFuncs, _numFields, _compOps, _theta, _numOps, _compFields, _numCompFields);

    _outputF = newf;
    _outputG = glm::vec3((x2-newf)/h, (y2-newf)/h, (z2-newf)/h);

}



//------------------------------------------------------------------------------------------------

__device__ void VertexProjection(glm::vec3 &_deformedVert,
                                 const float &_origIso,
                                 const float &_newIso,
                                 const glm::vec3 &_newIsoGrad,
                                 glm::vec3 &_prevIsoGrad,
                                 float &_gradAngle,
                                 const float &_sigma,
                                 const float &_contactAngle)
{
    float angle = _gradAngle = glm::degrees(glm::angle(glm::normalize(_newIsoGrad), glm::normalize(_prevIsoGrad)));

    if(angle <= _contactAngle)
    {
        glm::vec3 displacement = ( _sigma * (_newIso - _origIso) * (_newIsoGrad / glm::length2(_newIsoGrad)));
        _deformedVert = _deformedVert + displacement;
    }
}

//------------------------------------------------------------------------------------------------

__device__ void TangentialRelaxation (glm::vec3 &_deformedVert,
                                     const glm::vec3 &_normal,
                                     const float _origIso,
                                     const float _newIso,
                                      glm::vec3 *_verts,
                                     const int *_oneRingNeigh,
                                     const float *_centroidWeights,
                                     const int _numNeighs)
{
    float mu = 1.0f - powf(fabs(_newIso- _origIso) - 1.0f, 4.0f);
    mu = max(mu, 0.0f);

    glm::vec3 norm(0.0f, 0.0f, 0.0f);
    for(int i=0; i<_numNeighs; i++)
    {
        int nextNeigh = ((i+1)%_numNeighs);
        glm::vec3 neighVert = _verts[_oneRingNeigh[i]];
        glm::vec3 nextNeighVert = _verts[_oneRingNeigh[nextNeigh]];
        glm::vec3 faceNorm = glm::cross(neighVert - _deformedVert, nextNeighVert-_deformedVert);
        norm += faceNorm;
    }
    norm = glm::normalize(norm);

    glm::vec3 sumWeightedCentroid(0.0f);
    for(int i=0; i<_numNeighs; i++)
    {
        glm::vec3 neighVert = _verts[_oneRingNeigh[i]];
        glm::vec3 projNeighVert = ProjectPointOnToPlane(neighVert, _deformedVert, norm);//_normal);
        float barycentricCoord = _centroidWeights[i];
        sumWeightedCentroid += barycentricCoord * projNeighVert;
    }

    _deformedVert = ((1.0f - mu) * _deformedVert) + (mu * sumWeightedCentroid);
}

//------------------------------------------------------------------------------------------------

__device__ void LaplacianSmoothing(glm::vec3 &_deformedVert,
                                   const glm::vec3 &_normal,
                                   const int *_oneRingNeigh,
                                   const float *_centroidWeights,
                                   const int _numNeighs,
                                   const glm::vec3 *_verts,
                                   const float _beta)
{
    glm::vec3 centroid(0.0f, 0.0f, 0.0f);

    for(int i=0; i<_numNeighs; i++)
    {
        centroid += (_centroidWeights[i] * _verts[_oneRingNeigh[i]]);
    }

    _deformedVert = ((1.0f - _beta) * _deformedVert) + (_beta * centroid);
}


//------------------------------------------------------------------------------------------------
// CUDA Global Kernels
//------------------------------------------------------------------------------------------------

__global__ void EvalGlobalField_Kernel(float *_output,
                                       const glm::vec3 *_samplePoint,
                                       const int _numSamples,
                                       const glm::mat4 *_textureSpace,
                                       const glm::mat4 *_rigidTransforms,
                                       const cudaTextureObject_t *_fieldFuncs,
                                       const int _numFields,
                                       const cudaTextureObject_t *_compOps,
                                       const cudaTextureObject_t *_theta,
                                       const int _numOps,
                                       const ComposedFieldCuda *_compFields,
                                       const int _numCompFields)
{
    int tid = threadIdx.x + (blockIdx.x * blockDim.x);

    if(tid >= _numSamples)
    {
        return;
    }

    EvalGlobalField(_output[tid], _samplePoint[tid], _numSamples,
                    _textureSpace, _rigidTransforms, _fieldFuncs, _numFields,
                    _compOps, _theta, _numOps,
                    _compFields, _numCompFields);
}

//------------------------------------------------------------------------------------------------

__global__ void EvalGradGlobalField_Kernel(float *_output,
                                           glm::vec3 *_outputG,
                                       const glm::vec3 *_samplePoint,
                                       const int _numSamples,
                                       const glm::mat4 *_textureSpace,
                                       const glm::mat4 *_rigidTransforms,
                                       const cudaTextureObject_t *_fieldFuncs,
                                       const int _numFields,
                                       const cudaTextureObject_t *_compOps,
                                       const cudaTextureObject_t *_theta,
                                       const int _numOps,
                                       const ComposedFieldCuda *_compFields,
                                       const int _numCompFields)
{
    int tid = threadIdx.x + (blockIdx.x * blockDim.x);

    if(tid < _numSamples)
    {
        float f=0.0f;
        glm::vec3 grad=glm::vec3(0.0f, 0.0f, 0.0f);
        glm::vec3 sample = _samplePoint[tid];

        EvalGradGlobalField(f, grad, sample, _numSamples,
                            _textureSpace, _rigidTransforms, _fieldFuncs, _numFields,
                            _compOps, _theta, _numOps,
                            _compFields, _numCompFields);


//        printf("%i\n",tid);



        _output[tid] = f;
        _outputG[tid] = grad;
    }
}


//------------------------------------------------------------------------------------------------

__global__ void LinearBlendWeightSkin_Kernel(glm::vec3 *_deformedVert,
                                             const glm::vec3 *_origVert,
                                             glm::vec3 *_deformedNorms,
                                             const glm::vec3 *_origNorms,
                                             const glm::mat4 *_transform,
                                             const uint *_boneId,
                                             const float *_weight,
                                             const int _numVerts,
                                             const int _numBones)
{
    int tid = threadIdx.x + (blockIdx.x * blockDim.x);

    if(tid >= _numVerts)
    {
        return;
    }

    glm::mat4 boneTransform = glm::mat4(0.0f);

    float totalWeight = 0.0f;
    for(int i=0; i<4; i++)
    {
        uint boneId = _boneId[(tid*4) + i];
        float w = _weight[(tid*4) + i];
        boneTransform += (_transform[boneId] * w);

        totalWeight+=w;
    }

    _deformedVert[tid] = glm::vec3(boneTransform * glm::vec4(_origVert[tid], 1.0f));
    _deformedNorms[tid] = glm::transpose(glm::inverse(glm::mat3(boneTransform))) * _origNorms[tid];
}

//------------------------------------------------------------------------------------------------

__global__ void VertexProjection_Kernel(glm::vec3 *_deformedVert,
                                        const glm::vec3 *_normal,
                                        const float *_origIsoValue,
                                        glm::vec3 *_prevIsoGrad,
                                        const int _numVerts,
                                        const glm::mat4 *_textureSpace,
                                        const glm::mat4 *_rigidTransforms,
                                        const cudaTextureObject_t *_fieldFuncs,
                                        const int _numFields,
                                        const cudaTextureObject_t *_compOps,
                                        const cudaTextureObject_t *_theta,
                                        const int _numOps,
                                        const ComposedFieldCuda *_compFields,
                                        const int _numCompFields,
                                        const float _sigma,
                                        const float _contactAngle)
{

    int tid = threadIdx.x + (blockIdx.x * blockDim.x);

    if(tid >= _numVerts)
    {
        return;
    }

    //----------------------------------------------------
    // Get iso value from global field
    glm::vec3 deformedVert = _deformedVert[tid];
    float origIsoValue = _origIsoValue[tid];
    glm::vec3 prevGrad = _prevIsoGrad[tid];
    glm::vec3 newGrad = glm::vec3(0.0f, 0.0f, 0.0f);
    float newIsoValue = 0.0f;

    EvalGradGlobalField(newIsoValue, newGrad, deformedVert, _numVerts,
                        _textureSpace, _rigidTransforms, _fieldFuncs, _numFields,
                        _compOps, _theta, _numOps,
                        _compFields, _numCompFields);


    //----------------------------------------------------
    // Perform vertex projection along gradient of global field
    float gradAngle;

    VertexProjection(deformedVert, origIsoValue, newIsoValue, newGrad, prevGrad, gradAngle, _sigma, _contactAngle);
    prevGrad = newGrad;

    _deformedVert[tid] = deformedVert;
    _prevIsoGrad[tid] = newGrad;

}

//------------------------------------------------------------------------------------------------

__global__ void TangentialRelaxation_Kernel(glm::vec3 *_deformedVert,
                                            const glm::vec3 *_normal,
                                            const float *_origIsoValue,
                                            glm::vec3 *_prevIsoGrad,
                                            const int _numVerts,
                                            const glm::mat4 *_textureSpace,
                                            const glm::mat4 *_rigidTransforms,
                                            const cudaTextureObject_t *_fieldFuncs,
                                            const int _numFields,
                                            const cudaTextureObject_t *_compOps,
                                            const cudaTextureObject_t *_theta,
                                            const int _numOps,
                                            const ComposedFieldCuda *_compFields,
                                            const int _numCompFields,
                                            const int *_oneRingVerts,
                                            const float *_centroidWeights,
                                            const int *_oneRingScatterAddr,
                                            const float _sigma,
                                            const float _contactAngle)
{

    int tid = threadIdx.x + (blockIdx.x * blockDim.x);

    if(tid >= _numVerts)
    {
        return;
    }


    //----------------------------------------------------
    // initialise variables
    glm::vec3 deformedVert = _deformedVert[tid];
    glm::vec3 deformedNorm = _normal[tid];
    float origIsoValue = _origIsoValue[tid];
    int startNeighAddr = _oneRingScatterAddr[tid];
    int numNeighs = _oneRingScatterAddr[tid+1] - startNeighAddr;
    glm::vec3 newGrad = glm::vec3(0.0f, 0.0f, 0.0f);
    float newIsoValue = 0.0f;


    //----------------------------------------------------
    // Get iso value from global field
    EvalGradGlobalField(newIsoValue, newGrad, deformedVert, _numVerts,
                        _textureSpace, _rigidTransforms, _fieldFuncs, _numFields,
                        _compOps, _theta, _numOps,
                        _compFields, _numCompFields);

    _prevIsoGrad[tid] = newGrad;

    //----------------------------------------------------
    // Perform Tangential Relaxation
    const int *oneRing = (_oneRingVerts + startNeighAddr);
    const float *centroid = (_centroidWeights + startNeighAddr);

    TangentialRelaxation(deformedVert, deformedNorm, origIsoValue, newIsoValue, _deformedVert, oneRing, centroid, numNeighs);


    //----------------------------------------------------
    // Update data
    _deformedVert[tid] = deformedVert;


}


//------------------------------------------------------------------------------------------------

__global__ void GenerateOneRingCentroidWeights_Kernel(glm::vec3 *d_verts,
                                                      const glm::vec3 *d_normals,
                                                      const int _numVerts,
                                                      float *_centroidWeights,
                                                      const int *_oneRingIds,
                                                      const glm::vec3 *_oneRingVerts,
                                                      const int *_numNeighsPerVert,
                                                      const int *_oneRingScatterAddr)
{

    int tid = threadIdx.x + (blockIdx.x * blockDim.x);

    if(tid >= _numVerts)
    {
        return;
    }

    //-------------------------------------------------------------------
    // Passed sanity check lets get down to business

    glm::vec3 v = d_verts[tid];
    glm::vec3 n = d_normals[tid];
    int startNeighAddr = _oneRingScatterAddr[tid];
//    int numNeighs = _numNeighsPerVert[tid];
    int numNeighs = _oneRingScatterAddr[tid+1] - startNeighAddr;


    glm::vec3 oneRingVerts[10];
    glm::vec3 q[10];
    glm::vec3 s[10];
    for(int i=0; i<numNeighs; ++i)
    {
        int neighId = startNeighAddr + i;

        oneRingVerts[i] = _oneRingVerts[neighId];
        q[i] = ProjectPointOnToPlane(oneRingVerts[i], v, n);
        s[i] = q[i] - v;

        _centroidWeights[neighId] = 0.0f;
    }


    float r[10];
    float A[10];
    float D[10];

    // Check for coords close to/on boundary of cage
    for(int i=0; i<numNeighs; ++i)
    {
        int nextI = (i+1)%numNeighs;
        int neighId = startNeighAddr + i;
        int nextNeighId = startNeighAddr + nextI;

        r[i] = glm::length(s[i]);
        glm::vec3 x = glm::cross(s[i], s[nextI]);
        A[i] = 0.5f * glm::length(x);
//        float dot = glm::dot(x, n);
//        A[i] = (dot >= 0.0f) ? A[i] : -A[i];
        D[i] = glm::dot(s[i], s[nextI]);

        if(r[i] < FLT_EPSILON)
        {
            _centroidWeights[neighId] = 1.0f;
            return;
        }
        else if(fabs(A[i]) < FLT_EPSILON && D[i] < 0.0f)
        {
            glm::vec3 dv = q[nextI] - q[i];
            float dl = glm::length(dv);
            // TODO: handle assertions dl==0
            dv = v - q[i];
            float mu = glm::length(dv) / dl;
            // TODO: handle assertions 0<=mu<=1
            _centroidWeights[neighId] = 1.0f - mu;
            _centroidWeights[nextNeighId] = mu;
            return;
        }

    }


    float tanalpha[10]; // tan(alpha/2)
    for( int i = 0; i < numNeighs; ++i)
    {
        int nextI = (i+1)%numNeighs;
        tanalpha[i] = (r[i]*r[nextI] - D[i])/(2.0*A[i]);
    }


    float w[10];
    float W = 0.0f;
    for( int i = 0; i < numNeighs; ++i)
    {
        int prevI = (numNeighs+i-1)%numNeighs; // to avoid potential negative result of % operator

        w[i] = 2.0*( tanalpha[i] + tanalpha[prevI] )/r[i];
        W += w[i];
    }


    if( fabs(W) > 0.0)
    {
        for( int i = 0; i < numNeighs; ++i)
        {
            int neighId = startNeighAddr + i;
            _centroidWeights[neighId] = w[i] / W;
        }
    }
}

#ifndef CUDATEXTURE_H
#define CUDATEXTURE_H

//--------------------------------------------------------------------------------
#include <iostream>
#include <cuda_runtime.h>
#include <cuda.h>

#include "helper_cuda.h"


//--------------------------------------------------------------------------------
/// @author Idris Miles
/// @version 1.0
/// @date 18/04/2017
//--------------------------------------------------------------------------------


/// @class Cuda3DTexture<T>
/// @brief A templated class for creating a 3D cuda texture object.
/// @brief Templates can only be float1, float2 and float4
template<typename T>
class Texture3DCuda
{
public:

    /// @brief constructor.
    Texture3DCuda();

    /// @brief Destructor.
    ~Texture3DCuda();

    /// @brief Method to create a 3D texture cudaTextureObject from host side array
    /// @param _dim : the dimensions of the 3D texture.
    /// @param _data : The host side array of data to fill 3D texture with.
    void CreateCudaTexture(unsigned int _dim, T *_data, cudaTextureFilterMode _filterMode = cudaFilterModePoint);

    /// @brief Methot to get the cudaTextureObject_t for use within kernels.
    /// @return cudaTextureObject_t
    cudaTextureObject_t &GetCudaTextureObject();


private:

    //------------------------------------------------------------------------------------------------
    /// @brief Method to destroy cuda 3D texture.
    void DeleteCudaTexture();

    //------------------------------------------------------------------------------------------------

    /// @brief Attribute to determine if cuda 3D texture has been created yet.
    bool m_init;

    /// @brief cudaArray attribute, host side array is copied into cudaArray before it is bound to texture object.
    cudaArray *d_cuArray;

    /// @brief cudaTextureObject attribute that we can pass to CUDA kernels.
    cudaTextureObject_t d_cuTex;

};




//------------------------------------------------------------------------------------------------
// Implementation
//------------------------------------------------------------------------------------------------

template<typename T>
Texture3DCuda<T>::Texture3DCuda()
{
    m_init = false;
}

//------------------------------------------------------------------------------------------------

template<typename T>
Texture3DCuda<T>::~Texture3DCuda()
{
    DeleteCudaTexture();
}

//------------------------------------------------------------------------------------------------

template<typename T>
void Texture3DCuda<T>::CreateCudaTexture(unsigned int _dim, T *_data, cudaTextureFilterMode _filterMode)
{
    // just in case it's already been created
    DeleteCudaTexture();


    // Initialise cuda array
    cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc<T>();
    checkCudaErrors(cudaMalloc3DArray(&d_cuArray, &channelDesc, make_cudaExtent(_dim, _dim, _dim)));


    // Upload host data to device array
    cudaMemcpy3DParms copy3DParams = {0};
    copy3DParams.srcPtr = make_cudaPitchedPtr((void*)_data, _dim*sizeof(T), _dim, _dim);
    copy3DParams.dstArray = d_cuArray;
    copy3DParams.extent = make_cudaExtent(_dim, _dim, _dim);
    copy3DParams.kind = cudaMemcpyHostToDevice;
    checkCudaErrors(cudaMemcpy3D(&copy3DParams));


    // Initalise cuda texture
    struct cudaResourceDesc resDesc;
    memset(&resDesc, 0, sizeof(resDesc));
    resDesc.resType = cudaResourceTypeArray;
    resDesc.res.array.array = d_cuArray;

    struct cudaTextureDesc texDesc;
    memset(&texDesc, 0, sizeof(texDesc));
    texDesc.addressMode[0] = cudaAddressModeClamp;
    texDesc.addressMode[1] = cudaAddressModeClamp;
    texDesc.addressMode[2] = cudaAddressModeClamp;
    texDesc.filterMode = _filterMode;// cudaFilterModePoint;// cudaFilterModeLinear;
    texDesc.readMode = cudaReadModeElementType;
    texDesc.normalizedCoords = 1;

    d_cuTex = 0;
    checkCudaErrors(cudaCreateTextureObject(&d_cuTex, &resDesc, &texDesc, NULL));

    m_init = true;
}

//------------------------------------------------------------------------------------------------

template<typename T>
cudaTextureObject_t &Texture3DCuda<T>::GetCudaTextureObject()
{
    if(m_init)
    {
        return d_cuTex;
    }

    return d_cuTex;
}

//------------------------------------------------------------------------------------------------

template<typename T>
void Texture3DCuda<T>::DeleteCudaTexture()
{
    if(m_init)
    {
        cudaDestroyTextureObject(d_cuTex);
        cudaFreeArray(d_cuArray);

        m_init = false;
    }
}

#endif // CUDATEXTURE_H

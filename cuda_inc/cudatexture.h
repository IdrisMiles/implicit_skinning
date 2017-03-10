#ifndef CUDATEXTURE_H
#define CUDATEXTURE_H

#include <cuda_runtime.h>
#include <cuda.h>


template<typename T>
class CudaTexture
{
public:
    CudaTexture()
    {
        m_init = false;
    }

    ~CudaTexture()
    {
        DeleteCudaTexture();
    }


    /// @brief Method to create a 3D texture cudaTextureObject from host side array
    void CreateCudaTexture(unsigned int _dim, T *_data)
    {
        // just in case it's already been created
        DeleteCudaTexture();


        // Initialise cuda array
        cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc<T>();
        cudaMalloc3DArray(&d_cuArray, &channelDesc, make_cudaExtent(_dim*sizeof(T), _dim, _dim));


        // Upload host data to device array
        cudaMemcpy3DParms copy3DParams = {0};
        copy3DParams.srcPtr = make_cudaPitchedPtr((void*)_data, _dim*sizeof(T), _dim, _dim);
        copy3DParams.dstArray = d_cuArray;
        copy3DParams.extent = make_cudaExtent(_dim, _dim, _dim);
        copy3DParams.kind = cudaMemcpyHostToDevice;
        cudaMemcpy3D(&copy3DParams);


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
        texDesc.filterMode = cudaFilterModeLinear;
        texDesc.readMode = cudaReadModeElementType;
        texDesc.normalizedCoords = 1;

        d_cuTex = 0;
        cudaCreateTextureObject(&d_cuTex, &resDesc, &texDesc, NULL);

        m_init = true;
    }

    cudaTextureObject_t &GetCudaTextureObject()
    {
        if(m_init)
        {
            return d_cuTex;
        }

        return d_cuTex;
    }

private:
    void DeleteCudaTexture()
    {
        if(m_init)
        {
            cudaDestroyTextureObject(d_cuTex);
            cudaFreeArray(d_cuArray);
        }
    }

    bool m_init;
    cudaArray *d_cuArray;
    cudaTextureObject_t d_cuTex;
};

#endif // CUDATEXTURE_H

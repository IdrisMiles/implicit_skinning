#include "../cuda_inc/cudatexture.h"

//CudaTexture::CudaTexture()
//{

//}


//CudaTexture::~CudaTexture()
//{
//    cudaDestroyTextureObject(d_cuTex);
//    cudaFreeArray(d_cuArray);
//}


//template<typename T>
//void CudaTexture::CreateCudaTexture(unsigned int _dim, T *_data)
//{
//    // Initialise cuda array
//    cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc<T>();// cudaCreateChannelDesc(32, 0, 0, 0, cudaChannelFormatKindFloat);
//    cudaMalloc3DArray(&d_cuArray, &channelDesc, make_cudaExtent(_dim*sizeof(T), _dim, _dim));


//    // Upload data to device here
//    cudaMemcpy3DParms copy3DParams = {0};
//    copy3DParams.srcPtr = make_cudaPitchedPtr((void*)_data, _dim*sizeof(T), _dim, _dim);
//    copy3DParams.dstArray = d_cuArray;
//    copy3DParams.extent = make_cudaExtent(_dim, _dim, _dim);
//    copy3DParams.kind = cudaMemcpyHostToDevice;
//    cudaMemcpy3D(&copy3DParams);


//    // Initalise cuda texture
//    struct cudaResourceDesc resDesc;
//    memset(&resDesc, 0, sizeof(resDesc));
//    resDesc.resType = cudaResourceTypeArray;
//    resDesc.res.array.array = d_cuArray;

//    struct cudaTextureDesc texDesc;
//    memset(&texDesc, 0, sizeof(texDesc));
//    texDesc.addressMode[0] = cudaAddressModeClamp;
//    texDesc.addressMode[1] = cudaAddressModeClamp;
//    texDesc.addressMode[2] = cudaAddressModeClamp;
//    texDesc.filterMode = cudaFilterModeLinear;
//    texDesc.readMode = cudaReadModeElementType;
//    texDesc.normalizedCoords = 1;

//    d_cuTex = 0;
//    cudaCreateTextureObject(&d_cuTex, &resDesc, &texDesc, NULL);
//}

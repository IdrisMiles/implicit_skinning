#ifndef COMPOSITIONOP_H
#define COMPOSITIONOP_H

#include <functional>

#include <glm/glm.hpp>

#include <cuda.h>

#include "Texture/Texture3DCuda.h"
#include "Texture/Texture3DCpu.h"


//-------------------------------------------------------------------------------
/// @author Idris Miles
/// @version 1.0
/// @date 18/04/2017
//-------------------------------------------------------------------------------


/// @class CompositionOp
/// @brief A class to operate on 2 field and compose them into a new field.
class CompositionOp
{
public:
    CompositionOp(const unsigned int _dim = 32);
    ~CompositionOp();

    void SetTheta(std::function<float(float)> _theta);
    void SetCompositionOp(std::function<float(float, float, float)> _compositionOp);
    void SetParams(float _alpha0, float _alpha1, float _alpha2,
                   float _theta0, float _theta1, float _theta2,
                   float _w0, float _w1);

    void Precompute(const unsigned int _res = 32);

    /// @brief Method to compute the result value of the composed field functions
    float Eval(const float f1, const float f2, const float d);

    /// @brief Method to map angle to a value between [0:1]
    /// Refered to as controller dc(alpha) parameter for composition operator
    /// in "Robust Iso-Surface Tracking for Interactive Character Skinning"
    float Theta(const float _angleRadians);


    cudaTextureObject_t &GetFieldFunc3DTexture();
    cudaTextureObject_t &GetThetaTexture();

private:
    std::function<float(float)> m_theta;

    std::function<float(float)> m_kTheta;

    std::function<float(float, float)> m_gHat;

    std::function<float(float, float, float)> m_compositionOp;



    Texture3DCpu<float> m_field;

    Texture3DCuda<float4> d_field;
    cudaTextureObject_t d_theta;

    float m_alpha0;
    float m_alpha1;
    float m_alpha2;

    float m_theta0;
    float m_theta1;
    float m_theta2;

    float m_w0;
    float m_w1;

    bool m_precomputed;

};

#endif // COMPOSITIONOP_H

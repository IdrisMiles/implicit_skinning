#ifndef IMPLICITSKINKERNELS_H
#define IMPLICITSKINKERNELS_H


class ImplicitSkinKernels
{
public:
    ImplicitSkinKernels();

    void Deform();

    void PerformLBWSkinning();
    void PerformVertexProjection();
    void PerformTangentialRelaxation();
    void PerformLaplacianSmoothing();
};

#endif // IMPLICITSKINKERNELS_H

#include "include/BinaryTree/interiornode.h"
#include <glm/gtx/vector_angle.hpp>
#include <iostream>

InteriorNode::InteriorNode(std::shared_ptr<CompositionOp> _compositionOp,
                           std::shared_ptr<AbstractNode> _child0,
                           std::shared_ptr<AbstractNode> _child1) :
    AbstractNode(),
    m_compositionOp(_compositionOp)
{
    m_children[0] = _child0;
    m_children[1] = _child1;
}

InteriorNode::~InteriorNode()
{
    m_children[0] = nullptr;
    m_children[1] = nullptr;
    m_compositionOp = nullptr;
}

float InteriorNode::Eval(const glm::vec3 _x)
{
    // Do a bit of error checking
    if(m_compositionOp == nullptr)
    {
        std::cout<<"Composition op null in interiornode\n";
    }


    // Do actual evaluation
    float f1 = 0.0f;
    float f2 = 0.0f;
    float d = 0.0f;
    if(m_children[0] != nullptr)
    {
        f1 = m_children[0]->Eval(_x);
    }

    if(m_children[1] != nullptr)
    {
        f2 = m_children[1]->Eval(_x);
    }

    if(m_children[0] != nullptr && m_children[1] != nullptr)
    {
        glm::vec3 g1 = m_children[0]->Grad(_x);
        glm::vec3 g2 = m_children[1]->Grad(_x);
        float angle = glm::angle(g1, g2);

        d = m_compositionOp->Theta(angle);
    }

    return m_compositionOp->Eval(f1, f2, d);
}

glm::vec3 InteriorNode::Grad(const glm::vec3 _x)
{
    return glm::vec3(0.0f, 1.0f, 0.0f);

    // Do a bit of error checking
    if(m_compositionOp == nullptr)
    {
        std::cout<<"Composition op null in interiornode\n";
        return glm::vec3(0.0f, 1.0f, 0.0f);
    }


    // Do actual evaluation
    float h= 0.01f;
    float h2 = 2.0f*h;

    float f = Eval(_x);

    float dx = (Eval(_x + glm::vec3(h, 0.0f, 0.0f)) - f) / h;
    float dy = (Eval(_x + glm::vec3(0.0f, h, 0.0f)) - f) / h;
    float dz = (Eval(_x + glm::vec3(0.0f, 0.0f, h)) - f) / h;

    return glm::vec3(dx, dy, dz);

}


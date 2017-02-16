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
    try
    {
        // Do a bit of error checking
        if(m_compositionOp == nullptr)
        {
            std::cout<<"no comp op in interior node\n";
            throw NullCompositionOpException();
        }


        // Do actual evaluation
        float f1 = 1.0f;
        float f2 = 1.0f;
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
    catch(NullCompositionOpException e)
    {
        std::cout<<e.what();
    }
}

glm::vec3 InteriorNode::Grad(const glm::vec3 _x)
{
    try
    {
        // Do a bit of error checking
        if(m_compositionOp == nullptr)
        {
            throw NullCompositionOpException();
        }


        // Do actual evaluation
        float h= 0.01f;
        float h2 = 2.0f*h;

        float dx = (Eval(_x + glm::vec3(h, 0.0f, 0.0f)) - Eval(_x + glm::vec3(-h, 0.0f, 0.0f))) / h2;
        float dy = (Eval(_x + glm::vec3(0.0f, h, 0.0f)) - Eval(_x + glm::vec3(0.0f, -h, 0.0f))) / h2;
        float dz = (Eval(_x + glm::vec3(0.0f, 0.0f, h)) - Eval(_x + glm::vec3(0.0f, 0.0f, -h))) / h2;

        return glm::vec3(dx, dy, dz);

    }
    catch(NullCompositionOpException e)
    {
        std::cout<<e.what();
    }
}


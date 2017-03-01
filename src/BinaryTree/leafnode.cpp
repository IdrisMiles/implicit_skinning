#include "include/BinaryTree/leafnode.h"
#include <iostream>

LeafNode::LeafNode(std::shared_ptr<FieldFunction> _fieldFunction) :
    AbstractNode(),
    m_fieldFunction(_fieldFunction)
{

}

LeafNode::~LeafNode()
{
    m_fieldFunction = nullptr;
}

float LeafNode::Eval(const glm::vec3 _x)
{
    if(m_fieldFunction == nullptr)
    {
        std::cout<<"No field function in leafnode\n";
        return 0.0f;
    }

    return m_fieldFunction->Eval(_x);
}

glm::vec3 LeafNode::Grad(const glm::vec3 _x)
{
    if(m_fieldFunction == nullptr)
    {
        std::cout<<"No field function in leafnode\n";
        return glm::vec3(0.0f, 1.0f, 0.0f);
    }

    return m_fieldFunction->Grad(_x);
}



void LeafNode::SetFieldFunction(std::shared_ptr<FieldFunction> _fieldFunction)
{
    m_fieldFunction = _fieldFunction;
}

void LeafNode::SetFieldFunction(FieldFunction *_fieldFunction)
{
    m_fieldFunction.reset(_fieldFunction);
}

void LeafNode::SetFieldFunction(FieldFunction &_fieldFunction)
{
    m_fieldFunction.reset(&_fieldFunction);
}




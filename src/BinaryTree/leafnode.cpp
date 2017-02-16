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
    try
    {
        if(m_fieldFunction == nullptr)
        {
            throw NullFieldFuncException();
        }

        return m_fieldFunction->Eval(_x);
    }
    catch(NullFieldFuncException e)
    {
        std::cout<<e.what();
    }
}

glm::vec3 LeafNode::Grad(const glm::vec3 _x)
{
    try
    {
        if(m_fieldFunction == nullptr)
        {
            throw NullFieldFuncException();
        }


        return m_fieldFunction->Grad(_x);
    }
    catch(NullFieldFuncException e)
    {
        std::cout<<e.what();
    }
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




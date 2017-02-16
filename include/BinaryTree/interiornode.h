#ifndef INTERIORNODE_H
#define INTERIORNODE_H

#include "BinaryTree/abstractnode.h"
#include "ScalarField/compositionop.h"
#include <exception>

class InteriorNode : public AbstractNode
{
public:
    InteriorNode(std::shared_ptr<CompositionOp> _compositionOp,
                 std::shared_ptr<AbstractNode> _child0,
                 std::shared_ptr<AbstractNode> _child1);
    virtual ~InteriorNode();

    virtual float Eval(const glm::vec3 _x);
    virtual glm::vec3 Grad(const glm::vec3 _x);

private:
    std::shared_ptr<AbstractNode> m_children[2];
    std::shared_ptr<CompositionOp> m_compositionOp;




    class NullCompositionOpException : public std::exception
    {
    public:
      virtual const char* what() const throw()
      {
        return "Null Composition Operator in Leaf Node \n";
      }
    };
};

#endif // INTERIORNODE_H

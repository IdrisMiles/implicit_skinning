#ifndef LEAFNODE_H
#define LEAFNODE_H

#include "BinaryTree/abstractnode.h"
#include "ScalarField/fieldfunction.h"
#include <exception>


class LeafNode : public AbstractNode
{
public:
    LeafNode(std::shared_ptr<FieldFunction> _fieldFunction = nullptr);
    virtual ~LeafNode();

    virtual float Eval(const glm::vec3 _x);
    virtual glm::vec3 Grad(const glm::vec3 _x);

    void SetFieldFunction(std::shared_ptr<FieldFunction> _fieldFunction);

private:
    std::shared_ptr<FieldFunction> m_fieldFunction;




    class NullFieldFuncException : public std::exception
    {
    public:
      virtual const char* what() const throw()
      {
        return "Null Field Function in Leaf Node \n";
      }
    };

};

#endif // LEAFNODE_H

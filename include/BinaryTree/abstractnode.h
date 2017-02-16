#ifndef ABSTRACTNODE_H
#define ABSTRACTNODE_H

#include <memory>
#include <glm/glm.hpp>

class AbstractNode
{
public:
    AbstractNode(){}
    virtual ~AbstractNode(){}

    virtual float Eval(const glm::vec3 _x) = 0;
    virtual glm::vec3 Grad(const glm::vec3 _x) = 0;
};

#endif // ABSTRACTNODE_H

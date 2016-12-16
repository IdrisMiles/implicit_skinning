#ifndef BONEANIM_H
#define BONEANIM_H

#include <vector>
#include <string>

#include <glm/glm.hpp>
#include <glm/gtc/quaternion.hpp>

struct PosAnim
{
    float time;
    glm::vec3 pos;

    PosAnim(){}

    PosAnim(float _time, glm::vec3 _pos) : 
    time(_time),
    pos(_pos)
    {
    }
};

struct ScaleAnim
{
    float time;
    glm::vec3 scale;

    ScaleAnim(){}

    ScaleAnim(float _time, glm::vec3 _scale) : 
    time(_time),
    scale(_scale)
    {
    }
};

struct RotAnim
{
    float time;
    glm::quat rot;

    RotAnim(){}

    RotAnim(float _time, glm::quat _rot) : 
    time(_time),
    rot(_rot)
    {
    }
};

class BoneAnim
{
public:
    BoneAnim(){}

    std::string m_name;
    std::vector<PosAnim> m_posAnim;
    std::vector<ScaleAnim> m_scaleAnim;
    std::vector<RotAnim> m_rotAnim;

};

#endif // BONEANIM_H

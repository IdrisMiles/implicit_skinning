#ifndef BONEANIM_H
#define BONEANIM_H

#include <vector>
#include <string>

#include <glm/glm.hpp>
#include <glm/gtc/quaternion.hpp>


/// @author Idris Miles
/// @version 1.0


/// @struct PosAnim
/// @brief Structure to hold positional animation for a single keyframe.
struct PosAnim
{
    /// @brief Default constructor
    PosAnim(){}

    /// @brief Constructor
    PosAnim(float _time, glm::vec3 _pos) :
        time(_time),
        pos(_pos)
    {
    }

    /// @brief time stamp for this frame
    float time;

    /// @brief position as this keyframe
    glm::vec3 pos;
};


/// @struct ScaleAnim
/// @brief Structure to hold scaling animation for a single keyframe.
struct ScaleAnim
{
    /// @brief Default constructor
    ScaleAnim(){}

    /// @brief Constructor
    ScaleAnim(float _time, glm::vec3 _scale) : 
        time(_time),
        scale(_scale)
    {
    }

    /// @brief time stamp for this frame
    float time;

    /// @brief Scale for this keyframe
    glm::vec3 scale;
};


/// @struct RotAnim
/// @brief Structure to hold rotational animation for a single keyframe.
struct RotAnim
{
    /// @brief Default constructor
    RotAnim(){}

    /// @brief Constructor
    RotAnim(float _time, glm::quat _rot) : 
        time(_time),
        rot(_rot)
    {
    }

    /// @brief time stamp for this frame
    float time;

    /// @brief rotation for this keyframe
    glm::quat rot;
};


/// @struct BoneAnim
/// @brief Structure to hold animation for a single bone.
class BoneAnim
{
public:
    /// @brief Default constructor
    BoneAnim(){}

    std::string m_name;
    std::vector<PosAnim> m_posAnim;
    std::vector<ScaleAnim> m_scaleAnim;
    std::vector<RotAnim> m_rotAnim;

};

#endif // BONEANIM_H

#ifndef OPENGLSCENE_H
#define OPENGLSCENE_H

//-------------------------------------------------------------------------------
#include <GL/glew.h>

#include <QOpenGLWidget>
#include <QOpenGLContext>
#include <QOpenGLShaderProgram>
#include <QOpenGLVertexArrayObject>
#include <QOpenGLBuffer>
#include <QTimer>
#include <QKeyEvent>

#include <glm/glm.hpp>
#include <glm/gtx/transform.hpp>

#include <vector>
#include "Model/model.h"


//-------------------------------------------------------------------------------
/// @author Idris Miles
/// @version 1.0
/// @date 18/04/2017
//-------------------------------------------------------------------------------



/// @class OpenGLScene
/// @brief This class is iniherited from QOpenGLWidget and acts as our scene.
/// This is based on https://github.com/IdrisMiles/QtOpenGL
class OpenGLScene : public QOpenGLWidget
{

    Q_OBJECT

public:
    /// @brief constructor
    OpenGLScene(QWidget *parent = 0);

    /// @brief destructor
    ~OpenGLScene();

    /// @brief Method to add a model into the scene
    /// @param _modelField : File to load into the scene
    std::shared_ptr<Model> AddModel(const std::string &_modelFile);


public slots:
    /// @brief set X rotation of world
    void setXRotation(int angle);

    /// @brief set Y rotation of world
    void setYRotation(int angle);

    /// @brief set Z rotation of world
    void setZRotation(int angle);

    /// @brief set X translation of world
    void setXTranslation(int x);

    /// @brief set Y translation of world
    void setYTranslation(int y);

    /// @brief set Z translation of world
    void setZTranslation(int z);

    /// @brief clean up scene
    void cleanup();

    /// @brief update animation of models
    void UpdateAnim();

    /// @brief calls update
    void UpdateDraw();

signals:

protected:
    /// @brief overloaded method from QOpenGLWidget, initialises GL stuff and starts animation/drawing timers
    void initializeGL() Q_DECL_OVERRIDE;

    /// @brief overloaded method from QOpenGLWidget, draw our scene
    void paintGL() Q_DECL_OVERRIDE;

    /// @brief overloaded method from QOpenGLWidget, updates projection matrix
    void resizeGL(int width, int height) Q_DECL_OVERRIDE;

    /// @brief overloaded method from QOpenGLWidget, updates the m_lastPos
    void mousePressEvent(QMouseEvent *event) Q_DECL_OVERRIDE;

    /// @brief overloaded method from QOpenGLWidget,
    void mouseMoveEvent(QMouseEvent *event) Q_DECL_OVERRIDE;

    /// @brief overloaded method from QOpenGLWidget,
    void keyPressEvent(QKeyEvent *event) Q_DECL_OVERRIDE;


private:

    int m_xRot;
    int m_yRot;
    int m_zRot;
    int m_xDis;
    int m_yDis;
    int m_zDis;
    QPoint m_lastPos;
    glm::vec3 m_lightPos;

    bool m_initGL;
    QTimer *m_drawTimer;
    QTimer *m_animTimer;
    float m_animTime;

    glm::mat4 m_projMat;
    glm::mat4 m_viewMat;
    glm::mat4 m_modelMat;

    std::vector<std::shared_ptr<Model>> m_models;


};

#endif // OPENGLSCENE_H

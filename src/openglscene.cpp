#include "include/openglscene.h"
#include <iostream>
#include <QMouseEvent>
#include <math.h>

#include "modelloader.h"

OpenGLScene::OpenGLScene(QWidget *parent) : QOpenGLWidget(parent),
    m_xRot(0),
    m_yRot(180.0f*16.0f),
    m_zRot(0),
    m_xDis(0),
    m_yDis(0),
    m_zDis(1500)
{
    QSurfaceFormat format;
    format.setVersion(4, 3);
    format.setProfile(QSurfaceFormat::CoreProfile);
    format.setDepthBufferSize(24);
    format.setStencilBufferSize(8);
    setFormat(format);

    setFocus();
    setFocusPolicy(Qt::StrongFocus);

    m_initGL = false;
    m_animTime = 0.0f;

    m_drawTimer = new QTimer(this);
    connect(m_drawTimer, &QTimer::timeout, this, &OpenGLScene::UpdateDraw);

    m_animTimer = new QTimer(this);
    connect(m_animTimer, &QTimer::timeout, this, &OpenGLScene::UpdateAnim);

}


OpenGLScene::~OpenGLScene()
{
    cleanup();
}

void OpenGLScene::AddModel(const std::string &_modelFile)
{

    if(m_initGL)
    {
        makeCurrent();
        m_models.push_back(std::shared_ptr<Model>(ModelLoader::LoadModel(_modelFile)));
        m_models.back()->Initialise();
        doneCurrent();
        update();
    }
}


void OpenGLScene::initializeGL()
{
    connect(context(), &QOpenGLContext::aboutToBeDestroyed, this, &OpenGLScene::cleanup);

    glewInit();

    //initializeOpenGLFunctions();
    glClearColor(0.4, 0.4, 0.4, 1);


    // Light position is fixed.
    m_lightPos = glm::vec3(0, 600, 700);

    m_initGL = true;
    m_drawTimer->start(16);
    m_animTimer->start(16);

}

void OpenGLScene::paintGL()
{
    // clean gl window
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    glEnable(GL_DEPTH_TEST);;

    // update model matrix
    m_modelMat = glm::mat4(1);
    m_modelMat = glm::translate(m_modelMat, glm::vec3(0.5f*(m_zDis/250.0f)*m_xDis, -0.5f*(m_zDis/250.0f)*m_yDis, -1.0f*m_zDis));
//    m_modelMat = glm::translate(m_modelMat, glm::vec3(0,0, -1.0f*m_zDis));
    m_modelMat = glm::rotate(m_modelMat, glm::radians(m_xRot/16.0f), glm::vec3(1,0,0));
    m_modelMat = glm::rotate(m_modelMat, glm::radians(m_yRot/16.0f), glm::vec3(0,1,0));


    //---------------------------------------------------------------------------------------
    // Draw code - replace this with project specific draw stuff
    for(auto &&model : m_models)
    {
        model->SetLightPos(m_lightPos);
        model->SetModelMatrix(m_modelMat);
        model->SetNormalMatrix(glm::inverse(glm::mat3(m_modelMat)));
        model->SetViewMatrix(m_viewMat);
        model->SetProjectionMatrix(m_projMat);

        model->DrawMesh();
        model->DrawRig();
    }

    //---------------------------------------------------------------------------------------

}


void OpenGLScene::UpdateAnim()
{
    m_animTime += 0.016f;
    for(auto &&model : m_models)
    {
        model->Animate(m_animTime);
    }
}

void OpenGLScene::UpdateDraw()
{
    update();
}

void OpenGLScene::resizeGL(int w, int h)
{
    m_projMat = glm::perspective(45.0f, GLfloat(w) / h, 0.1f, 5000.0f);
}

QSize OpenGLScene::minimumSizeHint() const
{
    return QSize(50, 50);
}

QSize OpenGLScene::sizeHint() const
{
    return QSize(400, 400);
}

static void qNormalizeAngle(int &angle)
{
    while (angle < 0)
        angle += 360 * 16;
    while (angle > 360 * 16)
        angle -= 360 * 16;
}

void OpenGLScene::setXTranslation(int x)
{
    if (x != m_xDis) {
        m_xDis = x;
        emit xTranslationChanged(x);
        update();
    }
}

void OpenGLScene::setYTranslation(int y)
{
    if (y != m_yDis) {
        m_yDis = y;
        emit yTranslationChanged(y);
        update();
    }
}

void OpenGLScene::setZTranslation(int z)
{
    if (z != m_zDis) {
        m_zDis= z;
        emit zTranslationChanged(z);
        update();
    }
}

void OpenGLScene::setXRotation(int angle)
{
    qNormalizeAngle(angle);
    if (angle != m_xRot) {
        m_xRot = angle;
        emit xRotationChanged(angle);
        update();
    }
}

void OpenGLScene::setYRotation(int angle)
{
    qNormalizeAngle(angle);
    if (angle != m_yRot) {
        m_yRot = angle;
        emit yRotationChanged(angle);
        update();
    }
}

void OpenGLScene::setZRotation(int angle)
{
    qNormalizeAngle(angle);
    if (angle != m_zRot) {
        m_zRot = angle;
        emit zRotationChanged(angle);
        update();
    }
}

void OpenGLScene::cleanup()
{
    makeCurrent();
    m_models.clear();
    doneCurrent();
}


void OpenGLScene::mousePressEvent(QMouseEvent *event)
{
    m_lastPos = event->pos();
}

void OpenGLScene::mouseMoveEvent(QMouseEvent *event)
{
    int dx = event->x() - m_lastPos.x();
    int dy = event->y() - m_lastPos.y();

    if (event->buttons() & Qt::LeftButton) {
        setXRotation(m_xRot + 8 * dy);
        setYRotation(m_yRot + 8 * dx);
    }
    else if (event->buttons() & Qt::RightButton) {
        setZTranslation(m_zDis + dy);
    }

    else if(event->buttons() & Qt::MiddleButton)
    {
        setXTranslation(m_xDis + dx);
        setYTranslation(m_yDis + dy);
    }
    m_lastPos = event->pos();
}

void OpenGLScene::keyPressEvent(QKeyEvent *event)
{
    if(event->key() == Qt::Key_W)
    {
        for(auto &&model : m_models)
        {
            model->ToggleWireframe();
        }
    }

    if(event->key() == Qt::Key_E)
    {
        for(auto &&model : m_models)
        {
            model->ToggleSkinnedSurface();
        }
    }

    if(event->key() == Qt::Key_R)
    {
        for(auto &&model : m_models)
        {
            model->ToggleIsoSurface();
        }
    }

    if(event->key() == Qt::Key_T)
    {
        for(auto &&model : m_models)
        {
            model->ToggleSkinnedImplicitSurface();
        }
    }
}

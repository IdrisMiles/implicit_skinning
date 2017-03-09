#include "mainwindow.h"
#include "ui_mainwindow.h"
#include <QFileDialog>

MainWindow::MainWindow(QWidget *parent) :
    QMainWindow(parent),
    ui(new Ui::MainWindow)
{
    ui->setupUi(this);
    m_scene = new OpenGLScene(this);
    ui->gridLayout->addWidget(m_scene, 1, 1, 1, 1);

    connect(ui->s_loadModel, &QPushButton::clicked, this, &MainWindow::LoadModel);

}

MainWindow::~MainWindow()
{
    if(ui != nullptr)
    {
        delete ui;
    }

    if(m_scene != nullptr)
    {
        delete m_scene;
    }
}


void MainWindow::LoadModel()
{
    QString file = QFileDialog::getOpenFileName(this,QString("Open File"), QString("./"), QString("3D files (*.*)"));

    if (file.isNull())
    {
        return;
    }

    m_scene->AddModel(file.toStdString());
}

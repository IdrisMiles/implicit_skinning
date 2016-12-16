#include "mainwindow.h"
#include "ui_mainwindow.h"

MainWindow::MainWindow(QWidget *parent) :
    QMainWindow(parent),
    ui(new Ui::MainWindow)
{
    ui->setupUi(this);
    m_scene = new OpenGLScene(this);
    ui->gridLayout->addWidget(m_scene, 1, 1, 1, 1);


}

MainWindow::~MainWindow()
{
    delete ui;
    delete m_scene;
}


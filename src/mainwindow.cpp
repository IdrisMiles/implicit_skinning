#include "mainwindow.h"
#include "ui_mainwindow.h"
#include <QFileDialog>

MainWindow::MainWindow(QWidget *parent) :
    QMainWindow(parent),
    ui(new Ui::MainWindow)
{
    ui->setupUi(this);
    ui->gridLayout->addWidget(ui->glScene, 0, 0, 2, 1);

    connect(ui->loadModel, &QPushButton::clicked, this, &MainWindow::LoadModel);

}

MainWindow::~MainWindow()
{
    if(ui != nullptr)
    {
        delete ui;
    }
}


void MainWindow::LoadModel()
{
    QString file = QFileDialog::getOpenFileName(this,QString("Open File"), QString("./"), QString("3D files (*.*)"));

    if (file.isNull())
    {
        return;
    }


    ui->glScene->AddModel(file.toStdString());
}

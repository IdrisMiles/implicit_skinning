#include "GUI/mainwindow.h"
#include "ui_mainwindow.h"
#include <QFileDialog>
#include "GUI/implicitskinsettings.h"

//-------------------------------------------------------------------------------

MainWindow::MainWindow(QWidget *parent) :
    QMainWindow(parent),
    ui(new Ui::MainWindow)
{
    ui->setupUi(this);

    ui->gridLayout->addWidget(ui->glScene, 0, 0, 2, 4);
    ui->gridLayout->addWidget(ui->implicitSkinSettings, 0, 4, 2, 2);

    connect(ui->implicitSkinSettings, &ImplicitSkinSettings::LoadAnimationClicked, this, [this](){
        std::string file = ui->implicitSkinSettings->GetAnimationFile();

        if(!file.empty())
        {
            ui->glScene->AddModel(file);
        }
    });

}

//-------------------------------------------------------------------------------

MainWindow::~MainWindow()
{
    if(ui != nullptr)
    {
        delete ui;
    }
}

//-------------------------------------------------------------------------------

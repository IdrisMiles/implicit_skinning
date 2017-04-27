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
            auto model = ui->glScene->AddModel(file).get();
            auto implicitDeformer = model->GetImplicitDeformer();

            connect(ui->implicitSkinSettings, &ImplicitSkinSettings::SigmaChanged, [this, implicitDeformer](float sigma){
                implicitDeformer->SetSigma(sigma);
            });

            connect(ui->implicitSkinSettings, &ImplicitSkinSettings::ContactAngleChanged, [this, implicitDeformer](float contactAngle){
                implicitDeformer->SetContactAngle(contactAngle);
            });

            connect(ui->implicitSkinSettings, &ImplicitSkinSettings::IterationsChanged, [this, implicitDeformer](float iterations){
                implicitDeformer->SetIterations(iterations);
            });


            connect(ui->implicitSkinSettings, &ImplicitSkinSettings::RenderMeshChanged, [this, model](bool checked){
               model->SetSkinnedSurface(checked);
            });

            connect(ui->implicitSkinSettings, &ImplicitSkinSettings::WireframeChanged, [this, model](bool checked){
               model->SetWireframe(checked);
            });

            connect(ui->implicitSkinSettings, &ImplicitSkinSettings::IsoSurfaceChanged, [this, model](bool checked){
               model->SetIsoSurface(checked);
            });

            connect(ui->implicitSkinSettings, &ImplicitSkinSettings::ImplicitSkinChanged, [this, model](bool checked){
               model->SetSkinnedImplicitSurface(checked);
            });

            connect(ui->implicitSkinSettings, &ImplicitSkinSettings::LBWSkinChanged, [this, model](bool checked){
               model->SetSkinnedImplicitSurface(!checked);
            });
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

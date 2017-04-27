#include "GUI/implicitskinsettings.h"
#include "ui_implicitskinsettings.h"
#include <QFileDialog>

//-------------------------------------------------------------------------------

ImplicitSkinSettings::ImplicitSkinSettings(QWidget *parent) :
    QGroupBox(parent),
    ui(new Ui::ImplicitSkinSettings)
{
    ui->setupUi(this);

    ui->browseAnimationFiles->setIcon(style()->standardIcon(QStyle::SP_DialogOpenButton));

    connect(ui->browseAnimationFiles, &QPushButton::clicked, this, [this](bool){
        QString file = QFileDialog::getOpenFileName(this,QString("Open File"), QString("./"), QString("3D files (*.*)"));

        if (file.isNull())
        {
            return;
        }

        ui->animationFileText->setText(file);

        emit BrowseAnimationClicked();
    });

    connect(ui->loadAnimation, &QPushButton::clicked, this, [this](bool){
        emit LoadAnimationClicked();
    });


    connect(ui->sigma, QOverload<double>::of(&QDoubleSpinBox::valueChanged), this, [this](double value){
        emit SigmaChanged(value);
    });

    connect(ui->contactAngle, QOverload<double>::of(&QDoubleSpinBox::valueChanged), this, [this](double value){
        emit ContactAngleChanged(value);
    });

    connect(ui->iterations, QOverload<int>::of(&QSpinBox::valueChanged), this, [this](int value){
        emit IterationsChanged(value);
    });

    connect(ui->animationFileText, &QLineEdit::textChanged, this, [this](QString text){
        ui->animationFileText->setText(text);
        emit AnimationFileChnaged(text.toStdString());
    });


    connect(ui->renderMesh, &QCheckBox::clicked, this, [this](bool checked){
        emit RenderMeshChanged(checked);
    });

    connect(ui->wireframe, &QCheckBox::clicked, this, [this](bool checked){
        emit WireframeChanged(checked);
    });

    connect(ui->isoSurface, &QCheckBox::clicked, this, [this](bool checked){
        emit IsoSurfaceChanged(checked);
    });

    connect(ui->implicitSkin, &QRadioButton::clicked, this, [this](bool checked){
        emit ImplicitSkinChanged(checked);
    });

    connect(ui->lbwSkin, &QRadioButton::clicked, this, [this](bool checked){
        emit LBWSkinChanged(checked);
    });
}

//-------------------------------------------------------------------------------

ImplicitSkinSettings::~ImplicitSkinSettings()
{
    delete ui;
}

//-------------------------------------------------------------------------------

std::string ImplicitSkinSettings::GetAnimationFile()
{
    return ui->animationFileText->text().toStdString();
}

//-------------------------------------------------------------------------------

int ImplicitSkinSettings::GetIterations()
{
    ui->iterations->value();
}

//-------------------------------------------------------------------------------

double ImplicitSkinSettings::GetSigma()
{
    ui->sigma->value();
}

//-------------------------------------------------------------------------------

double ImplicitSkinSettings::GetContactAngle()
{
    return ui->contactAngle->value();
}
//-------------------------------------------------------------------------------

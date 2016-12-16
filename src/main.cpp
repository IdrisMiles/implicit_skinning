
#include <iostream>

#include "mainwindow.h"

#include <QApplication>
#include <QWindow>

int main(int argc, char* argv[])
{


    // create an OpenGL format specifier
    QSurfaceFormat format;
    format.setVersion(4, 3);
    format.setProfile(QSurfaceFormat::CoreProfile);
    format.setDepthBufferSize(24);
    format.setStencilBufferSize(8);
    QSurfaceFormat::setDefaultFormat(format);


    QApplication a(argc, argv);
    MainWindow w;
    w.show();


    return a.exec();

}

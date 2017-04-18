#ifndef MAINWINDOW_H
#define MAINWINDOW_H

//-------------------------------------------------------------------------------
#include <QMainWindow>
#include <QPushButton>

#include "openglscene.h"


//-------------------------------------------------------------------------------
/// @author Idris Miles
/// @version 1.0
/// @date 18/04/2017
//-------------------------------------------------------------------------------


namespace Ui {
class MainWindow;
}

class MainWindow : public QMainWindow
{
    Q_OBJECT

public:
    explicit MainWindow(QWidget *parent = 0);
    ~MainWindow();

public slots:
    void LoadModel();


private:
    Ui::MainWindow *ui;
    OpenGLScene *m_scene;
};

#endif // MAINWINDOW_H

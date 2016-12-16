#ifndef MAINWINDOW_H
#define MAINWINDOW_H


// Qt includes
#include <QMainWindow>

#include "openglscene.h"
#include <QPushButton>


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

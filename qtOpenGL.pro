
QT       += core gui opengl

greaterThan(QT_MAJOR_VERSION, 4): QT += widgets

TARGET = ElasticImplicitSkinning

DESTDIR = ./bin

TEMPLATE = app

CONFIG += console c++11



SOURCES +=  $$PWD/src/*.cpp \
            $$PWD/src/machingcube/*.cpp

HEADERS  += $$PWD/include/*.h \
            $$PWD/include/hrbf/*.h \
            $$PWD/include/machingcube/*.h

OTHER_FILES += shader/*



INCLUDEPATH +=  $$PWD/include \
                /usr/local/include \
                /usr/include \
                /home/idris/dev/include \
                /home/idris/dev/eigen/include/eigen3

LIBS += -L/usr/local/lib -L/usr/lib -lGL -lGLU -lGLEW \
        -L${HOME}/dev/lib -L/usr/local/lib -lassimp


OBJECTS_DIR = ./obj

MOC_DIR = ./moc

FORMS    += ./form/mainwindow.ui

UI_DIR += ./ui



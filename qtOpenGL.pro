
QT       += core gui opengl

greaterThan(QT_MAJOR_VERSION, 4): QT += widgets

TARGET = ElasticImplicitSkinning

DESTDIR = ./bin

TEMPLATE = app

CONFIG += console c++11


#QMAKE_CFLAGS+=-pg
#QMAKE_CXXFLAGS+=-pg
#QMAKE_LFLAGS+=-pg

#QMAKE_CXXFLAGS_DEBUG *= -pg
#QMAKE_LFLAGS_DEBUG *= -pg


SOURCES +=  src/*.cpp \
            src/ScalarField/*.cpp \
            src/Machingcube/*.cpp \
            src/MeshSampler/*.cpp \
            src/BinaryTree/*.cpp

HEADERS  += include/*.h \
            include/ScalarField/*.h \
            include/ScalarField/Hrbf/*.h \
            include/Machingcube/*.h \
            include/MeshSampler/*.h \
            include/BinaryTree/*.h

OTHER_FILES += shader/*



INCLUDEPATH +=  $$PWD/include \
                /usr/local/include \
                /usr/include \
                /home/idris/dev/include \
                /home/idris/dev/eigen/include/eigen3 \
                /usr/local/include/eigen3/

LIBS += -L/usr/local/lib -L/usr/lib -lGL -lGLU -lGLEW \
        -L${HOME}/dev/lib -L/usr/local/lib -lassimp


OBJECTS_DIR = ./obj

MOC_DIR = ./moc

FORMS    += ./form/mainwindow.ui

UI_DIR += ./ui



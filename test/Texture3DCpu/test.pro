
QT       -= core gui


TARGET = TestTexture3DCpu

DESTDIR = ../bin

TEMPLATE = app

CONFIG += console c++11

QMAKE_CXXFLAGS += -std=c++11 -g

SOURCES += main.cpp

HEADERS +=  *.h                                     \
            ../../include/Texture/Texture3DCpu.h

INCLUDEPATH +=  ../../include                       \
                /usr/local/include                  \
                /usr/include

LIBS += -L/usr/local/lib -L/usr/lib -lgtest


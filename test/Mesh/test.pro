
QT       += core gui

greaterThan(QT_MAJOR_VERSION, 4): QT += widgets

TARGET = TestMesh

DESTDIR = ../bin

TEMPLATE = app

CONFIG += console c++11

QMAKE_CXXFLAGS += -std=c++11 -g

SOURCES += main.cpp

HEADERS += ../../include/mesh.h

INCLUDEPATH +=  ../../include                       \
                /usr/local/include                  \
                /usr/include

LIBS += -L/usr/local/lib -L/usr/lib -lgtest


OBJECTS_DIR = ./obj

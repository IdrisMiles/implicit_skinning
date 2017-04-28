
QT       += core gui opengl

greaterThan(QT_MAJOR_VERSION, 4): QT += widgets

TARGET = ElasticImplicitSkinning

DESTDIR = ./bin

TEMPLATE = app

CONFIG += console c++11

QMAKE_CXXFLAGS += -std=c++11 -g


#QMAKE_CFLAGS+=-pg
#QMAKE_CXXFLAGS+=-pg
#QMAKE_LFLAGS+=-pg

#QMAKE_CXXFLAGS_DEBUG *= -pg
#QMAKE_LFLAGS_DEBUG *= -pg


SOURCES +=  src/*.cpp               \
            src/ScalarField/*.cpp   \
            src/Machingcube/*.cpp   \
            src/MeshSampler/*.cpp   \
            src/Model/*.cpp         \
            src/GUI/*.cpp

HEADERS  += include/ScalarField/*.h         \
            include/ScalarField/Hrbf/*.h    \
            include/Machingcube/*.h         \
            include/MeshSampler/*.h         \
            include/Model/*.h               \
            include/GUI/*.h                 \
            include/Texture/*.h

OTHER_FILES += shader/*



INCLUDEPATH +=  $$PWD/include                       \
                /usr/local/include                  \
                /usr/include                        \
                /usr/local/include/eigen3/          \
                /home/idris/dev/include             \
                /home/idris/dev/eigen/include/eigen3

LIBS += -L/usr/local/lib -L/usr/lib -lGL -lGLU -lGLEW \
        -L${HOME}/dev/lib -L/usr/local/lib -lassimp


OBJECTS_DIR = ./obj

MOC_DIR = ./moc

FORMS    += ./form/*

UI_DIR += ./ui


#--------------------------------------------------------------------------
# CUDA stuff
#--------------------------------------------------------------------------

HEADERS += $$PWD/cuda_inc/*.*h

INCLUDEPATH +=  ./cuda_inc \
                ./include
CUDA_SOURCES += ./cuda_src/*.cu
CUDA_PATH = /usr
NVCC = $$CUDA_PATH/bin/nvcc

SYSTEM_NAME = unix
SYSTEM_TYPE = 64
GENCODE_FLAGS += -arch=sm_50
NVCC_OPTIONS = -std=c++11 -ccbin g++ --use_fast_math --compiler-options -fno-strict-aliasing --ptxas-options=-v #-G -g#-rdc=true -Xptxas -O1

# include paths
INCLUDEPATH += $(CUDA_PATH)/include $(CUDA_PATH)/include/cuda

# library directories
QMAKE_LIBDIR += $$CUDA_PATH/lib/x86_64-linux-gnu $(CUDA_PATH)/include/cuda

CUDA_OBJECTS_DIR = $$PWD/cuda_obj

# The following makes sure all path names (which often include spaces) are put between quotation marks
CUDA_INC = $$join(INCLUDEPATH,' -I','-I','')
LIBS += -lcudart -lcurand #-lcudadevrt

cuda.input = CUDA_SOURCES
cuda.output = $$CUDA_OBJECTS_DIR/${QMAKE_FILE_BASE}_cuda.o
cuda.commands = $$NVCC -m$$SYSTEM_TYPE $$GENCODE_FLAGS -c -o ${QMAKE_FILE_OUT} ${QMAKE_FILE_NAME} $$NVCC_OPTIONS $$CUDA_INC
cuda.dependency_type = TYPE_C
QMAKE_EXTRA_COMPILERS += cuda

QT       += core gui opengl

greaterThan(QT_MAJOR_VERSION, 4): QT += widgets

CONFIG += c++11

# The following define makes your compiler emit warnings if you use
# any Qt feature that has been marked deprecated (the exact warnings
# depend on your compiler). Please consult the documentation of the
# deprecated API in order to know how to port your code away from it.
DEFINES += QT_DEPRECATED_WARNINGS

# You can also make your code fail to compile if it uses deprecated APIs.
# In order to do so, uncomment the following line.
# You can also select to disable deprecated APIs only up to a certain version of Qt.
#DEFINES += QT_DISABLE_DEPRECATED_BEFORE=0x060000    # disables all the APIs deprecated before Qt 6.0.0

SOURCES += \
    main.cpp \
    widget.cpp

HEADERS += \
    widget.h

FORMS += \
    widget.ui

# Default rules for deployment.
qnx: target.path = /tmp/$${TARGET}/bin
else: unix:!android: target.path = /opt/$${TARGET}/bin
!isEmpty(target.path): INSTALLS += target

RESOURCES += \
    res.qrc



win32:LIBS += -lOpengl32 \
                         -lglu32

unix:LIBS += -lglut -lGLU


#eigen函数库
INCLUDEPATH += $$PWD/eigen-3.4.0




#切换为静态库
QMAKE_CXXFLAGS_DEBUG += /MTd
QMAKE_CXXFLAGS_RELEASE += /MT

#win32:
LIBS += -L$$PWD/bullet3_3_25api/lib/ -lBulletCollision_vs2010_x64_debug
LIBS += -L$$PWD/bullet3_3_25api/lib/ -lBulletDynamics_vs2010_x64_debug
LIBS += -L$$PWD/bullet3_3_25api/lib/ -lBulletSoftBody_vs2010_x64_debug
LIBS += -L$$PWD/bullet3_3_25api/lib/ -lLinearMath_vs2010_x64_debug
LIBS += -L$$PWD/bullet3_3_25api/lib/ -lConvexDecomposition_vs2010_x64_debug


INCLUDEPATH += $$PWD/bullet3_3_25api/include
DEPENDPATH += $$PWD/bullet3_3_25api/include

#ifndef WIDGET_H
#define WIDGET_H

#define BT_THREADSAFE 1
#define BT_USE_DOUBLE_PRECISION
#define _DEBUG 1

#include <QOpenGLWidget>
#include <QOpenGLExtraFunctions>
#include <QOpenGLFunctions_3_3_Core>
#include <QOpenGLFunctions>
#include <QOpenGLShaderProgram>
#include <QOpenGLVertexArrayObject>
#include <QOpenGLBuffer>
#include <QOpenGLFramebufferObject>

#include <QVector>
#include <QStringList>
#include <QTimer>
#include <QTime>
#include <QMatrix4x4>
#include <QMatrix3x3>
#include <QQuaternion>
#include <QVector2D>


#include "Eigen/Dense"

#include <QTimer>
#include <QWidget>


#include "btBulletDynamicsCommon.h"
//#include "btBulletCollisionCommon.h"
//#include "ConvexDecomposition.h"


using namespace std;


typedef Eigen::MatrixXd Matrix;
typedef Eigen::VectorXd Vector;

QT_BEGIN_NAMESPACE
namespace Ui { class Widget; }
QT_END_NAMESPACE

class Widget : public QOpenGLWidget,public QOpenGLExtraFunctions
{
    Q_OBJECT

public:
    Widget(QWidget *parent = nullptr);
    ~Widget();
    QTimer *timer;
    void angle_update(double j1_angle, double j2_angle, double j3_angle, double j4_angle, double j5_angle, double j6_angle, double j7_angle);

protected:
    virtual void initializeGL() override;
    virtual void paintGL() override;
    virtual void resizeGL(int w,int h) override;
    QVector<float> loadAscllStl(QString filename,int ratio);//文件名和放大系数
    void mousePressEvent(QMouseEvent *event) override;
    void mouseMoveEvent(QMouseEvent *event) override;
    void wheelEvent(QWheelEvent *event) override;


private slots:
    void on_pushButton_clicked();

    void on_connect_robot_clicked();

    void on_pushButton_2_clicked();

    void on_J1_valueChanged(int value);

    void on_J2_valueChanged(int value);


    void on_J3_valueChanged(int value);

    void on_J4_valueChanged(int value);

    void on_J5_valueChanged(int value);

    void on_J6_valueChanged(int value);

    void on_home_clicked();

    void on_comboBox_activated(int index);

    void on_J7_valueChanged(int value);

    void on_add_X_clicked();

    void on_sub_X_clicked();

    void on_sub_Y_clicked();

    void on_add_Y_clicked();

    void on_sub_Z_clicked();

    void on_add_Z_clicked();

    void on_sub_RX_clicked();

    void on_add_RX_clicked();

    void on_sub_RY_clicked();

    void on_add_RY_clicked();

    void on_sub_RZ_clicked();

    void on_add_RZ_clicked();

    void on_arm_angle_valueChanged(int value);

private:
    Ui::Widget *ui;
    QVector<float> li_fang_ti;
    QVector<float> J1;
    QVector<float> J2;
    QVector<float> J3;
    QVector<float> J4;
    QVector<float> J5;
    QVector<float> J6;
    QVector<float> J7;
    QVector<float> Position;
    QVector<float> Normal;//读文件时的俩个临时变量顶点位置，法向量
    QOpenGLShaderProgram *shaderprogram;
    QOpenGLShaderProgram *shaderprogramXYZ;

    QOpenGLFramebufferObject *frame_buffer;

    QOpenGLVertexArrayObject VAO;//声明VAO顶点数组对象
    QOpenGLBuffer VBO;//声明VBO数组缓冲对象

    QOpenGLVertexArrayObject J1VAO;//声明VAO顶点数组对象
    QOpenGLBuffer J1VBO;//声明VBO数组缓冲对象

    QOpenGLVertexArrayObject J2VAO;//声明VAO顶点数组对象
    QOpenGLBuffer J2VBO;//声明VBO数组缓冲对象

    QOpenGLVertexArrayObject J3VAO;//声明VAO顶点数组对象
    QOpenGLBuffer J3VBO;//声明VBO数组缓冲对象

    QOpenGLVertexArrayObject J4VAO;//声明VAO顶点数组对象
    QOpenGLBuffer J4VBO;//声明VBO数组缓冲对象

    QOpenGLVertexArrayObject J5VAO;//声明VAO顶点数组对象
    QOpenGLBuffer J5VBO;//声明VBO数组缓冲对象

    QOpenGLVertexArrayObject J6VAO;//声明VAO顶点数组对象
    QOpenGLBuffer J6VBO;//声明VBO数组缓冲对象

    QOpenGLVertexArrayObject J7VAO;//声明VAO顶点数组对象
    QOpenGLBuffer J7VBO;//声明VBO数组缓冲对象

    QOpenGLVertexArrayObject VAOXYZ;//声明VAO顶点数组对象
    QOpenGLBuffer VBOXYZ;//声明VBO数组缓冲对象


    QMatrix4x4 model, per_model, base, frame0, frame1, frame2, frame3, frame4, frame5, frame6;//base为捕捉使用
    QMatrix4x4 view;
    QMatrix4x4 projection;

    GLfloat xtrans, ytrans, ztrans, Z; // translation on x,y,z-axis
    GLdouble object_x, object_y, object_z, x,y,z, c_x,c_y,c_z;
    QVector2D mousePos;
    QQuaternion rotation;

    QVector4D p;
    bool midbutton = false;

    float scale, s0;
    float angle_j1;
    float angle_j2;
    float angle_j3;
    float angle_j4;
    float angle_j5;
    float angle_j6;
    float angle_j7;


};
#endif // WIDGET_H


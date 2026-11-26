#include "widget.h"
#include "ui_widget.h"
#include <QDebug>
#include <QFile>
#include <QStringList>
#include <QtMath>
#include <QOpenGLShaderProgram>
#include <QMouseEvent>
#include <QWheelEvent>
#include <QThread>
#include <GL/glu.h>

#include <stdio.h>

#include <QElapsedTimer>
#include <iostream>

double deg_to_rad(double x)
{
    double y = 0;
    y = M_PI / 180 * x;//π/180×角度
    return y;
}
double rad_to_deg(double x)
{
    double y = 0;
    y = 180/ M_PI  * x;//180/π×弧度
    return y;
}
void RadarrToDegarr(double *pose)
{
    pose[3] = rad_to_deg(pose[3]);
    pose[4] = rad_to_deg(pose[4]);
    pose[5] = rad_to_deg(pose[5]);
//    return pose;
}
void DegarrToRadarr(double *pose)
{
    pose[3] = deg_to_rad(pose[3]);
    pose[4] = deg_to_rad(pose[4]);
    pose[5] = deg_to_rad(pose[5]);
//    return pose;
}

//求行列式
double det(int n, double *aa)//n为阶数，aa为矩阵,求行列式
{
    if (n == 1)
        return aa[0];
    double *bb = new double[(n - 1)*(n - 1)];//创建n-1阶的代数余子式阵bb
    int mov = 0;//判断行是否移动
    double sum = 0.0;//sum为行列式的值
    for (int arow = 0; arow<n; arow++) // a的行数把矩阵a(nn)赋值到b(n-1)
    {
        for (int brow = 0; brow<n - 1; brow++)//把aa阵第一列各元素的代数余子式存到bb
        {
            mov = arow > brow ? 0 : 1; //bb中小于arow的行，同行赋值，等于的错过，大于的加一
            for (int j = 0; j<n - 1; j++)  //从aa的第二列赋值到第n列
            {
                bb[brow*(n - 1) + j] = aa[(brow + mov)*n + j + 1];
            }
        }
        int flag = (arow % 2 == 0 ? 1: -1);//因为列数为0，所以行数是偶数时候，代数余子式为1.
        sum += flag* aa[arow*n] * det(n - 1, bb);//aa第一列各元素与其代数余子式积的和即为行列式
    }
    delete[]bb;
    return sum;
}
//矩阵乘法
template<size_t arows, size_t acols, size_t brows, size_t bcols, size_t ret_row, size_t ret_bcol>
void matrix_multiplication(double (&a)[arows][acols], double (&b)[brows][bcols],  double (&ret)[ret_row][ret_bcol])
{
    if(acols != brows)
    {
        qDebug() << "矩阵无法相乘";
        return;
    }
    for(int i = 0; i < arows; i++)
    {
        for(int j = 0; j < bcols; j++)
        {
            double t = 0;
            for(int k = 0; k < acols; k++)
            {
                t = t + (a[i][k] * b[k][j]);
            }
            ret[i][j] = t;
        }
    }
}
//矩阵转置
template<size_t arows, size_t acols, size_t ret_row, size_t ret_bcol>
void matrix_transpose(double (&a)[arows][acols],  double (&ret)[ret_row][ret_bcol])
{
    for(int i = 0; i < acols; i++)
    {
        for(int j = 0; j < arows; j++)
        {
            ret[i][j] = a[j][i];
        }
    }
}

Eigen::MatrixXd getT_fromPose(double pose[6])
{
    Eigen::MatrixXd Rx(3, 3);
    Eigen::MatrixXd Ry(3, 3);
    Eigen::MatrixXd Rz(3, 3);

    Rx << 1, 0, 0, 0, cos(pose[3]), -sin(pose[3]), 0, sin(pose[3]), cos(pose[3]);
    Ry << cos(pose[4]), 0, sin(pose[4]), 0, 1, 0, -sin(pose[4]), 0, cos(pose[4]);
    Rz << cos(pose[5]), - sin(pose[5]), 0, sin(pose[5]), cos(pose[5]), 0, 0, 0, 1;

    Eigen::MatrixXd R(3, 3);
    R = Rz * Ry * Rx;

    Eigen::MatrixXd P(3, 1);
    P << pose[0], pose[1], pose[2];

    Eigen::MatrixXd T(4,4);
    T << R, P, Eigen::MatrixXd::Zero(1, 3), Eigen::MatrixXd::Identity(1,1);

    return T;
}
void getPose_fromT(Eigen::MatrixXd T, double *pose)
{
    double x = T(0, 3);
    double y = T(1, 3);
    double z = T(2, 3);
    double rx = atan2(T(2, 1), T(2, 2));
    double ry = asin(-T(2, 0));
    double rz = atan2(T(1, 0), T(0, 0));

    pose[0] = x;
    pose[1] = y;
    pose[2] = z;
    pose[3] = rx;
    pose[4] = ry;
    pose[5] = rz;

}


//坐标系点
static float coordinate[] = {//我们所准备的需要提供给openGL的顶点数据
    // 位置              // 颜色
      0.0f, 0.0f, 0.0f,    1.0f, 0.0f, 0.0f,
      200.0f, 0.0f, 0.0f,  1.0f, 0.0f, 0.0f,
      0.0f, 0.0f, 0.0f,    0.0f, 1.0f, 0.0f,
      0.0f, 200.0f, 0.0f,  0.0f, 1.0f, 0.0f,
      0.0f, 0.0f, 0.0f,    0.0f, 0.0f, 1.0f,
      0.0f, 0.0f, 200.0f,  0.0f, 0.0f, 1.0f
};

Widget::Widget(QWidget *parent)
    : QOpenGLWidget(parent)
    ,VBO(QOpenGLBuffer::VertexBuffer)
    ,xtrans(0),ytrans(0),ztrans(0),ui(new Ui::Widget)
{
    ui->setupUi(this);
    QSurfaceFormat format;
    format.setAlphaBufferSize(24);  //设置alpha缓冲大小
    format.setVersion(3,3);         //设置版本号
    format.setSamples(10);          //设置重采样次数，用于反走样

    this->setFormat(format);

    li_fang_ti = loadAscllStl(":/model/model/base.txt",1);
    J1 = loadAscllStl(":/model/model/J1.txt",1);
    J2 = loadAscllStl(":/model/model/J2.txt",1);
    J3 = loadAscllStl(":/model/model/J3.txt",1);
    J4 = loadAscllStl(":/model/model/J4.txt",1);
    J5 = loadAscllStl(":/model/model/J5.txt",1);
    J6 = loadAscllStl(":/model/model/J6.txt",1);
    J7 = loadAscllStl(":/model/model/J7.txt",1);

    angle_j1 = 0.0f;
    angle_j2 = 0.0f;
    angle_j3 = 0.0f;
    angle_j4 = 0.0f;
    angle_j5 = 0.0f;
    angle_j6 = 0.0f;
    angle_j7 = 0.0f;

    scale = 1.0f;
    s0 = 1.001f;

    x = 0;
    y = 0;
    z = 0;


//    on_connect_robot_clicked();
}

Widget::~Widget()
{
    makeCurrent();
    delete ui;
}

void iiwa_forward6(double* joints, double* pose)
{
//    qDebug() << joints[0] << "," << joints[1] << "," << joints[2] << "," << joints[3] << "," << joints[4] << "," << joints[5];

    vector<double[4][4]> t(6);
    double matirx_tmp[4][4];
    double matirx_tmp1[4][4];
    QVector<double> tmp;

//    double ai[7] = {0.0},
//            alph[7] = {0,-1.5708,1.5708,1.5708, -1.5708,-1.5708, 1.5708},
//            di[7] = {0.36,0,0.42,0,0.4,0,0.126},
//            theta[7] = {0.0};
    double ai[6] = {0.0,0.0,0.42,0.0,0.0,0.0},
           alph[6] = {0,-1.5708, 3.1415926, -1.5708,-1.5708, 1.5708},
           di[6] = {0.36,0,0.0,0.4,0,0.126},
           theta[6] = {0.0,-1.5708,-1.5708,0.0,0.0,0.0};

//SDH
    {
//    for(int i = 0; i < 7; i++)
//    {
//        t[i][0][0]=cos(joints[i]+theta[i]);  t[i][0][1]=-sin(joints[i]+theta[i])*cos(alph[i]);  t[i][0][2]=sin(joints[i]+theta[i])*sin(alph[i]);   t[i][0][3]=cos(joints[i]+theta[i])*ai[i];
//        t[i][1][0]=sin(joints[i]+theta[i]);  t[i][1][1]=cos(joints[i]+theta[i])*cos(alph[i]);   t[i][1][2]=-cos(joints[i]+theta[i])*sin(alph[i]);  t[i][1][3]=sin(joints[i]+theta[i])*ai[i];
//        t[i][2][0]=0;                        t[i][2][1]=sin(alph[i]);                           t[i][2][2]=cos(alph[i]);                           t[i][2][3]=di[i];
//        t[i][3][0]=0;                        t[i][3][1]=0;                                      t[i][3][2]=0;                                      t[i][3][3]=1;
//    }
    }

//MDH
    for(int i = 0; i < 6; i++)
    {
        t[i][0][0]=cos(joints[i]+theta[i]);               t[i][0][1]=-sin(joints[i]+theta[i]);              t[i][0][2]=0;              t[i][0][3]=ai[i];
        t[i][1][0]=sin(joints[i]+theta[i])*cos(alph[i]);  t[i][1][1]=cos(joints[i]+theta[i])*cos(alph[i]);  t[i][1][2]=-sin(alph[i]);  t[i][1][3]=-di[i]*sin(alph[i]);
        t[i][2][0]=sin(joints[i]+theta[i])*sin(alph[i]);  t[i][2][1]=cos(joints[i]+theta[i])*sin(alph[i]);  t[i][2][2]=cos(alph[i]);   t[i][2][3]=di[i]*cos(alph[i]);
        t[i][3][0]=0;                                     t[i][3][1]=0;                                     t[i][3][2]=0;              t[i][3][3]=1;
    }

//    qDebug() << t[0][0][3] << "," << t[0][1][3] << "," << t[0][2][3];
//    qDebug() << "------------------------------------------------";
//    qDebug() << t[1][0][3] << "," << t[1][1][3] << "," << t[1][2][3];
//    qDebug() << "------------------------------------------------";
//    qDebug() << t[2][0][3] << "," << t[2][1][3] << "," << t[2][2][3];
//    qDebug() << "------------------------------------------------";
//    qDebug() << t[3][0][3] << "," << t[3][1][3] << "," << t[3][2][3];
//    qDebug() << "------------------------------------------------";
//    qDebug() << t[4][0][3] << "," << t[4][1][3] << "," << t[4][2][3];
//    qDebug() << "------------------------------------------------";
//    qDebug() << t[5][0][3] << "," << t[5][1][3] << "," << t[5][2][3];
//    qDebug() << "----------------------------------------------------------------------";

    matrix_multiplication(t[0], t[1], matirx_tmp);
//    qDebug() << matirx_tmp[0][3] << "," << matirx_tmp[1][3] << "," << matirx_tmp[2][3];
//    qDebug() << "------------------------------------------------";
    matrix_multiplication(matirx_tmp, t[2], matirx_tmp1);
//    qDebug() << matirx_tmp1[0][3] << "," << matirx_tmp1[1][3] << "," << matirx_tmp1[2][3];
//    qDebug() << "------------------------------------------------";
    matrix_multiplication(matirx_tmp1, t[3], matirx_tmp);
//    qDebug() << matirx_tmp[0][3] << "," << matirx_tmp[1][3] << "," << matirx_tmp[2][3];
//    qDebug() << "------------------------------------------------";
    matrix_multiplication(matirx_tmp, t[4], matirx_tmp1);
//    qDebug() << matirx_tmp1[0][3] << "," << matirx_tmp1[1][3] << "," << matirx_tmp1[2][3];
//    qDebug() << "------------------------------------------------";
    matrix_multiplication(matirx_tmp1, t[5], matirx_tmp);
//    qDebug() << matirx_tmp[0][3] << "," << matirx_tmp[1][3] << "," << matirx_tmp[2][3];
//    qDebug() << "------------------------------------------------";
//    matrix_multiplication(matirx_tmp, t[6], matirx_tmp1);
//    matrix_multiplication(matirx_tmp1, six_sensor_to_flange, matirx_tmp);

//    matrix_transpose(matirx_tmp1, matirx_tmp);


    for(int i = 0; i < 4; i++)
    {
        for(int j = 0; j < 4; j++)
        {
            tmp.push_back(matirx_tmp[i][j]);
        }
    }
//    homogeneous2Pose(tmp.data(), pose);
//    qDebug() << tmp;

        Eigen::MatrixXd T(4,4);
        T <<    tmp[0],  tmp[1],  tmp[2], tmp[3],
                tmp[4],  tmp[5],  tmp[6], tmp[7],
                tmp[8],  tmp[9],  tmp[10], tmp[11],
                tmp[12],  tmp[13],  tmp[14], tmp[15];

    getPose_fromT(T, pose);

}
void iiwa_forward7(double* joints, double* pose)
{
//    qDebug() << joints[0] << "," << joints[1] << "," << joints[2] << "," << joints[3] << "," << joints[4] << "," << joints[5] << "," << joints[6];

    vector<double[4][4]> t(7);
    double matirx_tmp[4][4];
    double matirx_tmp1[4][4];
    QVector<double> tmp;

    double ai[7] = {0.0},
            alph[7] = {0,-1.5708,1.5708,1.5708, -1.5708,-1.5708, 1.5708},
            di[7] = {0.36,0,0.42,0,0.4,0,0.126},
            theta[7] = {0.0};


//SDH
    {
//    for(int i = 0; i < 7; i++)
//    {
//        t[i][0][0]=cos(joints[i]+theta[i]);  t[i][0][1]=-sin(joints[i]+theta[i])*cos(alph[i]);  t[i][0][2]=sin(joints[i]+theta[i])*sin(alph[i]);   t[i][0][3]=cos(joints[i]+theta[i])*ai[i];
//        t[i][1][0]=sin(joints[i]+theta[i]);  t[i][1][1]=cos(joints[i]+theta[i])*cos(alph[i]);   t[i][1][2]=-cos(joints[i]+theta[i])*sin(alph[i]);  t[i][1][3]=sin(joints[i]+theta[i])*ai[i];
//        t[i][2][0]=0;                        t[i][2][1]=sin(alph[i]);                           t[i][2][2]=cos(alph[i]);                           t[i][2][3]=di[i];
//        t[i][3][0]=0;                        t[i][3][1]=0;                                      t[i][3][2]=0;                                      t[i][3][3]=1;
//    }
    }

//MDH
    for(int i = 0; i < 7; i++)
    {
        t[i][0][0]=cos(joints[i]+theta[i]);               t[i][0][1]=-sin(joints[i]+theta[i]);              t[i][0][2]=0;              t[i][0][3]=ai[i];
        t[i][1][0]=sin(joints[i]+theta[i])*cos(alph[i]);  t[i][1][1]=cos(joints[i]+theta[i])*cos(alph[i]);  t[i][1][2]=-sin(alph[i]);  t[i][1][3]=-di[i]*sin(alph[i]);
        t[i][2][0]=sin(joints[i]+theta[i])*sin(alph[i]);  t[i][2][1]=cos(joints[i]+theta[i])*sin(alph[i]);  t[i][2][2]=cos(alph[i]);   t[i][2][3]=di[i]*cos(alph[i]);
        t[i][3][0]=0;                                     t[i][3][1]=0;                                     t[i][3][2]=0;              t[i][3][3]=1;
    }

//    qDebug() << t[0][0][3] << "," << t[0][1][3] << "," << t[0][2][3];
//    qDebug() << "------------------------------------------------";
//    qDebug() << t[1][0][3] << "," << t[1][1][3] << "," << t[1][2][3];
//    qDebug() << "------------------------------------------------";
//    qDebug() << t[2][0][3] << "," << t[2][1][3] << "," << t[2][2][3];
//    qDebug() << "------------------------------------------------";
//    qDebug() << t[3][0][3] << "," << t[3][1][3] << "," << t[3][2][3];
//    qDebug() << "------------------------------------------------";
//    qDebug() << t[4][0][3] << "," << t[4][1][3] << "," << t[4][2][3];
//    qDebug() << "------------------------------------------------";
//    qDebug() << t[5][0][3] << "," << t[5][1][3] << "," << t[5][2][3];
//    qDebug() << "----------------------------------------------------------------------";

    matrix_multiplication(t[0], t[1], matirx_tmp);
//    qDebug() << matirx_tmp[0][3] << "," << matirx_tmp[1][3] << "," << matirx_tmp[2][3];
//    qDebug() << "------------------------------------------------";
    matrix_multiplication(matirx_tmp, t[2], matirx_tmp1);
//    qDebug() << matirx_tmp1[0][3] << "," << matirx_tmp1[1][3] << "," << matirx_tmp1[2][3];
//    qDebug() << "------------------------------------------------";
    matrix_multiplication(matirx_tmp1, t[3], matirx_tmp);
//    qDebug() << matirx_tmp[0][3] << "," << matirx_tmp[1][3] << "," << matirx_tmp[2][3];
//    qDebug() << "------------------------------------------------";
    matrix_multiplication(matirx_tmp, t[4], matirx_tmp1);
//    qDebug() << matirx_tmp1[0][3] << "," << matirx_tmp1[1][3] << "," << matirx_tmp1[2][3];
//    qDebug() << "------------------------------------------------";
    matrix_multiplication(matirx_tmp1, t[5], matirx_tmp);
//    qDebug() << matirx_tmp[0][3] << "," << matirx_tmp[1][3] << "," << matirx_tmp[2][3];
//    qDebug() << "------------------------------------------------";
    matrix_multiplication(matirx_tmp, t[6], matirx_tmp1);
//    matrix_multiplication(matirx_tmp1, six_sensor_to_flange, matirx_tmp);

//    matrix_transpose(matirx_tmp1, matirx_tmp);


    for(int i = 0; i < 4; i++)
    {
        for(int j = 0; j < 4; j++)
        {
            tmp.push_back(matirx_tmp1[i][j]);
        }
    }
//    homogeneous2Pose(tmp.data(), pose);
//    qDebug() << tmp;

        Eigen::MatrixXd T(4,4);
        T <<    tmp[0],  tmp[1],  tmp[2], tmp[3],
                tmp[4],  tmp[5],  tmp[6], tmp[7],
                tmp[8],  tmp[9],  tmp[10], tmp[11],
                tmp[12],  tmp[13],  tmp[14], tmp[15];

    getPose_fromT(T, pose);

}

//三维空间任意一点绕任意轴线旋转
//n为轴线法向量（且为单位向量），pn为轴线上一点，p为输入点，angle为旋转的角度（弧度）,p_out为输出点
void LineRotation(Eigen::Vector3f& n, Eigen::Vector3f& pn, Eigen::Vector4f& p, float angle, Eigen::Vector4f& p_out)
{
//    qDebug() << "angle2:" << angle;
//	//方法一：按过程推导
//	//在XYZ坐标系下：
//	//轴线n直线方程：x = x0 + n.x * t / y = y0 + n.y * t / z = z0 + n.z * t ——公式一;
//	//p点旋转到p_out点的圆弧轨迹所在的平面方程：n.x * (x - p.x) + n.y * (y - p.y) + n.z * (z - p.z) = 0 ——公式二
//	//由公式一和公式二可得：t0 = n.x * (p.x - pn.x) + n.y * (p.y - pn.y) + n.z * (p.z - pn.z)
//	float t0 = n(0) * (p(0) - pn(0)) + n(1) * (p(1) - pn(1)) + n(2) * (p(2) - pn(2));//表示轴线与圆弧平面交点处的t（直线方程）

//	//圆弧圆心O
//	float xc = pn(0) + n(0) * t0;
//	float yc = pn(1) + n(1) * t0;
//	float zc = pn(2) + n(2) * t0;

//	//圆弧半径
//	float r = sqrt(pow((p(0) - xc), 2) + pow((p(1) - yc), 2) + pow((p(2) - zc), 2));

//	//输入的任意点P,圆心到P点的向量：OP
//	Eigen::Vector3f OP;
//	OP << (p(0) - xc) / r, (p(1) - yc) / r, (p(2) - zc) / r;

//	//建立坐标系X'Y'Z',根据右手法则计算Y'，其中X' = OP，Z' = n
//	Eigen::Vector3f Y1 = n.cross(OP);

//	//因此坐标系X'Y'Z'与坐标系XYZ的转转矩阵为
//	Eigen::Matrix3f R;
//	R << OP(0), Y1(0), n(0),
//		 OP(1), Y1(1), n(1),
//		 OP(2), Y1(2), n(2);

//	//P1(也即P旋转后的点)在坐标系X'Y'Z'下的坐标为P_temp，在XYZ坐标系下坐标为P1
//	float x_temp = r * cos(angle);
//	float y_temp = r * sin(angle);
//	float z_temp = 0;

//	//利用齐次变换，将P1在坐标系X'Y'Z'的坐标变换到XYZ的坐标
//	//P1 = T * P_temp;
//	//其中T = [R t 0 1]——4x4矩阵
//	p_out(0) = R(0, 0) * x_temp + R(0, 1) * y_temp + R(0, 2) * z_temp + xc;
//	p_out(1) = R(1, 0) * x_temp + R(1, 1) * y_temp + R(1, 2) * z_temp + yc;
//	p_out(2) = R(2, 0) * x_temp + R(2, 1) * y_temp + R(2, 2) * z_temp + zc;
//	p_out(3) = 1;
//	//std::cout << "p_out1 = " << p_out.matrix().transpose() << std::endl;

//	方法二：直接按结论
    float K = 1 - cos(angle);
    float M = n(0) * pn(0) + n(1) * pn(1) + n(2) * pn(2);

    Eigen::Matrix4f T;
    T << pow(n(0), 2)* K + cos(angle), n(0) * n(1) * K - n(2) * sin(angle), n(0) * n(2) * K + n(1) * sin(angle), (pn(0) - n(0) * M) * K + (n(2) * pn(1) - n(1) * pn(2)) * sin(angle),
        n(0) * n(1) * K + n(2) * sin(angle), pow(n(1), 2) * K + cos(angle), n(1) * n(2) * K - n(0) * sin(angle), (pn(1) - n(1) * M) * K + (n(0) * pn(2) - n(2) * pn(0)) * sin(angle),
        n(0) * n(2) * K - n(1) * sin(angle), n(1) * n(2) * K + n(0) * sin(angle), pow(n(2), 2) * K + cos(angle), (pn(2) - n(2) * M) * K + (n(1) * pn(0) - n(0) * pn(1)) * sin(angle),
        0, 0, 0, 1;

    p_out = T * p;
//    std::cout << "p_out2 = " << p_out.matrix().transpose() << std::endl;
}

double theta[8][7];//八组解，每组解六个角，第0个都不用
void iiwa_inverse6(double *pose, double *joints)
{
//    qDebug() << pose[0] << "," << pose[1] << "," << pose[2] << "," << pose[3] << "," << pose[4] << "," << pose[5];
    double ai[6] = {0.0,0.0,0.42,0.0,0.0,0.0},
           alph[6] = {0,-1.5708, 3.1415926, -1.5708,-1.5708, 1.5708},
           di[6] = {0.36,0,0.0,0.4,0,0.126},
           theta111[6] = {0.0,-1.5708,-1.5708,0.0,0.0,0.0};

    Eigen::MatrixXd T(4,4);
    T = getT_fromPose(pose);
//    cout << T << endl;

    double d6[6] = {0,0,-di[5],0,0,0};
    Eigen::MatrixXd T1(4,4);
    T1 = getT_fromPose(d6);

    Eigen::MatrixXd T2(4,4);
    T2 = T * T1;
//    cout << T2 << endl;
//    qDebug() << "P点的位置:" << T2(0,3) << "," << T2(1,3) << "," << T2(2,3);

    double j[8][6] = {{0.0}};

    j[0][0] = atan2(T2(1,3),T2(0,3));j[1][0] = j[0][0];j[2][0] = j[0][0];j[3][0] = j[0][0];
    if(j[0][0] > 0)
    {
        j[4][0] = j[0][0] - M_PI;j[5][0] = j[4][0];j[6][0] = j[4][0];j[7][0] = j[4][0];
    }
    else
    {
        j[4][0] = j[0][0] + M_PI;j[5][0] = j[4][0];j[6][0] = j[4][0];j[7][0] = j[4][0];
    }
//    qDebug() << "j1:" << rad_to_deg(j[0][0]) << "," << rad_to_deg(j[4][0]);

    if((ai[2]+di[3])-sqrt(((T2(0,3)*T2(0,3))+(T2(1,3)*T2(1,3)))+((T2(2,3)-di[0])*(T2(2,3)-di[0]))) > 0)
    {
        double c2 = ((((T2(0,3)*T2(0,3))+(T2(1,3)*T2(1,3)))+((T2(2,3)-di[0])*(T2(2,3)-di[0]))-(ai[2]*ai[2])-(di[3]*di[3]))/(2*ai[2]*di[3]));
        double s2 = sqrt(1 - (c2*c2));
        j[0][2] = atan2(-s2,c2);
        j[2][2] = atan2(s2,c2);
//        qDebug() << "j3:" << rad_to_deg(j[0][2]) << "," << rad_to_deg(-j[0][2]);
        j[1][2] = j[0][2];j[3][2] = j[2][2];
        j[4][2] = j[2][2];j[5][2] = j[2][2];j[6][2] = j[0][2];j[7][2] = j[0][2];


        //        qDebug() << "j3:" << rad_to_deg(j[0][2]) << "," << rad_to_deg(-j[0][2]);
        j[1][3] = j[0][3];j[3][3] = j[2][3];
        j[4][3] = j[2][3];j[5][3] = j[2][3];j[6][3] = j[0][3];j[7][3] = j[0][3];
    }
    else
    {
        qDebug() << "error";
        return;
    }

    j[0][1] = M_PI/2 - (atan2((T2(2,3)-di[0]), sqrt((T2(0,3)*T2(0,3))+(T2(1,3)*T2(1,3)))) +
            acos((((T2(0,3)*T2(0,3))+(T2(1,3)*T2(1,3)))+((T2(2,3)-di[0])*(T2(2,3)-di[0]))+(ai[2]*ai[2])-(di[3]*di[3]))/
            (2*ai[2]*(sqrt(((T2(0,3)*T2(0,3))+(T2(1,3)*T2(1,3)))+((T2(2,3)-di[0])*(T2(2,3)-di[0])))))));
//    qDebug() << "j2:" << rad_to_deg(j[0][1]);
    j[1][1] = j[0][1];j[4][1] = -j[0][1];j[5][1] = -j[0][1];

    j[2][1] = M_PI/2 - (atan2((T2(2,3)-di[0]), sqrt((T2(0,3)*T2(0,3))+(T2(1,3)*T2(1,3)))) -
            acos((((T2(0,3)*T2(0,3))+(T2(1,3)*T2(1,3)))+((T2(2,3)-di[0])*(T2(2,3)-di[0]))+(ai[2]*ai[2])-(di[3]*di[3]))/
            (2*ai[2]*(sqrt(((T2(0,3)*T2(0,3))+(T2(1,3)*T2(1,3)))+((T2(2,3)-di[0])*(T2(2,3)-di[0])))))));
//    qDebug() << "j2:" << rad_to_deg(j[4][1]);
    j[3][1] = j[2][1];j[6][1] = -j[2][1];j[7][1] = -j[2][1];



    vector<Eigen::Matrix4d> t(6);
    //MDH
        for(int i = 0; i < 4; i++)
        {
            t[i](0,0)=cos(j[0][i]+theta111[i]);               t[i](0,1)=-sin(j[0][i]+theta111[i]);              t[i](0,2)=0;              t[i](0,3)=ai[i];
            t[i](1,0)=sin(j[0][i]+theta111[i])*cos(alph[i]);  t[i](1,1)=cos(j[0][i]+theta111[i])*cos(alph[i]);  t[i](1,2)=-sin(alph[i]);  t[i](1,3)=-di[i]*sin(alph[i]);
            t[i](2,0)=sin(j[0][i]+theta111[i])*sin(alph[i]);  t[i](2,1)=cos(j[0][i]+theta111[i])*sin(alph[i]);  t[i](2,2)=cos(alph[i]);   t[i](2,3)=di[i]*cos(alph[i]);
            t[i](3,0)=0;                                          t[i](3,1)=0;                                          t[i](3,2)=0;              t[i](3,3)=1;
        }
        T1 = t[0] * t[1];
        T2 = T1 * t[2];
        T1 = T2 * t[3];

        T2 = T1.inverse() * T;
//        cout << T2 << endl;

    j[0][4] = atan2(sqrt((T2(2,0)*T2(2,0))+(T2(2,1)*T2(2,1))), T2(2,2));
    j[1][4] = -j[0][4];

    j[0][3] = atan2(T2(1,2)/sin(j[0][4]), T2(0,2)/sin(j[0][4]));
    j[1][3] = atan2(T2(1,2)/sin(-j[0][4]), T2(0,2)/sin(-j[0][4]));

    j[0][5] = atan2(T2(2,1)/sin(j[0][4]), -(T2(2,0)/sin(j[0][4])));
    j[1][5] = atan2(T2(2,1)/sin(-j[0][4]), -(T2(2,0)/sin(-j[0][4])));


    //-------------------------------------------------------------------------------------------------------------

    Eigen::MatrixXd T3(4,4);
    Eigen::MatrixXd T4(4,4);
    for(int i = 0; i < 4; i++)
    {
        t[i](0,0)=cos(j[2][i]+theta111[i]);               t[i](0,1)=-sin(j[2][i]+theta111[i]);              t[i](0,2)=0;              t[i](0,3)=ai[i];
        t[i](1,0)=sin(j[2][i]+theta111[i])*cos(alph[i]);  t[i](1,1)=cos(j[2][i]+theta111[i])*cos(alph[i]);  t[i](1,2)=-sin(alph[i]);  t[i](1,3)=-di[i]*sin(alph[i]);
        t[i](2,0)=sin(j[2][i]+theta111[i])*sin(alph[i]);  t[i](2,1)=cos(j[2][i]+theta111[i])*sin(alph[i]);  t[i](2,2)=cos(alph[i]);   t[i](2,3)=di[i]*cos(alph[i]);
        t[i](3,0)=0;                                          t[i](3,1)=0;                                          t[i](3,2)=0;              t[i](3,3)=1;
    }
    T3 = t[0] * t[1];
    T4 = T3 * t[2];
    T3 = T4 * t[3];

    T4 = T3.inverse() * T;
//        cout << T4 << endl;
    j[2][4] = atan2(sqrt((T4(2,0)*T4(2,0))+(T4(2,1)*T4(2,1))), T4(2,2));
    j[3][4] = -j[2][4];

    j[2][3] = atan2(T4(1,2)/sin(j[2][4]), T4(0,2)/sin(j[2][4]));
    j[3][3] = atan2(T4(1,2)/sin(-j[2][4]), T4(0,2)/sin(-j[2][4]));

    j[2][5] = atan2(T4(2,1)/sin(j[2][4]), -(T4(2,0)/sin(j[2][4])));
    j[3][5] = atan2(T4(2,1)/sin(-j[2][4]), -(T4(2,0)/sin(-j[2][4])));

    //-------------------------------------------------------------------------------------------------------------

    Eigen::MatrixXd T5(4,4);
    Eigen::MatrixXd T6(4,4);
    for(int i = 0; i < 4; i++)
    {
        t[i](0,0)=cos(j[4][i]+theta111[i]);               t[i](0,1)=-sin(j[4][i]+theta111[i]);              t[i](0,2)=0;              t[i](0,3)=ai[i];
        t[i](1,0)=sin(j[4][i]+theta111[i])*cos(alph[i]);  t[i](1,1)=cos(j[4][i]+theta111[i])*cos(alph[i]);  t[i](1,2)=-sin(alph[i]);  t[i](1,3)=-di[i]*sin(alph[i]);
        t[i](2,0)=sin(j[4][i]+theta111[i])*sin(alph[i]);  t[i](2,1)=cos(j[4][i]+theta111[i])*sin(alph[i]);  t[i](2,2)=cos(alph[i]);   t[i](2,3)=di[i]*cos(alph[i]);
        t[i](3,0)=0;                                          t[i](3,1)=0;                                          t[i](3,2)=0;              t[i](3,3)=1;
    }
    T5 = t[0] * t[1];
    T6 = T5 * t[2];
    T5 = T6 * t[3];

    T6 = T5.inverse() * T;
//        cout << T6 << endl;
    j[4][4] = atan2(sqrt((T6(2,0)*T6(2,0))+(T6(2,1)*T6(2,1))), T6(2,2));
    j[5][4] = -j[4][4];

    j[4][3] = atan2(T6(1,2)/sin(j[4][4]), T6(0,2)/sin(j[4][4]));
    j[5][3] = atan2(T6(1,2)/sin(-j[4][4]), T6(0,2)/sin(-j[4][4]));

    j[4][5] = atan2(T6(2,1)/sin(j[4][4]), -(T6(2,0)/sin(j[4][4])));
    j[5][5] = atan2(T6(2,1)/sin(-j[4][4]), -(T6(2,0)/sin(-j[4][4])));

    //-------------------------------------------------------------------------------------------------------------

    Eigen::MatrixXd T7(4,4);
    Eigen::MatrixXd T8(4,4);
    for(int i = 0; i < 4; i++)
    {
        t[i](0,0)=cos(j[6][i]+theta111[i]);               t[i](0,1)=-sin(j[6][i]+theta111[i]);              t[i](0,2)=0;              t[i](0,3)=ai[i];
        t[i](1,0)=sin(j[6][i]+theta111[i])*cos(alph[i]);  t[i](1,1)=cos(j[6][i]+theta111[i])*cos(alph[i]);  t[i](1,2)=-sin(alph[i]);  t[i](1,3)=-di[i]*sin(alph[i]);
        t[i](2,0)=sin(j[6][i]+theta111[i])*sin(alph[i]);  t[i](2,1)=cos(j[6][i]+theta111[i])*sin(alph[i]);  t[i](2,2)=cos(alph[i]);   t[i](2,3)=di[i]*cos(alph[i]);
        t[i](3,0)=0;                                          t[i](3,1)=0;                                          t[i](3,2)=0;              t[i](3,3)=1;
    }
    T7 = t[0] * t[1];
    T8 = T7 * t[2];
    T7 = T8 * t[3];

    T8 = T7.inverse() * T;
//        cout << T4 << endl;
    j[6][4] = atan2(sqrt((T8(2,0)*T8(2,0))+(T8(2,1)*T8(2,1))), T8(2,2));
    j[7][4] = -j[6][4];

    j[6][3] = atan2(T8(1,2)/sin(j[6][4]), T8(0,2)/sin(j[6][4]));
    j[7][3] = atan2(T8(1,2)/sin(-j[6][4]), T8(0,2)/sin(-j[6][4]));

    j[6][5] = atan2(T8(2,1)/sin(j[6][4]), -(T8(2,0)/sin(j[6][4])));
    j[7][5] = atan2(T8(2,1)/sin(-j[6][4]), -(T8(2,0)/sin(-j[6][4])));


    qDebug() << rad_to_deg(j[0][0]) << "," << rad_to_deg(j[0][1]) << "," << rad_to_deg(j[0][2]) << "," << rad_to_deg(j[0][3]) << "," << rad_to_deg(j[0][4]) << "," << rad_to_deg(j[0][5]);
    qDebug() << rad_to_deg(j[1][0]) << "," << rad_to_deg(j[1][1]) << "," << rad_to_deg(j[1][2]) << "," << rad_to_deg(j[1][3]) << "," << rad_to_deg(j[1][4]) << "," << rad_to_deg(j[1][5]);
    qDebug() << rad_to_deg(j[2][0]) << "," << rad_to_deg(j[2][1]) << "," << rad_to_deg(j[2][2]) << "," << rad_to_deg(j[2][3]) << "," << rad_to_deg(j[2][4]) << "," << rad_to_deg(j[2][5]);
    qDebug() << rad_to_deg(j[3][0]) << "," << rad_to_deg(j[3][1]) << "," << rad_to_deg(j[3][2]) << "," << rad_to_deg(j[3][3]) << "," << rad_to_deg(j[3][4]) << "," << rad_to_deg(j[3][5]);
    qDebug() << rad_to_deg(j[4][0]) << "," << rad_to_deg(j[4][1]) << "," << rad_to_deg(j[4][2]) << "," << rad_to_deg(j[4][3]) << "," << rad_to_deg(j[4][4]) << "," << rad_to_deg(j[4][5]);
    qDebug() << rad_to_deg(j[5][0]) << "," << rad_to_deg(j[5][1]) << "," << rad_to_deg(j[5][2]) << "," << rad_to_deg(j[5][3]) << "," << rad_to_deg(j[5][4]) << "," << rad_to_deg(j[5][5]);
    qDebug() << rad_to_deg(j[6][0]) << "," << rad_to_deg(j[6][1]) << "," << rad_to_deg(j[6][2]) << "," << rad_to_deg(j[6][3]) << "," << rad_to_deg(j[6][4]) << "," << rad_to_deg(j[6][5]);
    qDebug() << rad_to_deg(j[7][0]) << "," << rad_to_deg(j[7][1]) << "," << rad_to_deg(j[7][2]) << "," << rad_to_deg(j[7][3]) << "," << rad_to_deg(j[7][4]) << "," << rad_to_deg(j[7][5]);

    memcpy(theta, j, 384);
    qDebug() << (joints[0]) << "," << (joints[1]) << "," << (joints[2]) << ","
             << (joints[3]) << "," << (joints[4]) << "," << (joints[5]);

    for(int i = 0; i < 8; i++)
    {
        for(int k = 0; k < 6; k++)
        {
            if(abs(joints[k] - rad_to_deg(j[i][k])) > 10)
            {
                qDebug() << k << "," << abs(joints[k] - rad_to_deg(j[i][k]));
                break;
            }
            if(k == 5)
            {
                joints[0] = j[i][0];
                joints[1] = j[i][1];
                joints[2] = j[i][2];
                joints[3] = j[i][3];
                joints[4] = j[i][4];
                joints[5] = j[i][5];
                i = 10;
                qDebug() << rad_to_deg(joints[0]) << "," << rad_to_deg(joints[1]) << "," << rad_to_deg(joints[2]) << ","
                         << rad_to_deg(joints[3]) << "," << rad_to_deg(joints[4]) << "," << rad_to_deg(joints[5]);
            }
        }
        if(i == 10)
        {
            break;
        }
        if(i == 7)
        {
            qDebug() << "error1";
        }
    }

}
void iiwa_inverse7(double *pose, double *joints, double angle)
{
//    qDebug() << pose[0] << "," << pose[1] << "," << pose[2] << "," << pose[3] << "," << pose[4] << "," << pose[5];
    double ai[7] = {0.0,0.0,0.0,0.0,0.0,0.0},
           alph[7] = {0,-1.5708, 1.5708, 1.5708, -1.5708,-1.5708, 1.5708},
           di[7] = {0.36,0,0.42,0,0.4,0,0.126},
           theta111[7] = {0.0,0,0,0.0,0.0,0.0};

    Eigen::MatrixXd T(4,4);
    T = getT_fromPose(pose);
//    cout << T << endl;

    double d6[6] = {0,0,-di[6],0,0,0};
    Eigen::MatrixXd T1(4,4);
    T1 = getT_fromPose(d6);

    Eigen::MatrixXd T2(4,4), TP(4,4);
    TP = T * T1;
    T2 = TP;
//    cout << TP << endl;
//    qDebug() << u8"P点的位置:" << TP(0,3) << "," << TP(1,3) << "," << TP(2,3);

    double j[8][7] = {{0.0}};


    j[0][0] = atan2(T2(1,3),T2(0,3));
//    qDebug() << "j1:" << rad_to_deg(j[0][0]);

    if((di[2]+di[4])-sqrt(((T2(0,3)*T2(0,3))+(T2(1,3)*T2(1,3)))+((T2(2,3)-di[0])*(T2(2,3)-di[0]))) > 0)
    {
        double c3 = ((((T2(0,3)*T2(0,3))+(T2(1,3)*T2(1,3)))+((T2(2,3)-di[0])*(T2(2,3)-di[0]))-(di[2]*di[2])-(di[4]*di[4]))/(2*di[2]*di[4]));
        double s3 = sqrt(1 - (c3*c3));
        j[0][2] = atan2(-s3,c3);j[0][3] = j[0][2];j[1][3] = j[0][2];
        j[2][2] = atan2(s3,c3); j[2][3] = j[0][2];j[3][3] = j[0][2];
//        qDebug() << "j4:" << rad_to_deg(j[0][2]) << "," << rad_to_deg(j[2][2]);
    }
    else
    {
        qDebug() << "error";
        return;
    }

    double c2 = (((T2(0,3)*T2(0,3))+(T2(1,3)*T2(1,3)))+((T2(2,3)-di[0])*(T2(2,3)-di[0]))+(di[2]*di[2])-(di[4]*di[4]))/
    (2*di[2]*(sqrt(((T2(0,3)*T2(0,3))+(T2(1,3)*T2(1,3)))+((T2(2,3)-di[0])*(T2(2,3)-di[0])))));
    double s2 = sqrt(1 - (c2*c2));
    j[0][1] = M_PI/2 - (atan2((T2(2,3)-di[0]), sqrt((T2(0,3)*T2(0,3))+(T2(1,3)*T2(1,3)))) + atan2(s2,c2));
//    qDebug() << "j2:" << rad_to_deg(j[0][1]);

    j[2][1] = M_PI/2 - (atan2((T2(2,3)-di[0]), sqrt((T2(0,3)*T2(0,3))+(T2(1,3)*T2(1,3)))) - atan2(s2,c2));
//    qDebug() << "j2:" << rad_to_deg(j[2][1]);

    vector<Eigen::Matrix4d> t(6);
    //MDH
        for(int i = 0; i < 3; i++)
        {
            t[i](0,0)=cos(j[0][i]+theta111[i]);               t[i](0,1)=-sin(j[0][i]+theta111[i]);              t[i](0,2)=0;              t[i](0,3)=ai[i];
            t[i](1,0)=sin(j[0][i]+theta111[i])*cos(alph[i]);  t[i](1,1)=cos(j[0][i]+theta111[i])*cos(alph[i]);  t[i](1,2)=-sin(alph[i]);  t[i](1,3)=-di[i]*sin(alph[i]);
            t[i](2,0)=sin(j[0][i]+theta111[i])*sin(alph[i]);  t[i](2,1)=cos(j[0][i]+theta111[i])*sin(alph[i]);  t[i](2,2)=cos(alph[i]);   t[i](2,3)=di[i]*cos(alph[i]);
            t[i](3,0)=0;                                          t[i](3,1)=0;                                          t[i](3,2)=0;              t[i](3,3)=1;
        }
        T1 = t[0] * t[1];
        T2 = T1 * t[2];
//        cout << "0度肩部位置:"<< T2 << endl;

    Eigen::Vector3f n;
    n(0) = TP(0,3) - 0;n(1) = TP(1,3) - 0;n(2) = TP(2,3) - di[0];
    double mo = sqrt((n(0)*n(0)) + (n(1)*n(1)) + (n(2)*n(2)));
    n(0) = n(0)/mo;n(1) = n(1)/mo;n(2) = n(2)/mo;
//    cout << n << endl;


    Eigen::Vector3f pn(TP(0,3),TP(1,3),TP(2,3));
    Eigen::Vector4f p(T2(0,3),T2(1,3),T2(2,3),1);
    Eigen::Vector4f p_out;
    LineRotation(n, pn, p, angle, p_out);

    j[0][0] = atan2(p_out(1),p_out(0));j[1][0] = j[0][0];
    if(j[0][0] > 0)
    {
        j[2][0] = j[0][0] - M_PI;j[3][0] = j[2][0];
    }
    else
    {
        j[2][0] = j[0][0] + M_PI;j[3][0] = j[2][0];
    }
//    qDebug() << "j1:" << rad_to_deg(j[0][0]) << "," << rad_to_deg(j[2][0]);

    j[0][1] = atan2(sqrt((p_out(1)*p_out(1))+(p_out(0)*p_out(0))),p_out(2)-di[0]);
    j[1][1] = j[0][1];j[2][1] = -j[0][1];j[3][1] = -j[0][1];
//    qDebug() << "j2:" << rad_to_deg(j[0][1]) << "," << rad_to_deg(j[2][1]);

    j[0][2] = 0;j[2][2] = 0;j[1][2] = 0;j[3][2] = 0;
    //MDH
    for(int k = 0; k < 2; k++)
    {
        for(int i = 0; i < 3; i++)
        {
            t[i](0,0)=cos(j[k*2][i]+theta111[i]);               t[i](0,1)=-sin(j[k*2][i]+theta111[i]);              t[i](0,2)=0;              t[i](0,3)=ai[i];
            t[i](1,0)=sin(j[k*2][i]+theta111[i])*cos(alph[i]);  t[i](1,1)=cos(j[k*2][i]+theta111[i])*cos(alph[i]);  t[i](1,2)=-sin(alph[i]);  t[i](1,3)=-di[i]*sin(alph[i]);
            t[i](2,0)=sin(j[k*2][i]+theta111[i])*sin(alph[i]);  t[i](2,1)=cos(j[k*2][i]+theta111[i])*sin(alph[i]);  t[i](2,2)=cos(alph[i]);   t[i](2,3)=di[i]*cos(alph[i]);
            t[i](3,0)=0;                                          t[i](3,1)=0;                                          t[i](3,2)=0;              t[i](3,3)=1;
        }
        T1 = t[0] * t[1];
        T2 = T1 * t[2];
        T1 = T2.inverse() * TP;
//            cout << "法兰相对于三轴位置："  << T1 << endl;

        j[k*2][2] = atan2(T1(1,3),T1(0,3));j[2*k+1][2] = j[k*2][2];
//        qDebug() << "j3:" << rad_to_deg(j[k*2][2]);
    }
    //-------------------------------------------------------------------------------------------------------------

    j[0][4] = 0;
    Eigen::MatrixXd T7(4,4);
    Eigen::MatrixXd T8(4,4);
    for(int i = 0; i < 5; i++)
    {
        t[i](0,0)=cos(j[0][i]+theta111[i]);               t[i](0,1)=-sin(j[0][i]+theta111[i]);              t[i](0,2)=0;              t[i](0,3)=ai[i];
        t[i](1,0)=sin(j[0][i]+theta111[i])*cos(alph[i]);  t[i](1,1)=cos(j[0][i]+theta111[i])*cos(alph[i]);  t[i](1,2)=-sin(alph[i]);  t[i](1,3)=-di[i]*sin(alph[i]);
        t[i](2,0)=sin(j[0][i]+theta111[i])*sin(alph[i]);  t[i](2,1)=cos(j[0][i]+theta111[i])*sin(alph[i]);  t[i](2,2)=cos(alph[i]);   t[i](2,3)=di[i]*cos(alph[i]);
        t[i](3,0)=0;                                          t[i](3,1)=0;                                          t[i](3,2)=0;              t[i](3,3)=1;
//        cout << t[i] << endl;
    }
    T7 = t[0] * t[1];
    T8 = T7 * t[2];
    T7 = T8 * t[3];
    T8 = T7 * t[4];
//    cout << "腕部位置："  << T8 << endl;
    T7 = T8.inverse() * T;
//    cout << "腕部位置角度误差："  << T7 << endl;
    j[0][5] = atan2(sqrt((T7(2,0)*T7(2,0))+(T7(2,1)*T7(2,1))), T7(2,2));
    j[1][5] = -j[0][5];j[2][5] = j[0][5];j[3][5] = -j[0][5];

    j[0][4] = atan2(T7(1,2)/sin(j[0][5]), T7(0,2)/sin(j[0][5]));
    j[1][4] = atan2(T7(1,2)/sin(-j[0][5]), T7(0,2)/sin(-j[0][5]));
    j[2][4] = j[0][4];j[3][4] = j[1][4];

    j[0][6] = atan2(T7(2,1)/sin(j[0][5]), -(T7(2,0)/sin(j[0][5])));
    j[1][6] = atan2(T7(2,1)/sin(-j[0][5]), -(T7(2,0)/sin(-j[0][5])));
    j[2][6] = j[0][6];j[3][6] = j[1][6];

//    qDebug() << rad_to_deg(j[0][0]) << "," << rad_to_deg(j[0][1]) << "," << rad_to_deg(j[0][2]) << "," << rad_to_deg(j[0][3]) << ","
//             << rad_to_deg(j[0][4]) << "," << rad_to_deg(j[0][5]) << "," << rad_to_deg(j[0][6]);
//    qDebug() << rad_to_deg(j[1][0]) << "," << rad_to_deg(j[1][1]) << "," << rad_to_deg(j[1][2]) << "," << rad_to_deg(j[1][3]) << ","
//             << rad_to_deg(j[1][4]) << "," << rad_to_deg(j[1][5]) << "," << rad_to_deg(j[1][6]);
//    qDebug() << rad_to_deg(j[2][0]) << "," << rad_to_deg(j[2][1]) << "," << rad_to_deg(j[2][2]) << "," << rad_to_deg(j[2][3]) << ","
//             << rad_to_deg(j[2][4]) << "," << rad_to_deg(j[2][5]) << "," << rad_to_deg(j[2][6]);
//    qDebug() << rad_to_deg(j[3][0]) << "," << rad_to_deg(j[3][1]) << "," << rad_to_deg(j[3][2]) << "," << rad_to_deg(j[3][3]) << ","
//             << rad_to_deg(j[3][4]) << "," << rad_to_deg(j[3][5]) << "," << rad_to_deg(j[3][6]);

//    joints[0] = rad_to_deg(j[0][0]);
//    joints[1] = rad_to_deg(j[0][1]);
//    joints[2] = rad_to_deg(j[0][2]);
//    joints[3] = rad_to_deg(j[0][3]);
//    joints[4] = rad_to_deg(j[0][4]);
//    joints[5] = rad_to_deg(j[0][5]);
//    joints[6] = rad_to_deg(j[0][6]);


    memcpy(theta, j, 448);


//    qDebug() << (joints[0]) << "," << (joints[1]) << "," << (joints[2]) << ","
//             << (joints[3]) << "," << (joints[4]) << "," << (joints[5]) << "," << (joints[6]) << "aaa";

    for(int i = 0; i < 4; i++)
    {
        for(int k = 0; k < 7; k++)
        {
            if(abs(joints[k] - rad_to_deg(j[i][k])) > 30)
            {
                qDebug() << k << "," << rad_to_deg(j[i][k]) << "bbb";
                break;
            }
            if(k == 6)
            {
                joints[0] = rad_to_deg(j[i][0]);
                joints[1] = rad_to_deg(j[i][1]);
                joints[2] = rad_to_deg(j[i][2]);
                joints[3] = rad_to_deg(j[i][3]);
                joints[4] = rad_to_deg(j[i][4]);
                joints[5] = rad_to_deg(j[i][5]);
                joints[6] = rad_to_deg(j[i][6]);
                i = 10;
//                qDebug() << (joints[0]) << "," << (joints[1]) << "," << (joints[2]) << ","
//                         << (joints[3]) << "," << (joints[4]) << "," << (joints[5]) << ","
//                         << (joints[6]) << "ccc";
            }
        }
        if(i == 10)
        {
            break;
        }
        if(i == 3)
        {
            qDebug() << "inverse fail";
        }
    }

}


void Widget::angle_update(double j1_angle, double j2_angle, double j3_angle, double j4_angle, double j5_angle, double j6_angle, double j7_angle)
{
    angle_j1 = j1_angle;
    angle_j2 = j2_angle;
    angle_j3 = j3_angle;
    angle_j4 = j4_angle;
    angle_j5 = j5_angle;
    angle_j6 = j6_angle;
    angle_j7 = j7_angle;
//    qDebug() << angle_j1;
    update();
}

QVector<float> Widget::loadAscllStl(QString filename,int ratio)
{
    QVector<float> vertices_temp;
//   qDebug() << "load text file:" << filename;

  QFile file(filename);
  if (!file.open(QIODevice::ReadOnly | QIODevice::Text))
  {
      qDebug() << "Open stl_file failed." << endl;
  }
  while (!file.atEnd())
  {
      QString line = file.readLine().trimmed(); // trimmed去除了开头和结尾的空白字符串
      QStringList  words = line.split(' ', QString::SkipEmptyParts);

      if (words[0] == "facet")
      {
          Normal = {ratio*words[2].toFloat(), ratio*words[3].toFloat(),ratio*words[4].toFloat()};
      }
      else if (words[0] == "vertex")
      {
          Position = {ratio*words[1].toFloat(), ratio*words[2].toFloat(),ratio*words[3].toFloat()};
          vertices_temp.append(Position);
          vertices_temp.append(Normal);
      }
      else
      {
          continue;
      }

  }

//    qDebug() << "write vertice_temp success!" << filename;

//    qDebug() << vertices_temp.length() << "11111";
    file.close();
    return vertices_temp;
}

void Widget::initializeGL()
{

    this->initializeOpenGLFunctions();//初始化opengl函数

    this->glEnable(GL_DEPTH_TEST);
//    this->glDepthFunc(GL_LESS);

//    glDepthMask(GL_TRUE);
//    glDepthFunc(GL_ALWAYS); // Change this to whatever kind of depth testing you want
//    glDepthRange(0.0f, 1.0f);

    shaderprogram = new QOpenGLShaderProgram;//生成着色器程序
    shaderprogram->bind();
    if(!shaderprogram->addShaderFromSourceFile(QOpenGLShader::Vertex,":/shaders/stl.vert")){
        qDebug()<<"ERROR:"<<shaderprogram->log();    //如果编译出错,打印报错信息
    }
    if(!shaderprogram->addShaderFromSourceFile(QOpenGLShader::Fragment,":/shaders/stl.frag")){
        qDebug()<<"ERROR:"<<shaderprogram->log();    //如果编译出错,打印报错信息
    }
    //将添加到此程序的着色器与addshader链接在一起
    if(!shaderprogram->link()){
        qDebug()<<"ERROR:"<<shaderprogram->log();    //如果链接出错,打印报错信息
    }
    //    QOpenGLVertexArrayObject::Binder{&VAO};

    shaderprogramXYZ = new QOpenGLShaderProgram;//生成着色器程序
    if(!shaderprogramXYZ->addShaderFromSourceFile(QOpenGLShader::Vertex,":/shaders/CoordinateX.vert")){
        qDebug()<<"ERROR:"<<shaderprogramXYZ->log();    //如果编译出错,打印报错信息
    }
    if(!shaderprogramXYZ->addShaderFromSourceFile(QOpenGLShader::Fragment,":/shaders/CoordinateX.frag")){
        qDebug()<<"ERROR:"<<shaderprogramXYZ->log();    //如果编译出错,打印报错信息
    }
    //将添加到此程序的着色器与addshader链接在一起
    if(!shaderprogramXYZ->link()){
        qDebug()<<"ERROR:"<<shaderprogramXYZ->log();    //如果链接出错,打印报错信息
    }

{
    VAO.create();// 创建一个VAO对象，OpenGL会给它（顶点数组缓存对象）分配一个id
    VAO.bind();//将RC中的当前顶点数组缓存对象Id设置为VAO的id
    VBO.create();
    VBO.bind();
    VBO.allocate(li_fang_ti.data(),sizeof(float)*li_fang_ti.size());//将顶点数据分配到VBO中，第一个参数为数据指针，第二个参数为数据的字节长度
    shaderprogram->setAttributeBuffer(0, GL_FLOAT, 0, 3, sizeof(GLfloat) * 6);
    shaderprogram->enableAttributeArray(0);
    shaderprogram->setAttributeBuffer(1, GL_FLOAT,sizeof(GLfloat) * 3, 3, sizeof(GLfloat) * 6);
    shaderprogram->enableAttributeArray(1);
    VAO.release();//释放
    VBO.release();
    shaderprogram->release();


    J1VAO.create();// 创建一个VAO对象，OpenGL会给它（顶点数组缓存对象）分配一个id
    J1VAO.bind();//将RC中的当前顶点数组缓存对象Id设置为VAO的id
    J1VBO.create();
    J1VBO.bind();
    J1VBO.allocate(J1.data(),sizeof(float)*J1.size());//将顶点数据分配到VBO中，第一个参数为数据指针，第二个参数为数据的字节长度
    shaderprogram->setAttributeBuffer(0, GL_FLOAT, 0, 3, sizeof(GLfloat) * 6);
    shaderprogram->enableAttributeArray(0);
    shaderprogram->setAttributeBuffer(1, GL_FLOAT,sizeof(GLfloat) * 3, 3, sizeof(GLfloat) * 6);
    shaderprogram->enableAttributeArray(1);
    J1VAO.release();//释放
    J1VBO.release();

    J2VAO.create();// 创建一个VAO对象，OpenGL会给它（顶点数组缓存对象）分配一个id
    J2VAO.bind();//将RC中的当前顶点数组缓存对象Id设置为VAO的id
    J2VBO.create();
    J2VBO.bind();
    J2VBO.allocate(J2.data(),sizeof(float)*J2.size());//将顶点数据分配到VBO中，第一个参数为数据指针，第二个参数为数据的字节长度
    shaderprogram->setAttributeBuffer(0, GL_FLOAT, 0, 3, sizeof(GLfloat) * 6);
    shaderprogram->enableAttributeArray(0);
    shaderprogram->setAttributeBuffer(1, GL_FLOAT,sizeof(GLfloat) * 3, 3, sizeof(GLfloat) * 6);
    shaderprogram->enableAttributeArray(1);
    J2VAO.release();//释放
    J2VBO.release();

    J3VAO.create();// 创建一个VAO对象，OpenGL会给它（顶点数组缓存对象）分配一个id
    J3VAO.bind();//将RC中的当前顶点数组缓存对象Id设置为VAO的id
    J3VBO.create();
    J3VBO.bind();
    J3VBO.allocate(J3.data(),sizeof(float)*J3.size());//将顶点数据分配到VBO中，第一个参数为数据指针，第二个参数为数据的字节长度
    shaderprogram->setAttributeBuffer(0, GL_FLOAT, 0, 3, sizeof(GLfloat) * 6);
    shaderprogram->enableAttributeArray(0);
    shaderprogram->setAttributeBuffer(1, GL_FLOAT,sizeof(GLfloat) * 3, 3, sizeof(GLfloat) * 6);
    shaderprogram->enableAttributeArray(1);
    J3VAO.release();//释放
    J3VBO.release();

    J4VAO.create();// 创建一个VAO对象，OpenGL会给它（顶点数组缓存对象）分配一个id
    J4VAO.bind();//将RC中的当前顶点数组缓存对象Id设置为VAO的id
    J4VBO.create();
    J4VBO.bind();
    J4VBO.allocate(J4.data(),sizeof(float)*J4.size());//将顶点数据分配到VBO中，第一个参数为数据指针，第二个参数为数据的字节长度
    shaderprogram->setAttributeBuffer(0, GL_FLOAT, 0, 3, sizeof(GLfloat) * 6);
    shaderprogram->enableAttributeArray(0);
    shaderprogram->setAttributeBuffer(1, GL_FLOAT,sizeof(GLfloat) * 3, 3, sizeof(GLfloat) * 6);
    shaderprogram->enableAttributeArray(1);
    J4VAO.release();//释放
    J4VBO.release();

    J5VAO.create();// 创建一个VAO对象，OpenGL会给它（顶点数组缓存对象）分配一个id
    J5VAO.bind();//将RC中的当前顶点数组缓存对象Id设置为VAO的id
    J5VBO.create();
    J5VBO.bind();
    J5VBO.allocate(J5.data(),sizeof(float)*J5.size());//将顶点数据分配到VBO中，第一个参数为数据指针，第二个参数为数据的字节长度
    shaderprogram->setAttributeBuffer(0, GL_FLOAT, 0, 3, sizeof(GLfloat) * 6);
    shaderprogram->enableAttributeArray(0);
    shaderprogram->setAttributeBuffer(1, GL_FLOAT,sizeof(GLfloat) * 3, 3, sizeof(GLfloat) * 6);
    shaderprogram->enableAttributeArray(1);
    J5VAO.release();//释放
    J5VBO.release();

    J6VAO.create();// 创建一个VAO对象，OpenGL会给它（顶点数组缓存对象）分配一个id
    J6VAO.bind();//将RC中的当前顶点数组缓存对象Id设置为VAO的id
    J6VBO.create();
    J6VBO.bind();
    J6VBO.allocate(J6.data(),sizeof(float)*J6.size());//将顶点数据分配到VBO中，第一个参数为数据指针，第二个参数为数据的字节长度
    shaderprogram->setAttributeBuffer(0, GL_FLOAT, 0, 3, sizeof(GLfloat) * 6);
    shaderprogram->enableAttributeArray(0);
    shaderprogram->setAttributeBuffer(1, GL_FLOAT,sizeof(GLfloat) * 3, 3, sizeof(GLfloat) * 6);
    shaderprogram->enableAttributeArray(1);
    J6VAO.release();//释放
    J6VBO.release();

    J7VAO.create();// 创建一个VAO对象，OpenGL会给它（顶点数组缓存对象）分配一个id
    J7VAO.bind();//将RC中的当前顶点数组缓存对象Id设置为VAO的id
    J7VBO.create();
    J7VBO.bind();
    J7VBO.allocate(J7.data(),sizeof(float)*J7.size());//将顶点数据分配到VBO中，第一个参数为数据指针，第二个参数为数据的字节长度
    shaderprogram->setAttributeBuffer(0, GL_FLOAT, 0, 3, sizeof(GLfloat) * 6);
    shaderprogram->enableAttributeArray(0);
    shaderprogram->setAttributeBuffer(1, GL_FLOAT,sizeof(GLfloat) * 3, 3, sizeof(GLfloat) * 6);
    shaderprogram->enableAttributeArray(1);
    J7VAO.release();//释放
    J7VBO.release();


    VAOXYZ.create();// 创建一个VAO对象，OpenGL会给它（顶点数组缓存对象）分配一个id
    VAOXYZ.bind();//将RC中的当前顶点数组缓存对象Id设置为VAO的id
    VBOXYZ.create();
    VBOXYZ.bind();
    VBOXYZ.allocate(coordinate,sizeof(coordinate));//将顶点数据分配到VBO中，第一个参数为数据指针，第二个参数为数据的字节长度
    //向顶点着色器传递其中定义为"aPos"的变量所需的数据
    shaderprogramXYZ->setAttributeBuffer(0, GL_FLOAT, 0, 3, sizeof(GLfloat) * 6);
    shaderprogramXYZ->enableAttributeArray(0);
    //向顶点着色器传递其中定义为"aColor"的变量所需的数据
    shaderprogramXYZ->setAttributeBuffer(1, GL_FLOAT, sizeof(GLfloat) * 3, 3, sizeof(GLfloat) * 6);
    shaderprogramXYZ->enableAttributeArray(1);
    VAOXYZ.release();//释放
    VBOXYZ.release();
}


    view.setToIdentity();
    view.lookAt(QVector3D(0.0f, 0.0f, 0.0f), QVector3D(0.0f,0.0f,0.0f), QVector3D(0.0f,1.0f,0.0f));
}
void Widget::resizeGL(int w,int h)
{
//    this->glViewport(0,0,w,h);

    projection.setToIdentity();

//    projection.perspective(60, (float)w / h, 0.001, 2000);
//    view.setToIdentity();
//    view.lookAt(QVector3D(0, 0, 2000), QVector3D(0, 0, 0), QVector3D(0, 1, 0));

    projection.ortho(-1000, 1000, -1000, 1000, -1000, 1000);
//  view.setToIdentity();
//  view.lookAt(QVector3D(0.0f, 0.0f, 0.0f), QVector3D(0.0f,0.0f,0.0f), QVector3D(0.0f,1.0f,0.0f));
    //lookAt的位置指的是距离平截头体XYZ边缘的距离,不是距离世界坐标的距离，这个会影响深度值的获取


    if(w > h)
    {
        w = h;
        resize(w,h);
    }
    else
    {
        h = w;
        resize(w,h);
    }


//    shaderprogram->setUniformValue("projection", projection);
//    shaderprogram->setUniformValue("view", view);

    //必须加上此行，否则窗口尺寸更新后无法绘制图形

    frame_buffer = nullptr;
    delete frame_buffer;
    frame_buffer = new QOpenGLFramebufferObject(w, h, QOpenGLFramebufferObject::Depth);

}

void Widget::paintGL()
{
    frame_buffer->bind();//绑定自定义帧缓冲

//    //检查帧缓冲的完成性
//    if(this->glCheckFramebufferStatus(GL_FRAMEBUFFER) == GL_FRAMEBUFFER_COMPLETE)
//    {        qDebug() << "aaaa";    }    else    {        qDebug() << "bbbb";    }

    ui->J1->setValue(angle_j1*1000);
    ui->J2->setValue(angle_j2*1000);
    ui->J3->setValue(angle_j3*1000);
    ui->J4->setValue(angle_j4*1000);
    ui->J5->setValue(angle_j5*1000);
    ui->J6->setValue(angle_j6*1000);
    ui->J7->setValue(angle_j7*1000);
    ui->J1_value->setNum(angle_j1);
    ui->J2_value->setNum(angle_j2);
    ui->J3_value->setNum(angle_j3);
    ui->J4_value->setNum(angle_j4);
    ui->J5_value->setNum(angle_j5);
    ui->J6_value->setNum(angle_j6);
    ui->J7_value->setNum(angle_j7);

//    qDebug() << "paintGL" << angle_j1 << "," << angle_j2 << "," << angle_j3 << "," << angle_j4 << "," << angle_j5 << "," << angle_j6;
//    double joints[6] = {deg_to_rad(angle_j1), deg_to_rad(angle_j2), deg_to_rad(angle_j4), deg_to_rad(angle_j5),
//                        deg_to_rad(angle_j6), deg_to_rad(angle_j7)};
//    double pose[6] = {0};
//    iiwa_forward6(joints, pose);

//    ui->x_value->setText(QString::number(pose[0]*1000, 'f', 2));
//    ui->y_value->setText(QString::number(pose[1]*1000, 'f', 2));
//    ui->z_value->setText(QString::number(pose[2]*1000, 'f', 2));
//    ui->rx_value->setText(QString::number(rad_to_deg(pose[3]), 'f', 2));
//    ui->ry_value->setText(QString::number(rad_to_deg(pose[4]), 'f', 2));
//    ui->rz_value->setText(QString::number(rad_to_deg(pose[5]), 'f', 2));

//    ui->X->setValue(pose[0]*1000);
//    ui->Y->setValue(pose[1]*1000);
//    ui->Z->setValue(pose[2]*1000);
//    ui->RX->setValue(rad_to_deg(pose[3]));
//    ui->RY->setValue(rad_to_deg(pose[4]));
//    ui->RZ->setValue(rad_to_deg(pose[5]));


    glEnable(GL_DEPTH_TEST);
    this->glClearColor(0.0f,0.0f,0.0f,1.0f);//设置清屏颜色
    this->glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);//清空颜色和深度缓冲区

    {
        shaderprogram->bind();
        //将此着色器程序绑定到活动的qopenglcontext，并使其成为当前着色器程序。任何先前绑定的着色器程序都将被释放
        //成功绑定返回ture,反之，返回false.

        QVector3D lightColor(1.0f,1.0f,1.0f);
        //    QVector3D objectColor(1.0f,0.5f,0.31f);
        QVector3D objectColor(1.0f,1.0f,1.0f);
        QVector3D lightPos(0.0f,0.0f,10000.0f);

        shaderprogram->setUniformValue("objectColor",objectColor);
        shaderprogram->setUniformValue("lightColor",lightColor);
        shaderprogram->setUniformValue("lightPos", lightPos);


        model.setToIdentity();


    if(midbutton)
    {
        model.translate(xtrans, ytrans, ztrans);
        model.rotate(rotation);
//        qDebug() << "xtrans: " <<  xtrans << " ytrans: " <<  ytrans << " ztrans: " <<  ztrans;
        midbutton = false;
    }
    else
    {
        if((object_z > -999 && object_z < -0.001)||(object_z < 999 && object_z > 0.001))
        {
            model.translate(object_x, object_y, object_z);
            model.rotate(rotation);
            QMatrix4x4 bbb = model.inverted() * per_model;
            p = bbb.column(3);
            model.translate(p.x(), p.y(), p.z());
//            qDebug() << "p_x: " <<  p.x() << " p_y: " <<  p.y() << " p_z: " <<  p.z();
            x = object_x;
            y = object_y;
            z = object_z;
            object_x = 0;
            object_y = 0;
            object_z = 0;
//            qDebug() << "=================================";
        }
        else
        {
//            qDebug() << "per_modelx: " <<  per_model.column(3).x() << " per_modely: " <<  per_model.column(3).y() << " per_modelz: " <<  per_model.column(3).z();
            model.translate(x, y, z);//需要在鼠标滚轮函数操作xyz
            model.rotate(rotation);
            model.translate(p.x(), p.y(), p.z());

//            qDebug() << "x: " <<  x << " y: " <<  y << " z: " <<  z;

//            qDebug() << "modelx: " <<  model.column(3).x() << " modely: " <<  model.column(3).y() << " modelz: " <<  model.column(3).z();
            xtrans = model.column(3).x();
            ytrans = model.column(3).y();
            ztrans = model.column(3).z();
        }

    }

        per_model = model;//记住上一次的矩阵

        model.scale(scale, scale, scale);

        shaderprogram->setUniformValue("view", view);
        shaderprogram->setUniformValue("projection", projection);
        shaderprogram->setUniformValue("model", model);
        shaderprogram->release();

        base = model;//base为捕捉使用

        shaderprogramXYZ->bind();
        shaderprogramXYZ->setUniformValue("view", view);
        shaderprogramXYZ->setUniformValue("projection", projection);
        shaderprogramXYZ->setUniformValue("model", model);
        QOpenGLVertexArrayObject::Binder bindXYZ(&VAOXYZ);//绑定VAO
        this->glDrawArrays(GL_LINES, 0, 2);
        this->glDrawArrays(GL_LINES, 2, 2);
        this->glDrawArrays(GL_LINES, 4, 2);
        bindXYZ.release();
        shaderprogramXYZ->release();


        shaderprogram->bind();
//        shaderprogram->setUniformValue("model", model);

        int n = li_fang_ti.capacity()/sizeof(float);
        QOpenGLVertexArrayObject::Binder bind(&VAO);//绑定VAO
        this->glDrawArrays(GL_TRIANGLES,0,n);
        bind.release();

        model.rotate(angle_j1, 0.0f, 0.0f, 1.0f);
        shaderprogram->setUniformValue("model", model);

        n = J1.capacity()/sizeof(float);
        QOpenGLVertexArrayObject::Binder bind1(&J1VAO);//绑定VAO
        this->glDrawArrays(GL_TRIANGLES,0,n);
        bind1.release();

        frame0 = model;//用于碰撞检测

        model.translate(0.0f, 0.0f, 360.0f);
        model.rotate(-90, 1.0f, 0.0f, 0.0f);
        model.rotate(angle_j2, 0.0f, 0.0f, 1.0f);
        shaderprogram->setUniformValue("model", model);

        n = J2.capacity()/sizeof(float);
        QOpenGLVertexArrayObject::Binder bind2(&J2VAO);//绑定VAO
        this->glDrawArrays(GL_TRIANGLES,0,n);
        bind2.release();

        frame1 = model;//用于碰撞检测

        model.rotate(90, 1.0f, 0.0f, 0.0f);
        model.rotate(angle_j3, 0.0f, 0.0f, 1.0f);
        shaderprogram->setUniformValue("model", model);

        n = J3.capacity()/sizeof(float);
        QOpenGLVertexArrayObject::Binder bind3(&J3VAO);//绑定VAO
        this->glDrawArrays(GL_TRIANGLES,0,n);
        bind3.release();

        frame2 = model;//用于碰撞检测

        model.translate(0.0f, 0.0f, 420.0f);
        model.rotate(90, 1.0f, 0.0f, 0.0f);
        model.rotate(angle_j4, 0.0f, 0.0f, 1.0f);
        shaderprogram->setUniformValue("model", model);

        n = J4.capacity()/sizeof(float);
        QOpenGLVertexArrayObject::Binder bind4(&J4VAO);//绑定VAO
        this->glDrawArrays(GL_TRIANGLES,0,n);
        bind4.release();

        frame3 = model;//用于碰撞检测

        model.rotate(-90, 1.0f, 0.0f, 0.0f);
        model.rotate(angle_j5, 0.0f, 0.0f, 1.0f);
        shaderprogram->setUniformValue("model", model);

        n = J5.capacity()/sizeof(float);
        QOpenGLVertexArrayObject::Binder bind5(&J5VAO);//绑定VAO
        this->glDrawArrays(GL_TRIANGLES,0,n);
        bind5.release();

        frame4 = model;//用于碰撞检测

        model.translate(0.0f, 0.0f, 400.0f);
        model.rotate(-90, 1.0f, 0.0f, 0.0f);
        model.rotate(angle_j6, 0.0f, 0.0f, 1.0f);
        shaderprogram->setUniformValue("model", model);

        n = J6.capacity()/sizeof(float);
        QOpenGLVertexArrayObject::Binder bind6(&J6VAO);//绑定VAO
        this->glDrawArrays(GL_TRIANGLES,0,n);
        bind6.release();


        frame5 = model;//用于碰撞检测

        model.rotate(90, 1.0f, 0.0f, 0.0f);
        model.translate(0.0f, 0.0f, 126.0f);
        model.rotate(angle_j7, 0.0f, 0.0f, 1.0f);
        shaderprogram->setUniformValue("model", model);

        n = J7.capacity()/sizeof(float);
        QOpenGLVertexArrayObject::Binder bind7(&J7VAO);//绑定VAO
        this->glDrawArrays(GL_TRIANGLES,0,n);
        bind7.release();
        shaderprogram->release();

//        //TCP位置
//        double poses[6] = {0.0};
//        getDefaultActiveTcpPose (poses);
//        AxisToRpy(poses);
//        RadarrToDegarr(poses);


        frame6 = model;//用于碰撞检测


//        model.translate(poses[0]*1000, poses[1]*1000, poses[2]*1000);
//        model.rotate(poses[3], 1.0f, 0.0f, 0.0f);
//        model.rotate(poses[4], 0.0f, 1.0f, 0.0f);
//        model.rotate(poses[5], 0.0f, 0.0f, 1.0f);


        shaderprogramXYZ->bind();
        shaderprogramXYZ->setUniformValue("view", view);
        shaderprogramXYZ->setUniformValue("projection", projection);
        shaderprogramXYZ->setUniformValue("model", model);
        QOpenGLVertexArrayObject::Binder bindXYX(&VAOXYZ);//绑定VAO
        this->glDrawArrays(GL_LINES, 0, 2);
        this->glDrawArrays(GL_LINES, 2, 2);
        this->glDrawArrays(GL_LINES, 4, 2);
        bindXYX.release();
        shaderprogramXYZ->release();
    }

//    qDebug() << "model----------------" << model.column(3);
    model.setToIdentity();
//    model.translate(500, 500, 0);
    shaderprogramXYZ->bind();
    shaderprogramXYZ->setUniformValue("view", view);
    shaderprogramXYZ->setUniformValue("projection", projection);
    shaderprogramXYZ->setUniformValue("model", model);
    QOpenGLVertexArrayObject::Binder bindXYZ1(&VAOXYZ);//绑定VAO
    this->glDrawArrays(GL_LINES, 0, 2);
    this->glDrawArrays(GL_LINES, 2, 2);
    this->glDrawArrays(GL_LINES, 4, 2);
    bindXYZ1.release();
    shaderprogramXYZ->release();


    //将自定义帧缓冲区内容传到默认缓冲区显示，nullptr则为默认缓存区
    QOpenGLFramebufferObject::blitFramebuffer(nullptr,rect(),frame_buffer,QRect(0,0,frame_buffer->width(),frame_buffer->height()));

    frame_buffer->release();//不能打开，打开后后面无法使用glReadPixels，在默认帧缓冲区无法glReadPixels


}

void Widget::mousePressEvent(QMouseEvent *event)
{
//    mouseLastPos = event->pos();
    mousePos = QVector2D(event->pos());

    if(event->buttons() == Qt::LeftButton)
    {
        this->makeCurrent();
        frame_buffer->bind();//绑定自定义帧缓冲
        double modelview[16],projection1[16];
        int viewport[4];
    //    double object_x = 0,object_y = 0,object_z = 0; //3D坐标
        float a[16] = {0};
        model.copyDataTo(a);
        for (int i = 0; i < 16; i++)
        {
            modelview[i] = a[i];
        }
        float b[16] = {0};
        projection.copyDataTo(b);
        for (int i = 0; i < 16; i++)
        {
            projection1[i] = b[i];
        }
        glGetIntegerv(GL_VIEWPORT, viewport);
        GLfloat Z;
        glReadPixels(event->pos().x(), viewport[3]-event->pos().y(),1,1,GL_DEPTH_COMPONENT,GL_FLOAT, &Z);
//        GLenum err = glGetError();
//        qDebug() << "err2: " <<  err << "z" << Z;
        gluUnProject((GLdouble)event->pos().x(),(GLdouble)(viewport[3]-event->pos().y()), (GLdouble)Z,modelview,projection1,viewport,
                             &object_x,&object_y,&object_z);
//        qDebug() << "object_x: " <<  object_x << " object_y: " <<  object_y << " object_z: " <<  object_z;
        frame_buffer->release();
        this->doneCurrent();

        if(ui->checkBox->isChecked())
        {
            QMatrix4x4 click = base;
            QVector4D xyz(object_x, object_y, object_z, 1);
            click.setColumn(3, xyz);
            QMatrix4x4 click_in_base = base.inverted() * click;
            xyz = click_in_base.column(3);

            double pose[6] = {xyz.x(), xyz.y(), xyz.z(), M_PI, 0, 0};
            double joints[6] = {0};
            iiwa_inverse6(pose, joints);
//            qDebug() << joints[0] << "," <<  joints[1] << "," << joints[2] << "," << joints[3] << "," << joints[4] << "," << joints[5];

            angle_j1 = joints[0];
            angle_j2 = joints[1];
            angle_j3 = joints[2];
            angle_j4 = joints[3];
            angle_j5 = joints[4];
            angle_j6 = joints[5];
//            qDebug() << "mousePressEvent" << angle_j1 << "," << angle_j2 << "," << angle_j3 << "," << angle_j4 << "," << angle_j5 << "," << angle_j6;
        }

        update();
    }


    event->accept();
}

void Widget::mouseMoveEvent(QMouseEvent *event)
{
    if(event->buttons() == Qt::LeftButton)
    {

        QVector2D newPos = (QVector2D)event->pos();
        QVector2D diff = newPos - mousePos;
        qreal angle = (diff.length())/3.6;
        // Rotation axis is perpendicular to the mouse position difference
        // vector
        QVector3D rotationAxis = QVector3D(diff.y(), diff.x(), 0.0).normalized();
        rotation = QQuaternion::fromAxisAndAngle(rotationAxis, angle) * rotation;
        mousePos = newPos;
        this->update();

    }
    if(event->buttons() == Qt::MidButton)
    {
        int dx = event->x() - mousePos.x();
        int dy = event->y() - mousePos.y();
        mousePos = QVector2D(event->pos());


        xtrans += dx;
        ytrans -= dy;

        x += dx;
        y -= dy;

        midbutton = true;
        update();
    }
    if(event->buttons() == Qt::RightButton)
    {
//        int dx = event->x() - mouseLastPos.x();
//        int dy = event->y() - mouseLastPos.y();
//        mouseLastPos = event->pos();
//        xRot += dy;
//        yRot += dx;
//        update();
    }

    event->accept();
}

void Widget::wheelEvent(QWheelEvent *event)
{
//    QPoint numDegrees = event->angleDelta() / 8;

//    if (numDegrees.y() > 0) {
//        ztrans += 100;
//        qDebug() << "aaaa";
//    } else if (numDegrees.y() < 0) {
//        ztrans -= 100;
//        qDebug() << "bbbb";
//    }
//    this->update();
//    event->accept();

    scale *= 1 + event->angleDelta().y() * (s0 - 1);
//    midbutton = true;
    update();
}

void Widget::on_pushButton_clicked()
{
//    xtrans = 0;
//    ytrans = 0;
//    ztrans = 0; // translation on x,y,z-axis

//    rotation=QQuaternion(1.0f, 0.0f, 0.0f, 0.0f);;
//    scale = 1.0f;
//    s0 = 1.001f;
//    update();


////    double joints[7] = {0.0,24.0,0.0,-90.0,-30.0,90.0,0.0};
//    double joints[6] = {0.0,0.0,-90.0,-60.0,66.0,0.0};
//    for(int i = 0; i < 6; i++)
//    {
//        joints[i] = deg_to_rad(joints[i]);
//    }

//    double pose[6] = {0.0};
//    iiwa_forward6(joints,pose);
//    RadarrToDegarr(pose);
//    qDebug() << "aaa" << pose[0] << "," << pose[1] << "," << pose[2] << "," << pose[3] << "," << pose[4] << "," << pose[5];



    double pose[6] = {0.6,0,0.234,M_PI,0,M_PI};
    double joints[7] = {0};
    iiwa_inverse7(pose, joints, deg_to_rad(45));

    angle_j1 = rad_to_deg(joints[0]);
    angle_j2 = rad_to_deg(joints[1]);
    angle_j3 = rad_to_deg(joints[2]);
    angle_j4 = rad_to_deg(joints[3]);
    angle_j5 = rad_to_deg(joints[4]);
    angle_j6 = rad_to_deg(joints[5]);
    angle_j7 = rad_to_deg(joints[6]);

    update();




//    QElapsedTimer timer;

//    // 开始计时
//    timer.start();

//    // 执行你想要测试的代码
//    for (int i = 0; i < 100; ++i)
//    {
//        iiwa_inverse(pose, joints);
//    }

//    // 结束计时
//    qDebug() << u8"运行时间:" << timer.elapsed() << u8"毫秒";
}

void Widget::on_connect_robot_clicked()
{
//    srv_net_st * pinfo = new srv_net_st();
//    memset(pinfo ->SrvIp, 0x00, sizeof(pinfo ->SrvIp));
//    memcpy(pinfo ->SrvIp, "192.168.10.75", strlen("192.168.10.75"));
//    pinfo->LocHeartbeatPort = 0;
//    pinfo->LocRobotStatePort = 0;
//    pinfo->LocSrvPort = 0;
//    int ret = initSrv(nullptr, nullptr, pinfo);
//    if(ret < 0)
//    {
//     printf("192.168.10.75 initSrv failed! Return value = %d\n", ret);
//    }
//    if(pinfo)
//    {
//     delete pinfo;
//     pinfo = nullptr;
//    }

//    releaseBrake ();
//    QThread::sleep(2);

//    timer = new QTimer;
//    connect(timer, &QTimer::timeout, this, [=](){
//        double joints[7] = {0.0};
//        getJointPos(joints);
//        angle_update(rad_to_deg(joints[0]), rad_to_deg(joints[1]), rad_to_deg(joints[2]), rad_to_deg(joints[3]),
//                rad_to_deg(joints[4]), rad_to_deg(joints[5]), rad_to_deg(joints[6]));
//    });
//    if(ret == 0)
//    {
//        timer->start(10);
//        ui->connect_robot->setEnabled(false);
//    }
}


void Widget::on_pushButton_2_clicked()
{

}


void Widget::on_J1_valueChanged(int value)
{
//    qDebug() << value/1000.0;
    angle_j1 = value/1000.0;
    update();

    double joints[7] = {deg_to_rad(angle_j1), deg_to_rad(angle_j2), deg_to_rad(angle_j3), deg_to_rad(angle_j4),
                        deg_to_rad(angle_j5), deg_to_rad(angle_j6), deg_to_rad(angle_j7)};
    double pose[6] = {0};
    iiwa_forward7(joints, pose);

    ui->x_value->setText(QString::number(pose[0]*1000, 'f', 2));
    ui->y_value->setText(QString::number(pose[1]*1000, 'f', 2));
    ui->z_value->setText(QString::number(pose[2]*1000, 'f', 2));
    ui->rx_value->setText(QString::number(rad_to_deg(pose[3]), 'f', 2));
    ui->ry_value->setText(QString::number(rad_to_deg(pose[4]), 'f', 2));
    ui->rz_value->setText(QString::number(rad_to_deg(pose[5]), 'f', 2));

}

void Widget::on_J2_valueChanged(int value)
{
    angle_j2 = value/1000.0;
    update();

    double joints[7] = {deg_to_rad(angle_j1), deg_to_rad(angle_j2), deg_to_rad(angle_j3), deg_to_rad(angle_j4),
                        deg_to_rad(angle_j5), deg_to_rad(angle_j6), deg_to_rad(angle_j7)};
    double pose[6] = {0};
    iiwa_forward7(joints, pose);

    ui->x_value->setText(QString::number(pose[0]*1000, 'f', 2));
    ui->y_value->setText(QString::number(pose[1]*1000, 'f', 2));
    ui->z_value->setText(QString::number(pose[2]*1000, 'f', 2));
    ui->rx_value->setText(QString::number(rad_to_deg(pose[3]), 'f', 2));
    ui->ry_value->setText(QString::number(rad_to_deg(pose[4]), 'f', 2));
    ui->rz_value->setText(QString::number(rad_to_deg(pose[5]), 'f', 2));
}

void Widget::on_J3_valueChanged(int value)
{
    angle_j3 = value/1000.0;
    update();

    double joints[7] = {deg_to_rad(angle_j1), deg_to_rad(angle_j2), deg_to_rad(angle_j3), deg_to_rad(angle_j4),
                        deg_to_rad(angle_j5), deg_to_rad(angle_j6), deg_to_rad(angle_j7)};
    double pose[6] = {0};
    iiwa_forward7(joints, pose);

    ui->x_value->setText(QString::number(pose[0]*1000, 'f', 2));
    ui->y_value->setText(QString::number(pose[1]*1000, 'f', 2));
    ui->z_value->setText(QString::number(pose[2]*1000, 'f', 2));
    ui->rx_value->setText(QString::number(rad_to_deg(pose[3]), 'f', 2));
    ui->ry_value->setText(QString::number(rad_to_deg(pose[4]), 'f', 2));
    ui->rz_value->setText(QString::number(rad_to_deg(pose[5]), 'f', 2));
}

void Widget::on_J4_valueChanged(int value)
{
    angle_j4 = value/1000.0;
    update();

    double joints[7] = {deg_to_rad(angle_j1), deg_to_rad(angle_j2), deg_to_rad(angle_j3), deg_to_rad(angle_j4),
                        deg_to_rad(angle_j5), deg_to_rad(angle_j6), deg_to_rad(angle_j7)};
    double pose[6] = {0};
    iiwa_forward7(joints, pose);

    ui->x_value->setText(QString::number(pose[0]*1000, 'f', 2));
    ui->y_value->setText(QString::number(pose[1]*1000, 'f', 2));
    ui->z_value->setText(QString::number(pose[2]*1000, 'f', 2));
    ui->rx_value->setText(QString::number(rad_to_deg(pose[3]), 'f', 2));
    ui->ry_value->setText(QString::number(rad_to_deg(pose[4]), 'f', 2));
    ui->rz_value->setText(QString::number(rad_to_deg(pose[5]), 'f', 2));
}

void Widget::on_J5_valueChanged(int value)
{
    angle_j5 = value/1000.0;
    update();

    double joints[7] = {deg_to_rad(angle_j1), deg_to_rad(angle_j2), deg_to_rad(angle_j3), deg_to_rad(angle_j4),
                        deg_to_rad(angle_j5), deg_to_rad(angle_j6), deg_to_rad(angle_j7)};
    double pose[6] = {0};
    iiwa_forward7(joints, pose);

    ui->x_value->setText(QString::number(pose[0]*1000, 'f', 2));
    ui->y_value->setText(QString::number(pose[1]*1000, 'f', 2));
    ui->z_value->setText(QString::number(pose[2]*1000, 'f', 2));
    ui->rx_value->setText(QString::number(rad_to_deg(pose[3]), 'f', 2));
    ui->ry_value->setText(QString::number(rad_to_deg(pose[4]), 'f', 2));
    ui->rz_value->setText(QString::number(rad_to_deg(pose[5]), 'f', 2));
}

void Widget::on_J6_valueChanged(int value)
{
    angle_j6 = value/1000.0;
    update();

    double joints[7] = {deg_to_rad(angle_j1), deg_to_rad(angle_j2), deg_to_rad(angle_j3), deg_to_rad(angle_j4),
                        deg_to_rad(angle_j5), deg_to_rad(angle_j6), deg_to_rad(angle_j7)};
    double pose[6] = {0};
    iiwa_forward7(joints, pose);

    ui->x_value->setText(QString::number(pose[0]*1000, 'f', 2));
    ui->y_value->setText(QString::number(pose[1]*1000, 'f', 2));
    ui->z_value->setText(QString::number(pose[2]*1000, 'f', 2));
    ui->rx_value->setText(QString::number(rad_to_deg(pose[3]), 'f', 2));
    ui->ry_value->setText(QString::number(rad_to_deg(pose[4]), 'f', 2));
    ui->rz_value->setText(QString::number(rad_to_deg(pose[5]), 'f', 2));
}

void Widget::on_J7_valueChanged(int value)
{
    angle_j7 = value/1000.0;
    update();

    double joints[7] = {deg_to_rad(angle_j1), deg_to_rad(angle_j2), deg_to_rad(angle_j3), deg_to_rad(angle_j4),
                        deg_to_rad(angle_j5), deg_to_rad(angle_j6), deg_to_rad(angle_j7)};
    double pose[6] = {0};
    iiwa_forward7(joints, pose);

    ui->x_value->setText(QString::number(pose[0]*1000, 'f', 2));
    ui->y_value->setText(QString::number(pose[1]*1000, 'f', 2));
    ui->z_value->setText(QString::number(pose[2]*1000, 'f', 2));
    ui->rx_value->setText(QString::number(rad_to_deg(pose[3]), 'f', 2));
    ui->ry_value->setText(QString::number(rad_to_deg(pose[4]), 'f', 2));
    ui->rz_value->setText(QString::number(rad_to_deg(pose[5]), 'f', 2));
}

void Widget::on_home_clicked()
{
    ui->label_2->setText(0);
    ui->arm_angle->setValue(0);
    angle_j1 = 0;
    angle_j2 = 0;
    angle_j3 = 0;
    angle_j4 = 0;
    angle_j5 = 0;
    angle_j6 = 0;
    angle_j7 = 0;
    update();

}

void Widget::on_comboBox_activated(int index)
{
//    qDebug() << index;
    angle_j1 = rad_to_deg(theta[index][0]);
    angle_j2 = rad_to_deg(theta[index][1]);
    angle_j3 = rad_to_deg(theta[index][2]);
    angle_j4 = rad_to_deg(theta[index][3]);
    angle_j5 = rad_to_deg(theta[index][4]);
    angle_j6 = rad_to_deg(theta[index][5]);
    angle_j7 = rad_to_deg(theta[index][6]);

    update();
}

void Widget::on_add_X_clicked()
{
    double pose[6] = {0.0};
    pose[0] = (ui->x_value->text().toDouble())/1000;
    pose[1] = ui->y_value->text().toDouble()/1000;
    pose[2] = ui->z_value->text().toDouble()/1000;
    pose[3] = deg_to_rad(ui->rx_value->text().toDouble());
    pose[4] = deg_to_rad(ui->ry_value->text().toDouble());
    pose[5] = deg_to_rad(ui->rz_value->text().toDouble());
    if(ui->tool->isChecked())
    {
        Eigen::MatrixXd T(4,4),T1(4,4),T2(4,4);
        T << 1,0,0,0.01,
             0,1,0,0,
             0,0,1,0,
             0,0,0,1;
        T1 = getT_fromPose(pose);
        T2 = T1 * T;
        getPose_fromT(T2, pose);
    }
    else
    {
        pose[0] = pose[0] + 0.01;
    }

    double joints[7] = {angle_j1,angle_j2,angle_j3,angle_j4,angle_j5,angle_j6,angle_j7};
    iiwa_inverse7(pose, joints, deg_to_rad(ui->arm_angle->value()));
    angle_j1 = (joints[0]);
    angle_j2 = (joints[1]);
    angle_j3 = (joints[2]);
    angle_j4 = (joints[3]);
    angle_j5 = (joints[4]);
    angle_j6 = (joints[5]);
    angle_j7 = (joints[6]);
    update();
}

void Widget::on_sub_X_clicked()
{
    double pose[6] = {0.0};
    pose[0] = (ui->x_value->text().toDouble())/1000;
    pose[1] = ui->y_value->text().toDouble()/1000;
    pose[2] = ui->z_value->text().toDouble()/1000;
    pose[3] = deg_to_rad(ui->rx_value->text().toDouble());
    pose[4] = deg_to_rad(ui->ry_value->text().toDouble());
    pose[5] = deg_to_rad(ui->rz_value->text().toDouble());
    if(ui->tool->isChecked())
    {
        Eigen::MatrixXd T(4,4),T1(4,4),T2(4,4);
        T << 1,0,0,-0.01,
             0,1,0,0,
             0,0,1,0,
             0,0,0,1;
        T1 = getT_fromPose(pose);
        T2 = T1 * T;
        getPose_fromT(T2, pose);
    }
    else
    {
        pose[0] = pose[0] - 0.01;
    }

    double joints[7] = {angle_j1,angle_j2,angle_j3,angle_j4,angle_j5,angle_j6,angle_j7};
    iiwa_inverse7(pose, joints, deg_to_rad(ui->arm_angle->value()));
    angle_j1 = (joints[0]);
    angle_j2 = (joints[1]);
    angle_j3 = (joints[2]);
    angle_j4 = (joints[3]);
    angle_j5 = (joints[4]);
    angle_j6 = (joints[5]);
    angle_j7 = (joints[6]);
    update();
}

void Widget::on_sub_Y_clicked()
{
    double pose[6] = {0.0};
    pose[0] = (ui->x_value->text().toDouble())/1000;
    pose[1] = (ui->y_value->text().toDouble())/1000;
    pose[2] = ui->z_value->text().toDouble()/1000;
    pose[3] = deg_to_rad(ui->rx_value->text().toDouble());
    pose[4] = deg_to_rad(ui->ry_value->text().toDouble());
    pose[5] = deg_to_rad(ui->rz_value->text().toDouble());
    if(ui->tool->isChecked())
    {
        Eigen::MatrixXd T(4,4),T1(4,4),T2(4,4);
        T << 1,0,0,0,
             0,1,0,-0.01,
             0,0,1,0,
             0,0,0,1;
        T1 = getT_fromPose(pose);
        T2 = T1 * T;
        getPose_fromT(T2, pose);
    }
    else
    {
        pose[1] = pose[1] - 0.01;
    }

    double joints[7] = {angle_j1,angle_j2,angle_j3,angle_j4,angle_j5,angle_j6,angle_j7};
    iiwa_inverse7(pose, joints, deg_to_rad(ui->arm_angle->value()));
    angle_j1 = (joints[0]);
    angle_j2 = (joints[1]);
    angle_j3 = (joints[2]);
    angle_j4 = (joints[3]);
    angle_j5 = (joints[4]);
    angle_j6 = (joints[5]);
    angle_j7 = (joints[6]);
    update();
}

void Widget::on_add_Y_clicked()
{
    double pose[6] = {0.0};
    pose[0] = ui->x_value->text().toDouble()/1000;
    pose[1] = (ui->y_value->text().toDouble())/1000;
    pose[2] = ui->z_value->text().toDouble()/1000;
    pose[3] = deg_to_rad(ui->rx_value->text().toDouble());
    pose[4] = deg_to_rad(ui->ry_value->text().toDouble());
    pose[5] = deg_to_rad(ui->rz_value->text().toDouble());
    if(ui->tool->isChecked())
    {
        Eigen::MatrixXd T(4,4),T1(4,4),T2(4,4);
        T << 1,0,0,0,
             0,1,0,0.01,
             0,0,1,0,
             0,0,0,1;
        T1 = getT_fromPose(pose);
        T2 = T1 * T;
        getPose_fromT(T2, pose);
    }
    else
    {
        pose[1] = pose[1] + 0.01;
    }

    double joints[7] = {angle_j1,angle_j2,angle_j3,angle_j4,angle_j5,angle_j6,angle_j7};
    iiwa_inverse7(pose, joints, deg_to_rad(ui->arm_angle->value()));
    angle_j1 = (joints[0]);
    angle_j2 = (joints[1]);
    angle_j3 = (joints[2]);
    angle_j4 = (joints[3]);
    angle_j5 = (joints[4]);
    angle_j6 = (joints[5]);
    angle_j7 = (joints[6]);
    update();
}

void Widget::on_sub_Z_clicked()
{
    double pose[6] = {0.0};
    pose[0] = (ui->x_value->text().toDouble())/1000;
    pose[1] = (ui->y_value->text().toDouble())/1000;
    pose[2] = (ui->z_value->text().toDouble())/1000;
    pose[3] = deg_to_rad(ui->rx_value->text().toDouble());
    pose[4] = deg_to_rad(ui->ry_value->text().toDouble());
    pose[5] = deg_to_rad(ui->rz_value->text().toDouble());
    if(ui->tool->isChecked())
    {
        Eigen::MatrixXd T(4,4),T1(4,4),T2(4,4);
        T << 1,0,0,0,
             0,1,0,0,
             0,0,1,-0.01,
             0,0,0,1;
        T1 = getT_fromPose(pose);
        T2 = T1 * T;
        getPose_fromT(T2, pose);
    }
    else
    {
        pose[2] = pose[2] - 0.01;
    }

    double joints[7] = {angle_j1,angle_j2,angle_j3,angle_j4,angle_j5,angle_j6,angle_j7};
    iiwa_inverse7(pose, joints, deg_to_rad(ui->arm_angle->value()));
    angle_j1 = (joints[0]);
    angle_j2 = (joints[1]);
    angle_j3 = (joints[2]);
    angle_j4 = (joints[3]);
    angle_j5 = (joints[4]);
    angle_j6 = (joints[5]);
    angle_j7 = (joints[6]);
    update();
}

void Widget::on_add_Z_clicked()
{
    double pose[6] = {0.0};
    pose[0] = (ui->x_value->text().toDouble())/1000;
    pose[1] = (ui->y_value->text().toDouble())/1000;
    pose[2] = (ui->z_value->text().toDouble())/1000;
    pose[3] = deg_to_rad(ui->rx_value->text().toDouble());
    pose[4] = deg_to_rad(ui->ry_value->text().toDouble());
    pose[5] = deg_to_rad(ui->rz_value->text().toDouble());
    if(ui->tool->isChecked())
    {
        Eigen::MatrixXd T(4,4),T1(4,4),T2(4,4);
        T << 1,0,0,0,
             0,1,0,0,
             0,0,1,0.01,
             0,0,0,1;
        T1 = getT_fromPose(pose);
        T2 = T1 * T;
        getPose_fromT(T2, pose);
    }
    else
    {
        pose[2] = pose[2] + 0.01;
    }

    double joints[7] = {angle_j1,angle_j2,angle_j3,angle_j4,angle_j5,angle_j6,angle_j7};
    iiwa_inverse7(pose, joints, deg_to_rad(ui->arm_angle->value()));
    angle_j1 = (joints[0]);
    angle_j2 = (joints[1]);
    angle_j3 = (joints[2]);
    angle_j4 = (joints[3]);
    angle_j5 = (joints[4]);
    angle_j6 = (joints[5]);
    angle_j7 = (joints[6]);
    update();
}

void Widget::on_sub_RX_clicked()
{
    double pose[6] = {0.0};
    pose[0] = (ui->x_value->text().toDouble())/1000;
    pose[1] = ui->y_value->text().toDouble()/1000;
    pose[2] = ui->z_value->text().toDouble()/1000;
    pose[3] = deg_to_rad(ui->rx_value->text().toDouble());
    pose[4] = deg_to_rad(ui->ry_value->text().toDouble());
    pose[5] = deg_to_rad(ui->rz_value->text().toDouble());
    Eigen::MatrixXd T(4,4),T1(4,4),T2(4,4);
    if(ui->tool->isChecked())
    {
        T << 1,                  0,                    0,0,
             0, cos(deg_to_rad(-5)),  -sin(deg_to_rad(-5)),0,
             0, sin(deg_to_rad(-5)),   cos(deg_to_rad(-5)),0,
             0,                  0,                    0,1;
        T1 = getT_fromPose(pose);
        T2 = T1 * T;
        getPose_fromT(T2, pose);
    }
    else
    {
        T << 1,                  0,                    0,0,
             0, cos(deg_to_rad(5)),  -sin(deg_to_rad(5)),0,
             0, sin(deg_to_rad(5)),   cos(deg_to_rad(5)),0,
             0,                  0,                    0,1;
        T1 = getT_fromPose(pose);
        T1(0,3)=0;T1(1,3)=0;T1(2,3)=0;
        T2 = T1.inverse() * T;
        T1 = T2.inverse();T1(0,3)=pose[0];T1(1,3)=pose[1];T1(2,3)=pose[2];
        getPose_fromT(T1, pose);
    }

    double joints[7] = {angle_j1,angle_j2,angle_j3,angle_j4,angle_j5,angle_j6,angle_j7};
    iiwa_inverse7(pose, joints, deg_to_rad(ui->arm_angle->value()));
    angle_j1 = (joints[0]);
    angle_j2 = (joints[1]);
    angle_j3 = (joints[2]);
    angle_j4 = (joints[3]);
    angle_j5 = (joints[4]);
    angle_j6 = (joints[5]);
    angle_j7 = (joints[6]);
    update();
}

void Widget::on_add_RX_clicked()
{
    double pose[6] = {0.0};
    pose[0] = (ui->x_value->text().toDouble())/1000;
    pose[1] = (ui->y_value->text().toDouble())/1000;
    pose[2] = ui->z_value->text().toDouble()/1000;
    pose[3] = deg_to_rad(ui->rx_value->text().toDouble());
    pose[4] = deg_to_rad(ui->ry_value->text().toDouble());
    pose[5] = deg_to_rad(ui->rz_value->text().toDouble());
    Eigen::MatrixXd T(4,4),T1(4,4),T2(4,4);
    if(ui->tool->isChecked())
    {
        T << 1,                  0,                    0,0,
             0, cos(deg_to_rad(5)),  -sin(deg_to_rad(5)),0,
             0, sin(deg_to_rad(5)),   cos(deg_to_rad(5)),0,
             0,                  0,                    0,1;
        T1 = getT_fromPose(pose);
        T2 = T1 * T;
        getPose_fromT(T2, pose);
    }
    else
    {
        T << 1,                  0,                    0,0,
             0, cos(deg_to_rad(-5)),  -sin(deg_to_rad(-5)),0,
             0, sin(deg_to_rad(-5)),   cos(deg_to_rad(-5)),0,
             0,                  0,                    0,1;
        T1 = getT_fromPose(pose);
        T1(0,3)=0;T1(1,3)=0;T1(2,3)=0;
        T2 = T1.inverse() * T;
        T1 = T2.inverse();T1(0,3)=pose[0];T1(1,3)=pose[1];T1(2,3)=pose[2];
        getPose_fromT(T1, pose);
    }

    double joints[7] = {angle_j1,angle_j2,angle_j3,angle_j4,angle_j5,angle_j6,angle_j7};
    iiwa_inverse7(pose, joints, deg_to_rad(ui->arm_angle->value()));
    angle_j1 = (joints[0]);
    angle_j2 = (joints[1]);
    angle_j3 = (joints[2]);
    angle_j4 = (joints[3]);
    angle_j5 = (joints[4]);
    angle_j6 = (joints[5]);
    angle_j7 = (joints[6]);
    update();
}

void Widget::on_sub_RY_clicked()
{
    double pose[6] = {0.0};
    pose[0] = (ui->x_value->text().toDouble())/1000;
    pose[1] = ui->y_value->text().toDouble()/1000;
    pose[2] = ui->z_value->text().toDouble()/1000;
    pose[3] = deg_to_rad(ui->rx_value->text().toDouble());
    pose[4] = deg_to_rad(ui->ry_value->text().toDouble());
    pose[5] = deg_to_rad(ui->rz_value->text().toDouble());
    Eigen::MatrixXd T(4,4),T1(4,4),T2(4,4);
    if(ui->tool->isChecked())
    {
        T <<  cos(deg_to_rad(-5)),  0, sin(deg_to_rad(-5)),0,
                                0,  1,                   0,0,
             -sin(deg_to_rad(-5)),  0, cos(deg_to_rad(-5)),0,
             0,                     0,                   0,1;
        T1 = getT_fromPose(pose);
        T2 = T1 * T;
        getPose_fromT(T2, pose);
    }
    else
    {
        T <<  cos(deg_to_rad(5)),  0, sin(deg_to_rad(5)),0,
                               0,  1,                  0,0,
             -sin(deg_to_rad(5)),  0, cos(deg_to_rad(5)),0,
             0,                    0,                  0,1;
        T1 = getT_fromPose(pose);
        T1(0,3)=0;T1(1,3)=0;T1(2,3)=0;
        T2 = T1.inverse() * T;
        T1 = T2.inverse();T1(0,3)=pose[0];T1(1,3)=pose[1];T1(2,3)=pose[2];
        getPose_fromT(T1, pose);
    }

    double joints[7] = {angle_j1,angle_j2,angle_j3,angle_j4,angle_j5,angle_j6,angle_j7};
    iiwa_inverse7(pose, joints, deg_to_rad(ui->arm_angle->value()));
    angle_j1 = (joints[0]);
    angle_j2 = (joints[1]);
    angle_j3 = (joints[2]);
    angle_j4 = (joints[3]);
    angle_j5 = (joints[4]);
    angle_j6 = (joints[5]);
    angle_j7 = (joints[6]);
    update();
}

void Widget::on_add_RY_clicked()
{
    double pose[6] = {0.0};
    pose[0] = (ui->x_value->text().toDouble())/1000;
    pose[1] = ui->y_value->text().toDouble()/1000;
    pose[2] = ui->z_value->text().toDouble()/1000;
    pose[3] = deg_to_rad(ui->rx_value->text().toDouble());
    pose[4] = deg_to_rad(ui->ry_value->text().toDouble());
    pose[5] = deg_to_rad(ui->rz_value->text().toDouble());
    Eigen::MatrixXd T(4,4),T1(4,4),T2(4,4);
    if(ui->tool->isChecked())
    {
        T <<  cos(deg_to_rad(5)),  0, sin(deg_to_rad(5)),0,
                                0,  1,                   0,0,
             -sin(deg_to_rad(5)),  0, cos(deg_to_rad(5)),0,
             0,                     0,                   0,1;
        T1 = getT_fromPose(pose);
        T2 = T1 * T;
        getPose_fromT(T2, pose);
    }
    else
    {
        T <<  cos(deg_to_rad(-5)),  0, sin(deg_to_rad(-5)),0,
                               0,  1,                  0,0,
             -sin(deg_to_rad(-5)),  0, cos(deg_to_rad(-5)),0,
             0,                    0,                  0,1;
        T1 = getT_fromPose(pose);
        T1(0,3)=0;T1(1,3)=0;T1(2,3)=0;
        T2 = T1.inverse() * T;
        T1 = T2.inverse();T1(0,3)=pose[0];T1(1,3)=pose[1];T1(2,3)=pose[2];
        getPose_fromT(T1, pose);
    }

    double joints[7] = {angle_j1,angle_j2,angle_j3,angle_j4,angle_j5,angle_j6,angle_j7};
    iiwa_inverse7(pose, joints, deg_to_rad(ui->arm_angle->value()));
    angle_j1 = (joints[0]);
    angle_j2 = (joints[1]);
    angle_j3 = (joints[2]);
    angle_j4 = (joints[3]);
    angle_j5 = (joints[4]);
    angle_j6 = (joints[5]);
    angle_j7 = (joints[6]);
    update();
}

void Widget::on_sub_RZ_clicked()
{
    double pose[6] = {0.0};
    pose[0] = (ui->x_value->text().toDouble())/1000;
    pose[1] = ui->y_value->text().toDouble()/1000;
    pose[2] = ui->z_value->text().toDouble()/1000;
    pose[3] = deg_to_rad(ui->rx_value->text().toDouble());
    pose[4] = deg_to_rad(ui->ry_value->text().toDouble());
    pose[5] = deg_to_rad(ui->rz_value->text().toDouble());
    Eigen::MatrixXd T(4,4),T1(4,4),T2(4,4);
    if(ui->tool->isChecked())
    {
        T <<  cos(deg_to_rad(-5)),  -sin(deg_to_rad(-5)), 0,0,
              sin(deg_to_rad(-5)),   cos(deg_to_rad(-5)), 0,0,
                               0,                    0, 1,0,
                               0,                    0, 0,1;
        T1 = getT_fromPose(pose);
        T2 = T1 * T;
        getPose_fromT(T2, pose);
    }
    else
    {
        T <<  cos(deg_to_rad(5)),  -sin(deg_to_rad(5)), 0,0,
              sin(deg_to_rad(5)),   cos(deg_to_rad(5)), 0,0,
                               0,                    0, 1,0,
                               0,                    0, 0,1;
        T1 = getT_fromPose(pose);
        T1(0,3)=0;T1(1,3)=0;T1(2,3)=0;
        T2 = T1.inverse() * T;
        T1 = T2.inverse();T1(0,3)=pose[0];T1(1,3)=pose[1];T1(2,3)=pose[2];
        getPose_fromT(T1, pose);
    }

    double joints[7] = {angle_j1,angle_j2,angle_j3,angle_j4,angle_j5,angle_j6,angle_j7};
    iiwa_inverse7(pose, joints, deg_to_rad(ui->arm_angle->value()));
    angle_j1 = (joints[0]);
    angle_j2 = (joints[1]);
    angle_j3 = (joints[2]);
    angle_j4 = (joints[3]);
    angle_j5 = (joints[4]);
    angle_j6 = (joints[5]);
    angle_j7 = (joints[6]);
    update();
}

void Widget::on_add_RZ_clicked()
{
    double pose[6] = {0.0};
    pose[0] = (ui->x_value->text().toDouble())/1000;
    pose[1] = ui->y_value->text().toDouble()/1000;
    pose[2] = ui->z_value->text().toDouble()/1000;
    pose[3] = deg_to_rad(ui->rx_value->text().toDouble());
    pose[4] = deg_to_rad(ui->ry_value->text().toDouble());
    pose[5] = deg_to_rad(ui->rz_value->text().toDouble());
    Eigen::MatrixXd T(4,4),T1(4,4),T2(4,4);
    if(ui->tool->isChecked())
    {
        T <<  cos(deg_to_rad(5)),  -sin(deg_to_rad(5)), 0,0,
              sin(deg_to_rad(5)),   cos(deg_to_rad(5)), 0,0,
                               0,                    0, 1,0,
                               0,                    0, 0,1;
        T1 = getT_fromPose(pose);
        T2 = T1 * T;
        getPose_fromT(T2, pose);
    }
    else
    {
        T <<  cos(deg_to_rad(-5)),  -sin(deg_to_rad(-5)), 0,0,
              sin(deg_to_rad(-5)),   cos(deg_to_rad(-5)), 0,0,
                               0,                    0, 1,0,
                               0,                    0, 0,1;
        T1 = getT_fromPose(pose);
        T1(0,3)=0;T1(1,3)=0;T1(2,3)=0;
        T2 = T1.inverse() * T;
        T1 = T2.inverse();T1(0,3)=pose[0];T1(1,3)=pose[1];T1(2,3)=pose[2];
        getPose_fromT(T1, pose);
    }

    double joints[7] = {angle_j1,angle_j2,angle_j3,angle_j4,angle_j5,angle_j6,angle_j7};
    iiwa_inverse7(pose, joints, deg_to_rad(ui->arm_angle->value()));
    angle_j1 = (joints[0]);
    angle_j2 = (joints[1]);
    angle_j3 = (joints[2]);
    angle_j4 = (joints[3]);
    angle_j5 = (joints[4]);
    angle_j6 = (joints[5]);
    angle_j7 = (joints[6]);
    update();
}

void Widget::on_arm_angle_valueChanged(int value)
{
    ui->label_2->setText(QString::number(value));
    double pose[6] = {0.0};
    pose[0] = (ui->x_value->text().toDouble())/1000;
    pose[1] = ui->y_value->text().toDouble()/1000;
    pose[2] = ui->z_value->text().toDouble()/1000;
    pose[3] = deg_to_rad(ui->rx_value->text().toDouble());
    pose[4] = deg_to_rad(ui->ry_value->text().toDouble());
    pose[5] = deg_to_rad(ui->rz_value->text().toDouble());

    double joints[7] = {angle_j1,angle_j2,angle_j3,angle_j4,angle_j5,angle_j6,angle_j7};
    iiwa_inverse7(pose, joints, deg_to_rad(value));
    angle_j1 = (joints[0]);
    angle_j2 = (joints[1]);
    angle_j3 = (joints[2]);
    angle_j4 = (joints[3]);
    angle_j5 = (joints[4]);
    angle_j6 = (joints[5]);
    angle_j7 = (joints[6]);
    update();
}

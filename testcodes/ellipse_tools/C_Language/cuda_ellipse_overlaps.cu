#include "cuda_ellipse_overlaps.cuh"

#include <vector>

#define PI 3.14159265358979
using std::vector;

__device__ inline void CalculateRangeAtY(double *elpparm, double y, double *x1, double *x2)
{
    double A, B, C, D, E, F;
    A = elpparm[0], B = elpparm[1], C = elpparm[2];
    D = elpparm[3], E = elpparm[4], F = elpparm[5];

    double Delta = pow(B*y + D, 2) - A*(C*y*y + 2 * E*y + F);

    if (Delta < 0)
        *x1 = -10, *x2 = -20;
    else
    {
        double t1, t2;
        t1 = (-(B*y + D) - sqrt(Delta)) / A;
        t2 = (-(B*y + D) + sqrt(Delta)) / A;

        if (t2 < t1)
        {
            double tmp = t1;
            t1 = t2;
            t2 = tmp;
        }

        *x1 = t1;
        *x2 = t2;
    }

}


__device__ inline void ELPShape2Equation(double *elpshape, double *outparms)
{
    double xc, yc, a, b, theta;
    xc = elpshape[0], yc = elpshape[1], a = elpshape[2]/2, b = elpshape[3]/2, theta = elpshape[4];

    double parm[6];

    parm[0] = cos(theta)*cos(theta) / (a*a) + pow(sin(theta), 2) / (b*b);
    parm[1] = -(sin(2 * theta)*(a*a - b*b)) / (2 * a*a*b*b);
    parm[2] = pow(cos(theta), 2) / (b*b) + pow(sin(theta), 2) / (a*a);
    parm[3] = (-a*a*xc*pow(sin(theta), 2) + a*a*yc*sin(2 * theta) / 2) / (a*a*b*b) - (xc*pow(cos(theta), 2) + yc*sin(2 * theta) / 2) / (a*a);
    parm[4] = (-a*a*yc*pow(cos(theta), 2) + a*a*xc*sin(2 * theta) / 2) / (a*a*b*b) - (yc*pow(sin(theta), 2) + xc*sin(2 * theta) / 2) / (a*a);
    parm[5] = pow(xc*cos(theta) + yc*sin(theta), 2) / (a*a) + pow(yc*cos(theta) - xc*sin(theta), 2) / (b*b) - 1;

    double k = parm[0] * parm[2] - parm[1] * parm[1];

    for (int i = 0; i < 6; i++)
        outparms[i] = parm[i] / sqrt(fabs(k));

}





__device__ inline void CalculateOverlap(double *elp1, double *elp2, double *_ration)
{

    /*
    for(int i=0; i< 5;i++)
    {
        std::cout<<elp1[i]<<std::endl;

    }
    for(int i=0; i< 5;i++)
    {
        std::cout<<elp2[i]<<std::endl;

    }
    */

    double parm1[6], parm2[6];
    ELPShape2Equation(elp1, parm1);
    ELPShape2Equation(elp2, parm2);

    double y1_min, y1_max, y2_min, y2_max, y_min, y_max;
    y1_min = elp1[1] - fmax(elp1[2], elp1[3]); y1_max = elp1[1] + fmax(elp1[2], elp1[3]);
    y2_min = elp2[1] - fmax(elp2[2], elp2[3]); y2_max = elp2[1] + fmax(elp2[2], elp2[3]);
    y_min = floor(fmax(y1_min, y2_min));
    y_max = ceil(fmin(y1_max, y2_max));

    double search_step = 0.2;
    double S12 = 0;

    for (double i = y_min; i <= y_max+1e-6; i = i + search_step)
    {
        double x11, x12, x21, x22;
        CalculateRangeAtY(parm1, i, &x11, &x12);
        CalculateRangeAtY(parm2, i, &x21, &x22);

        //mexPrintf("[%.4f,%.4f],[%.4f,%.4f]\n", x11, x12, x21, x22);

        if (x11 <= x12&& x21 <= x22)
        {
            if (x11 <= x21 && x12 >= x21)
            {
                if (x12 < x22)
                {
                    S12 += x12 - x21;
                }
                else
                {
                    S12 += x22 - x21;
                }
            }
            else if (x21 <= x11 && x22 >= x11)
            {
                if (x22 < x12)
                {
                    S12 += x22 - x11;
                }
                else
                {
                    S12 += x12 - x11;
                }
            }
        }

    }

    //mexPrintf("%.4f\n", S12);
    *_ration = S12 *search_step / (PI*elp1[2] * elp1[3]/4 + PI*elp2[2] * elp2[3]/4 - S12*search_step);

}
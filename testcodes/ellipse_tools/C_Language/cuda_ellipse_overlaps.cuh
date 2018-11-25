#ifndef _CUDA_ELLIPSE_OVERLAPS_H
#define _CUDA_ELLIPSE_OVERLAPS_H


__device__  void CalculateRangeAtY(double *elpparm, double y, double *x1, double *x2);
__device__  void ELPShape2Equation(double *elpshape, double *outparms);
__device__  void CalculateOverlap(double *elp1, double *elp2, double *_ration);

#endif

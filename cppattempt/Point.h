
#ifndef _POINT_H
#define _POINT_H

#include <torch/serialize/tensor.h>
#include <vector>
#include <cuda.h>
#include <cuda_runtime_api.h>

void select_cube(at::Tensor xyz, at::Tensor idx_out, int b, int n,float radius);
void group_points(int b, int n, int c , int m , int nsamples, at::Tensor xyz, at::Tensor idx,  at::Tensor out);
void ball_query (int b, int n, int m, float radius, int nsample, at::Tensor xyz1, at::Tensor xyz2, at::Tensor  idx, at::Tensor   pts_cnt);
void farthestPoint(int b,int n,int m, at::Tensor  inp, at::Tensor  temp,at::Tensor  out);
void interpolate(int b, int n, int m,  at::Tensor  xyz1p, at::Tensor  xyz2p, at::Tensor  distp,   at::Tensor  idxp);
void three_interpolate(int b, int m, int c, int n, at::Tensor points, at::Tensor idx, at::Tensor weight, at::Tensor out);





void cubeSelectLauncher(int b, int n, float radius, float * xyz, int * idx_out);
void queryBallPointLauncher(int b, int n, int m, float radius, int nsample, const float *xyz1, const float *xyz2, int *idx, int *pts_cnt);
void farthestpointsamplingLauncher(int b,int n,int m,const float * inp,float * temp,int * out);
void threennLauncher(int b, int n, int m, const float *xyz1, const float *xyz2, float *dist, int *idx);
void interpolateLauncher(int b, int m, int c, int n, const float *points, const int *idx, const float *weight, float *out);
void group_pointsLauncher(int b, int n, int c, int m, int nsamples, const float * pointsp, const int * idxp, float * outp);

#endif

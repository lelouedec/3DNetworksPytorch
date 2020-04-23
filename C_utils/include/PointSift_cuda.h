#ifndef _POINTSHIFT_CUDA_H
#define _POINTSHIFT_CUDA_H

#ifdef __cplusplus
extern "C" {
#endif

void cubeSelectLauncher(int b, int n, float radius, const float* xyz, int* idx_out);
void group_pointsLauncher(int b, int n, int c, int m, int nsamples, const float * pointsp, const int * idxp, float * outp);
void queryBallPointLauncher(int b, int n, int m, float radius, int nsample, const float *xyz1, const float *xyz2, int *idx, int *pts_cnt);
void farthestpointsamplingLauncher(int b,int n,int m,const float * inp,float * temp,int * out);
void threennLauncher(int b, int n, int m, const float *xyz1, const float *xyz2, float *dist, int *idx);
void interpolateLauncher(int b, int m, int c, int n, const float *points, const int *idx, const float *weight, float *out);
void groupPointLauncher(int b, int n, int c, int m, int nsample, const float *points, const int *idx, float *out);
#ifdef __cplusplus
}
#endif

#endif

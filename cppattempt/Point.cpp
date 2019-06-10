#include <ATen/cuda/CUDAContext.h>
#include <torch/extension.h>
#include <iostream>
#include "Point.h"

void select_cube(at::Tensor xyz, at::Tensor idx_out, int b, int n,float radius)
{
  cubeSelectLauncher(b,n,radius,xyz.contiguous().data<float>(), idx_out.contiguous().data<int>());
}

void group_points(int b, int n, int c , int m , int nsamples, at::Tensor xyz, at::Tensor idx,  at::Tensor out)
{
  group_pointsLauncher(b,n,c,m,nsamples,xyz.contiguous().data<float>(),idx.contiguous().data<int>(),out.contiguous().data<float>());
}
void ball_query (int b, int n, int m, float radius, int nsample, at::Tensor xyz1, at::Tensor xyz2, at::Tensor  idx, at::Tensor   pts_cnt)
{
  queryBallPointLauncher(b, n, m, radius, nsample, xyz1.contiguous().data<float>(), xyz2.contiguous().data<float>(), idx.contiguous().data<int>(), pts_cnt.contiguous().data<int>());
}

void farthestPoint(int b,int n,int m, at::Tensor  inp, at::Tensor  temp,at::Tensor  out)
{
  farthestpointsamplingLauncher(b, n, m,  inp.contiguous().data<float>(),  temp.contiguous().data<float>(),out.contiguous().data<int>());
}

void interpolate(int b, int n, int m,  at::Tensor  xyz1p, at::Tensor  xyz2p, at::Tensor  distp,   at::Tensor  idxp){

  auto   xyz1   = xyz1p.contiguous().data<float>();
  auto    xyz2  = xyz2p.contiguous().data<float>();
  auto   dist   = distp.contiguous().data<float>();
  auto    idx   = idxp.contiguous().data<int>();

  for (int i=0;i<b;++i) {
     for (int j=0;j<n;++j) {
          float x1=xyz1[j*3+0];
          float y1=xyz1[j*3+1];
          float z1=xyz1[j*3+2];
          double best1=1e40; double best2=1e40; double best3=1e40;
          int besti1=0; int besti2=0; int besti3=0;
          for (int k=0;k<m;++k) {
             float x2=xyz2[k*3+0];
          float y2=xyz2[k*3+1];
          float z2=xyz2[k*3+2];
          //float d=max(sqrtf((x2-x1)*(x2-x1)+(y2-y1)*(y2-y1)+(z2-z1)*(z2-z1)),1e-20f);
          double d=(x2-x1)*(x2-x1)+(y2-y1)*(y2-y1)+(z2-z1)*(z2-z1);
             if (d<best1) {
                 best3=best2;
                 besti3=besti2;
                 best2=best1;
                 besti2=besti1;
                 best1=d;
                 besti1=k;
             } else if (d<best2) {
                 best3=best2;
                 besti3=besti2;
                 best2=d;
                 besti2=k;
             } else if (d<best3) {
                 best3=d;
                 besti3=k;
             }
         }
         distp.contiguous().data<float>()[j*3]=best1;
         idxp.contiguous().data<int>()[j*3]=besti1;
         distp.contiguous().data<float>()[j*3+1]=best2;
         idxp.contiguous().data<int>()[j*3+1]=besti2;
         distp.contiguous().data<float>()[j*3+2]=best3;
         idxp.contiguous().data<int>()[j*3+2]=besti3;
     }
     xyz1+=n*3;
     xyz2+=m*3;
     dist+=n*3;
     idx+=n*3;
 }
}

void three_interpolate(int b, int m, int c, int n, at::Tensor points, at::Tensor idx, at::Tensor weight, at::Tensor out){

  float * pointsp = points.contiguous().data<float>();
  float * weightp = weight.contiguous().data<float>();
  float * outp    = out.contiguous().data<float>();
  int   * idxp    = idx.contiguous().data<int>();
  float w1,w2,w3;
  int i1,i2,i3;
  for (int i=0;i<b;++i) {
     for (int j=0;j<n;++j) {
         w1=weightp[j*3];
         w2=weightp[j*3+1];
         w3=weightp[j*3+2];
         i1=idxp[j*3];
         i2=idxp[j*3+1];
         i3=idxp[j*3+2];
         for (int l=0;l<c;++l) {
             out.contiguous().data<float>()[j*c+l] = pointsp[i1*c+l]*w1 + pointsp[i2*c+l]*w2 + pointsp[i3*c+l]*w3;
          }
     }
     pointsp+=m*c;
     idxp+=n*3;
     weightp+=n*3;
     outp+=n*c;
 }
}

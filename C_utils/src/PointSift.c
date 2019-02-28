
#include <THC/THC.h>
#include "PointSift_cuda.h"
#include <math.h>
#include <time.h>

#define MAX(x, y) (((x) > (y)) ? (x) : (y))
#define MIN(x, y) (((x) < (y)) ? (x) : (y))

extern THCState *state;

void select_cube(THCudaTensor *xyz, THCudaIntTensor *idx_out, int b, int n,float radius)
{
  //select_cuda(int b, int n,float radius, float* xyz, float* idx_out)
  int *  output = THCudaIntTensor_data(state, idx_out);
  float * input = THCudaTensor_data(state, xyz);
  cubeSelectLauncher(b,n,radius,input,output);
}

void group_points(int b, int n, int c , int m , int nsamples, THCudaTensor *xyz, THCudaIntTensor *idx,  THCudaTensor *out)
{
  int *  idxp = THCudaIntTensor_data(state, idx);
  float * pointsp = THCudaTensor_data(state, xyz);
  float * outp = THCudaTensor_data(state, out);
  group_pointsLauncher(b,n,c,m,nsamples,pointsp,idxp,outp);
}
void ball_query (int b, int n, int m, float radius, int nsample, THCudaTensor *xyz1, THCudaTensor *xyz2, THCudaIntTensor * idx, THCudaIntTensor *  pts_cnt){

  int *  idxp = THCudaIntTensor_data(state, idx);
  int *  pts_cntp = THCudaIntTensor_data(state, pts_cnt);
  float * xyz1p = THCudaTensor_data(state, xyz1);
  float * xyz2p = THCudaTensor_data(state, xyz2);
  queryBallPointLauncher(b, n, m, radius, nsample, xyz1p, xyz2p, idxp, pts_cntp);
}

void farthestPoint(int b,int n,int m,THCudaTensor * inp, THCudaTensor * temp,THCudaIntTensor * out){

  int *  out2 = THCudaIntTensor_data(state, out);
  float * inp2 = THCudaTensor_data(state, inp);
  float * temp2 = THCudaTensor_data(state, temp);
  // clock_t t;
  // t = clock();
  farthestpointsamplingLauncher(b, n, m, inp2, temp2,  out2);
  // t = clock() - t;
  // double time_taken = ((double)t)/CLOCKS_PER_SEC; // in seconds
  // printf("fun() took %f seconds to execute \n", time_taken);
}
void interpolate(int b, int n, int m, THFloatTensor *xyz1p, THFloatTensor *xyz2p, THFloatTensor *distp, THIntTensor *idxp){

  float * xyz1 = THCudaTensor_data(state, xyz1p);
  float * xyz2 = THCudaTensor_data(state, xyz2p);
  float * dist = THCudaTensor_data(state, distp);
  int   * idx  = THCudaIntTensor_data(state, idxp);

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
         dist[j*3]=best1;
         idx[j*3]=besti1;
         dist[j*3+1]=best2;
         idx[j*3+1]=besti2;
         dist[j*3+2]=best3;
         idx[j*3+2]=besti3;
     }
     xyz1+=n*3;
     xyz2+=m*3;
     dist+=n*3;
     idx+=n*3;
 }
}
void three_interpolate(int b, int m, int c, int n, THFloatTensor *points, THIntTensor *idx, THFloatTensor *weight, THFloatTensor *out){

  float * pointsp = THFloatTensor_data(points);
  float * weightp = THFloatTensor_data(weight);
  float * outp    = THFloatTensor_data(out);
  int   * idxp    = THIntTensor_data(idx);
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
             outp[j*c+l] = pointsp[i1*c+l]*w1 + pointsp[i2*c+l]*w2 + pointsp[i3*c+l]*w3;
         }
     }
     pointsp+=m*c;
     idxp+=n*3;
     weightp+=n*3;
     outp+=n*c;
 }
}
void IOUcalc(THFloatTensor * b1b, THFloatTensor *b2b,THFloatTensor *outb , int nb1, int nb2)
{
  int counter = 0;
  float * b1 = THFloatTensor_data(b1b);
  float * b2 = THFloatTensor_data(b2b);
  float * out = THFloatTensor_data(outb);
  int i,j;
  float x,y,z,width,height,depth,x2,y2,z2,width2,height2,depth2;
  for( i = 0; i < nb1; i = i + 1 ){
      x       = b1[i*6 +0] ;
      y       = b1[i*6 +1] ;
      z       = b1[i*6 +2] ;
      width   = b1[i*6 +3] ;
      height  = b1[i*6 +4] ;
      depth   = b1[i*6 +5] ;
    for( j = 0; j < nb2; j = j + 1 ){
      x2      = b2[j*6 +0] ;
      y2      = b2[j*6 +1] ;
      z2      = b2[j*6 +2] ;
      width2  = b2[j*6 +3] ;
      height2 = b2[j*6 +4] ;
      depth2  = b2[j*6 +5] ;

      if( pow((x - x2),2) + pow((y - y2),2) + pow((z - z2),2)  < (height2*height2)  ){
        counter = counter + 1;
      	float x_inter, x2_inter, y_inter, y2_inter,z_inter, z2_inter;
        float xa  = x  -  0.5  *  width;
        float ya  = y  -  0.5  *  height;
        float za  = z  +  0.5  *  depth;

        float xb = x2  -  0.5  *  width2;
        float yb = y2  -  0.5  *  height2;
        float zb = z2  +  0.5  *  depth2;

        /////////////IOU over x axis //////////////////

        x_inter = MAX(xa,xb);
        y_inter = MAX(ya,yb);
        z_inter = MIN(za,zb);


      	x2_inter = MIN((xa + width),(xb + width2));
      	y2_inter = MIN((ya + height),(yb + height2));
        z2_inter = MAX((za + depth),(zb + depth2));

        float x_intersec = fabsf(x2_inter  -  x_inter);
        float y_intersec = fabsf(y2_inter  -  y_inter);
        float z_intersec = fabsf(z2_inter  -  z_inter);



        float volume = width * height * depth ;
        float volume2 = width2 * height2  * depth2;
        float intersection = x_intersec * y_intersec * z_intersec;
        float iou = (intersection) / ( volume + volume2 - intersection );
        // printf("\n");
        // printf(" width : %0.3f , height : %0.3f , depth : %0.3f,width2 : %0.3f , height2 : %0.3f , depth2 : %0.3f,  x %0.3f , y %0.3f , z %0.3f ,x2  %0.3f ,y2 %0.3f , z2 %0.3f\n",width , height ,  depth, width2 , height2 ,  depth2, xa , ya , za, xb , yb , zb);
        // printf(" %0.3f , %0.3f ,%0.3f, %0.3f , %0.3f ,%0.3f \n",x_intersec , y_intersec , volume, volume2, intersection, iou);
        // printf("\n");
        out[nb2 * i + j] = iou ;
      }else{
        out[nb2 * i + j] = 0.0;
      }
    }
  }
  printf("COUNTER %d\n",counter);
}

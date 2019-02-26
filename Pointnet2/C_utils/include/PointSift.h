void select_cube(THCudaTensor *xyz, THCudaIntTensor *idx_out, int b, int n,float radius);
void group_points(int b, int n, int c , int m , int nsamples, THCudaTensor *xyz, THCudaIntTensor *idx,  THCudaTensor *out);
void ball_query(int b, int n, int m, float radius, int nsample, THCudaTensor *xyz1, THCudaTensor *xyz2, THCudaIntTensor * idx, THCudaIntTensor *  pts_cnt);
void farthestPoint(int b,int n,int m,THCudaTensor * inp, THCudaTensor * temp,THCudaIntTensor * out);
void interpolate(int b, int n, int m, THFloatTensor *xyz1p, THFloatTensor *xyz2p, THFloatTensor *distp, THIntTensor *idxp);
void three_interpolate(int b, int m, int c, int n, THFloatTensor *points, THIntTensor *idx, THFloatTensor *weight, THFloatTensor *out);
void IOUcalc(THFloatTensor * b1b, THFloatTensor *b2b,THFloatTensor *out , int nb1, int nb2);

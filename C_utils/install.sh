#!bin/bash
/usr/local/cuda-9.0/bin/nvcc -c -o build/libsift.so src/PointSift_cuda.cu -x cu -Xcompiler -fPIC
python build.py

#gcc  PointSift.c pointSIFT_g.cu.o -o tf_pointSIFT_so.so -shared -fPIC

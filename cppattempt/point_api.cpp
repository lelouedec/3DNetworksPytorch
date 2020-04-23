#include <torch/serialize/tensor.h>
#include <torch/extension.h>

#include "Point.h"


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("select_cube", &select_cube);
  m.def("group_points", &group_points);
  m.def("ball_query", &ball_query);
  m.def("farthestPoint", &farthestPoint);
  m.def("interpolate", &interpolate);
  m.def("three_interpolate", &three_interpolate);
}

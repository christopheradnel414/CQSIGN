/*
 * Use pytorch c++ extension to export c++ functions to python
 */

#include <ATen/ATen.h>
#include <torch/extension.h>
#include <ATen/NativeFunctions.h>

namespace at {
namespace native {

// Copied from 
// https://github.com/pytorch/pytorch/blob/8deb4fe809ca956276e8d6edaa184de7118be58f/aten/src/ATen/native/layer_norm.h#L11
std::tuple<Tensor, Tensor, Tensor, int64_t, int64_t> prepare_layer_norm_inputs(
    const Tensor& input,
    IntArrayRef normalized_shape,
    const Tensor& weight /* optional */,
    const Tensor& bias /* optional */) {

  const int64_t normalized_ndim = normalized_shape.size();
  TORCH_CHECK(
      normalized_ndim >= 1,
      "Expected normalized_shape to be at least 1-dimensional, i.e., ",
      "containing at least one element, but got normalized_shape = ",
      normalized_shape);
  TORCH_CHECK(
      !weight.defined() || weight.sizes().equals(normalized_shape),
      "Expected weight to be of same shape as normalized_shape, but got ",
      "weight of shape ",
      weight.sizes(),
      " and normalized_shape = ",
      normalized_shape);
  TORCH_CHECK(
      !bias.defined() || bias.sizes().equals(normalized_shape),
      "Expected bias to be of same shape as normalized_shape, but got ",
      "bias of shape ",
      bias.sizes(),
      " and normalized_shape = ",
      normalized_shape);

  const auto input_shape = input.sizes();
  const auto input_ndim = input.dim();

  if (input_ndim < normalized_ndim ||
      !input_shape.slice(input_ndim - normalized_ndim)
           .equals(normalized_shape)) {
    std::stringstream ss;
    ss << "Given normalized_shape=" << normalized_shape
       << ", expected input with shape [*";
    for (auto size : normalized_shape) {
      ss << ", " << size;
    }
    ss << "], but got input of size" << input_shape;
    AT_ERROR(ss.str());
  }

  const int axis = input_ndim - normalized_ndim;
  const int64_t M = std::accumulate(
      input_shape.cbegin(),
      input_shape.cbegin() + axis,
      1LL,
      std::multiplies<int64_t>());
  const int64_t N = std::accumulate(
      input_shape.cbegin() + axis,
      input_shape.cend(),
      1LL,
      std::multiplies<int64_t>());

  const auto& X = input.is_contiguous() ? input : input.contiguous();
  const auto& gamma = weight.is_contiguous() ? weight : weight.contiguous();
  const auto& beta = bias.is_contiguous() ? bias : bias.contiguous();

  return std::make_tuple(X, gamma, beta, M, N);
}


}  // namespace native
}  // namespace at


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  // m.def("cudnn_convolution_backward",           &at::cudnn_convolution_backward);
  // m.def("cudnn_convolution_transpose_backward", &at::cudnn_convolution_transpose_backward);
  m.def("prepare_layer_norm_inputs",  &at::native::prepare_layer_norm_inputs);
  m.def("layer_norm_cuda",            &at::native::layer_norm_cuda);
  m.def("layer_norm_backward_cuda",   &at::native::layer_norm_backward_cuda);
  m.def("cudnn_batch_norm",           &at::native::cudnn_batch_norm);
  m.def("cudnn_batch_norm_backward",  &at::native::cudnn_batch_norm_backward);
  m.def("native_batch_norm",          &at::native_batch_norm);
  m.def("native_batch_norm_backward", &at::native_batch_norm_backward);
}

#include <iostream>
#include <vector>
#include "acl.h"
// unary
#include "aclnnop/aclnn_abs.h"
#include "aclnnop/aclnn_neg.h"
#include "aclnnop/aclnn_exp.h"
#include "aclnnop/aclnn_log.h"
#include "aclnnop/aclnn_sqrt.h"
#include "aclnnop/aclnn_ceil.h"
#include "aclnnop/aclnn_floor.h"
#include "aclnnop/aclnn_round.h"
#include "aclnnop/aclnn_sin.h"
#include "aclnnop/aclnn_cos.h"
#include "aclnnop/aclnn_tan.h"
#include "aclnnop/aclnn_asin.h"
#include "aclnnop/aclnn_acos.h"
#include "aclnnop/aclnn_atan.h"
#include "aclnnop/aclnn_sinh.h"
#include "aclnnop/aclnn_cosh.h"
#include "aclnnop/aclnn_tanh.h"
#include "aclnnop/aclnn_asinh.h"
#include "aclnnop/aclnn_acosh.h"
#include "aclnnop/aclnn_atanh.h"
#include "aclnnop/aclnn_sigmoid.h"
#include "aclnnop/aclnn_erf.h"
#include "aclnnop/aclnn_erfinv.h"
#include "aclnnop/aclnn_logical_not.h"
#include "aclnnop/aclnn_bitwise_not.h"
#include "aclnnop/aclnn_cast.h"
#include "aclnnop/aclnn_nonzero.h"
// binary
#include "aclnnop/aclnn_maximum.h"
#include "aclnnop/aclnn_minimum.h"
#include "aclnnop/aclnn_add.h"
#include "aclnnop/aclnn_sub.h"
#include "aclnnop/aclnn_mul.h"
#include "aclnnop/aclnn_div.h"
#include "aclnnop/aclnn_floor_divide.h"
#include "aclnnop/aclnn_le_tensor.h"
#include "aclnnop/aclnn_lt_tensor.h"
#include "aclnnop/aclnn_ge_tensor.h"
#include "aclnnop/aclnn_gt_tensor.h"
#include "aclnnop/aclnn_eq_tensor.h"
#include "aclnnop/aclnn_ne_tensor.h"
#include "aclnnop/aclnn_logical_and.h"
#include "aclnnop/aclnn_logical_or.h"
#include "aclnnop/aclnn_logical_xor.h"
#include "aclnnop/aclnn_bitwise_and_tensor.h"
#include "aclnnop/aclnn_bitwise_or_tensor.h"
#include "aclnnop/aclnn_bitwise_xor_tensor.h"
#include "aclnnop/aclnn_pow_tensor_tensor.h"
#include "aclnnop/aclnn_expand.h"
#include "aclnnop/aclnn_matmul.h"
#include "aclnnop/aclnn_batch_matmul.h"
#include "aclnnop/aclnn_convolution.h"
#include "aclnnop/aclnn_convolution_backward.h"
#include "aclnnop/aclnn_reduce_sum.h"
#include "aclnnop/aclnn_amax.h"
#include "aclnnop/aclnn_amin.h"
#include "aclnnop/aclnn_mean.h"
#include "aclnnop/aclnn_prod.h"
#include "aclnnop/aclnn_triu.h"
#include "aclnnop/aclnn_s_where.h"
#include "aclnnop/aclnn_random.h"
#include "aclnnop/aclnn_normal.h"
#include "aclnnop/aclnn_permute.h"
#include "aclnnop/aclnn_max_pool2d_with_indices.h"
#include "aclnnop/aclnn_max_pool2d_with_indices_backward.h"
#include "aclnnop/aclnn_avgpool2d.h"
#include "aclnnop/aclnn_avgpool2d_backward.h"
#include "aclnnop/aclnn_flip.h"
#include "aclnnop/aclnn_cat.h"
#include "aclnnop/aclnn_gather.h"
#include "aclnnop/aclnn_cumsum.h"
#include "aclnnop/aclnn_index.h"
#include "aclnnop/aclnn_scatter.h"
#include "aclnnop/aclnn_index.h"
#include "aclnnop/aclnn_strided_slice_assign_v2.h"
#include "aclnnop/aclnn_slice_v2.h"
#include "aclnnop/aclnn_index_put_impl.h"
#include "aclnnop/aclnn_range.h"
#include "aclnnop/aclnn_relu.h"
#include "aclnnop/aclnn_dropout.h"
#include "aclnnop/aclnn_dropout_backward.h"
#include "aclnnop/aclnn_leaky_relu.h"
#include "aclnnop/aclnn_leaky_relu_backward.h"
#include "aclnnop/aclnn_uniform.h"
#include "aclnnop/aclnn_silu.h"
#include "aclnnop/aclnn_silu_backward.h"
#include "aclnnop/aclnn_sigmoid.h"
#include "aclnnop/aclnn_sigmoid_backward.h"
#include "aclnnop/aclnn_embedding.h"
#include "aclnnop/aclnn_embedding_dense_backward.h"
#include "aclnnop/aclnn_masked_scatter.h"
#include "aclnnop/aclnn_masked_select.h"
#include "aclnnop/aclnn_split_with_size.h"
#include "aclnnop/aclnn_flash_attention_score.h"
#include "aclnnop/aclnn_flash_attention_score_grad.h"
#include "aclnnop/aclnn_softmax.h"
#include "aclnnop/aclnn_softmax_backward.h"
#include "aclnnop/aclnn_batch_norm.h"
#include "aclnnop/aclnn_batch_norm_backward.h"
#include "aclnnop/aclnn_layer_norm.h"

#define CHECK_RET(cond, return_expr) \
  do                                 \
  {                                  \
    if (!(cond))                     \
    {                                \
      return_expr;                   \
    }                                \
  } while (0)

#define LOG_PRINT(message, ...)     \
  do                                \
  {                                 \
    printf(message, ##__VA_ARGS__); \
  } while (0)

int64_t GetShapeSize(const std::vector<int64_t> &shape);

void PrintOutResult(std::vector<int64_t> &shape, void **deviceAddr);

int Init(int32_t deviceId);

/*
template <typename T>
int CreateAclTensor(const std::vector<T>& hostData, const std::vector<int64_t>& shape, void** deviceAddr,
                    aclDataType dataType, aclTensor** tensor);
*/

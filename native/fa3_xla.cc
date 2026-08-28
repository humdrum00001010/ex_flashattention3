#include <array>
#include <atomic>
#include <cmath>
#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <type_traits>
#include <vector>

#include <cuda.h>
#include <cuda_runtime.h>
#include <cutlass/numeric_types.h>

#include "flash.h"
#include "xla/ffi/api/ffi.h"
#include "xla/ffi/ffi_api.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_instructions.h"
#include "xla/hlo/ir/hlo_sharding.h"
#include "xla/service/custom_call_sharding_helper.h"
#include "xla/service/spmd/spmd_partitioner.h"
#include "xla/service/spmd/spmd_partitioner_util.h"
#include "xla/shape_util.h"

namespace ffi = xla::ffi;

namespace {

constexpr char kTarget[] = "exla_fa3_forward";
constexpr char kTargetF16[] = "exla_fa3_forward_f16";
constexpr char kBackwardTarget[] = "exla_fa3_backward";
constexpr char kBackwardTargetF16[] = "exla_fa3_backward_f16";
constexpr int kMaxCudaDevices = 16;

std::array<std::atomic<int>, kMaxCudaDevices> cached_num_sms{};

ffi::Error Invalid(std::string message) {
  return ffi::Error(ffi::ErrorCode::kInvalidArgument, std::move(message));
}

ffi::Error CudaError(cudaError_t status, const char *operation) {
  return ffi::Error(ffi::ErrorCode::kInternal,
                    std::string(operation) +
                        " failed: " + cudaGetErrorString(status));
}

template <typename Dims> bool HasRank(const Dims &dims, std::size_t rank) {
  return dims.size() == rank;
}

ffi::ErrorOr<int> DeviceNumSms(int device) {
  if (device < 0 || device >= kMaxCudaDevices) {
    return ffi::Unexpected(
        Invalid("CUDA device index exceeds device-property cache"));
  }

  int num_sm = cached_num_sms[device].load(std::memory_order_acquire);
  if (num_sm != 0) {
    return num_sm;
  }

  cudaError_t status =
      cudaDeviceGetAttribute(&num_sm, cudaDevAttrMultiProcessorCount, device);
  if (status != cudaSuccess) {
    return ffi::Unexpected(CudaError(status, "cudaDeviceGetAttribute"));
  }

  cached_num_sms[device].store(num_sm, std::memory_order_release);
  return num_sm;
}

template <ffi::DataType FfiType, typename KernelType>
ffi::Error Fa3ForwardTyped(CUstream platform_stream, int32_t device,
                           ffi::Buffer<FfiType> q, ffi::Buffer<FfiType> k,
                           ffi::Buffer<FfiType> v, bool causal,
                           float softmax_scale,
                           ffi::Result<ffi::Buffer<FfiType>> output,
                           ffi::Result<ffi::Buffer<ffi::F32>> lse,
                           ffi::Result<ffi::Buffer<ffi::S32>> workspace) {
  auto q_dims = q.dimensions();
  auto k_dims = k.dimensions();
  auto v_dims = v.dimensions();
  auto o_dims = output->dimensions();
  auto lse_dims = lse->dimensions();
  auto workspace_dims = workspace->dimensions();

  if (!HasRank(q_dims, 4) || !HasRank(k_dims, 4) || !HasRank(v_dims, 4) ||
      !HasRank(o_dims, 4) || !HasRank(lse_dims, 3) ||
      !HasRank(workspace_dims, 1)) {
    return Invalid(
        "FA3 expects rank-4 Q/K/V/O, rank-3 LSE, and rank-1 workspace");
  }

  int64_t batch = q_dims[0];
  int64_t seqlen_q = q_dims[1];
  int64_t q_heads = q_dims[2];
  int64_t head_dim = q_dims[3];
  int64_t seqlen_k = k_dims[1];
  int64_t kv_heads = k_dims[2];
  int64_t value_dim = v_dims[3];

  if (batch <= 0 || seqlen_q <= 0 || seqlen_k <= 0 || q_heads <= 0 ||
      kv_heads <= 0 || q_heads % kv_heads != 0) {
    return Invalid("FA3 received invalid batch/sequence/GQA dimensions");
  }

  if (head_dim != 128 && head_dim != 256) {
    return Invalid("FA3 experiment supports head dimensions 128 and 256 only");
  }

  if (value_dim != head_dim || k_dims[0] != batch || v_dims[0] != batch ||
      k_dims[3] != head_dim || v_dims[1] != seqlen_k || v_dims[2] != kv_heads) {
    return Invalid("FA3 K/V dimensions do not match Q");
  }

  if (o_dims[0] != batch || o_dims[1] != seqlen_q || o_dims[2] != q_heads ||
      o_dims[3] != value_dim || lse_dims[0] != batch ||
      lse_dims[1] != q_heads || lse_dims[2] != seqlen_q ||
      workspace_dims[0] != batch) {
    return Invalid(
        "FA3 output, LSE, or workspace dimensions do not match the ABI");
  }

  if (causal && seqlen_q != seqlen_k) {
    return Invalid("causal FA3 experiment requires equal Q/K sequence lengths");
  }

  auto num_sm = DeviceNumSms(device);
  if (!num_sm) {
    return num_sm.error();
  }

  cudaStream_t stream = reinterpret_cast<cudaStream_t>(platform_stream);
  Flash_fwd_params params{};

  params.q_ptr = q.untyped_data();
  params.k_ptr = k.untyped_data();
  params.v_ptr = v.untyped_data();
  params.o_ptr = output->untyped_data();
  params.softmax_lse_ptr = lse->untyped_data();

  params.q_batch_stride = seqlen_q * q_heads * head_dim;
  params.k_batch_stride = seqlen_k * kv_heads * head_dim;
  params.v_batch_stride = seqlen_k * kv_heads * value_dim;
  params.o_batch_stride = seqlen_q * q_heads * value_dim;
  params.q_row_stride = q_heads * head_dim;
  params.k_row_stride = kv_heads * head_dim;
  params.v_row_stride = kv_heads * value_dim;
  params.o_row_stride = q_heads * value_dim;
  params.q_head_stride = head_dim;
  params.k_head_stride = head_dim;
  params.v_head_stride = value_dim;
  params.o_head_stride = value_dim;
  params.v_dim_stride = 1;

  params.b = static_cast<int>(batch);
  params.b_k = static_cast<int>(batch);
  params.h = static_cast<int>(q_heads);
  params.h_k = static_cast<int>(kv_heads);
  params.seqlen_q = static_cast<int>(seqlen_q);
  params.seqlen_k = static_cast<int>(seqlen_k);
  params.seqlen_q_rounded = static_cast<int>((seqlen_q + 127) / 128 * 128);
  params.seqlen_k_rounded = static_cast<int>((seqlen_k + 127) / 128 * 128);
  params.total_q = static_cast<int>(batch * seqlen_q);
  params.total_k = static_cast<int>(batch * seqlen_k);
  params.d = static_cast<int>(head_dim);
  params.d_rounded = static_cast<int>(head_dim);
  params.dv = static_cast<int>(value_dim);
  params.dv_rounded = static_cast<int>(value_dim);

  params.scale_softmax = softmax_scale;
  params.softcap = 0.0f;
  params.p_dropout = 1.0f;
  params.p_dropout_in_uint8_t = 255;
  params.rp_dropout = 1.0f;
  params.is_bf16 = std::is_same_v<KernelType, cutlass::bfloat16_t>;
  params.is_fp32 = false;
  params.is_e4m3 = false;
  params.is_causal = causal;
  params.is_local = false;
  params.window_size_left = static_cast<int>(seqlen_k - 1);
  params.window_size_right = causal ? 0 : static_cast<int>(seqlen_q - 1);
  params.attention_chunk = 0;

  params.arch = 90;
  params.num_sm = *num_sm;
  params.num_splits = 1;
  params.pack_gqa = false;
  params.pagedkv_tma = false;
  params.page_size = 1;
  params.varlen_sort_batches = true;
  params.head_swizzle = causal;
  params.skip_scheduler_metadata_computation = false;
  params.tile_count_semaphore_offset = 0;

  if (causal) {
    cudaError_t status =
        cudaMemsetAsync(workspace->untyped_data(), 0, sizeof(int), stream);
    if (status != cudaSuccess) {
      return CudaError(status, "cudaMemsetAsync scheduler workspace");
    }
    params.tile_count_semaphore = static_cast<int *>(workspace->untyped_data());
  }

  if (head_dim == 128) {
    run_mha_fwd_<90, KernelType, 128, 128, false, false, false, false>(params,
                                                                       stream);
  } else {
    run_mha_fwd_<90, KernelType, 256, 256, false, false, false, false>(params,
                                                                       stream);
  }

  return ffi::Error::Success();
}

ffi::Error Fa3ForwardBf16(CUstream stream, int32_t device,
                          ffi::Buffer<ffi::BF16> q, ffi::Buffer<ffi::BF16> k,
                          ffi::Buffer<ffi::BF16> v, bool causal,
                          float softmax_scale,
                          ffi::Result<ffi::Buffer<ffi::BF16>> output,
                          ffi::Result<ffi::Buffer<ffi::F32>> lse,
                          ffi::Result<ffi::Buffer<ffi::S32>> workspace) {
  return Fa3ForwardTyped<ffi::BF16, cutlass::bfloat16_t>(
      stream, device, q, k, v, causal, softmax_scale, output, lse, workspace);
}

ffi::Error Fa3ForwardF16(CUstream stream, int32_t device,
                         ffi::Buffer<ffi::F16> q, ffi::Buffer<ffi::F16> k,
                         ffi::Buffer<ffi::F16> v, bool causal,
                         float softmax_scale,
                         ffi::Result<ffi::Buffer<ffi::F16>> output,
                         ffi::Result<ffi::Buffer<ffi::F32>> lse,
                         ffi::Result<ffi::Buffer<ffi::S32>> workspace) {
  return Fa3ForwardTyped<ffi::F16, cutlass::half_t>(
      stream, device, q, k, v, causal, softmax_scale, output, lse, workspace);
}

template <ffi::DataType FfiType, typename KernelType>
ffi::Error Fa3BackwardTyped(
    CUstream platform_stream, int32_t device, ffi::Buffer<FfiType> q,
    ffi::Buffer<FfiType> k, ffi::Buffer<FfiType> v, ffi::Buffer<FfiType> output,
    ffi::Buffer<ffi::F32> lse, ffi::Buffer<FfiType> doutput, bool causal,
    float softmax_scale, ffi::Result<ffi::Buffer<FfiType>> dq,
    ffi::Result<ffi::Buffer<FfiType>> dk, ffi::Result<ffi::Buffer<FfiType>> dv,
    ffi::Result<ffi::Buffer<ffi::F32>> softmax_d,
    ffi::Result<ffi::Buffer<ffi::F32>> softmax_lse_log2,
    ffi::Result<ffi::Buffer<ffi::F32>> dq_accum,
    ffi::Result<ffi::Buffer<ffi::S32>> dq_semaphore,
    ffi::Result<ffi::Buffer<ffi::F32>> dk_accum,
    ffi::Result<ffi::Buffer<ffi::F32>> dv_accum) {
  auto q_dims = q.dimensions();
  auto k_dims = k.dimensions();
  auto v_dims = v.dimensions();
  auto o_dims = output.dimensions();
  auto lse_dims = lse.dimensions();
  auto do_dims = doutput.dimensions();

  if (!HasRank(q_dims, 4) || !HasRank(k_dims, 4) || !HasRank(v_dims, 4) ||
      !HasRank(o_dims, 4) || !HasRank(lse_dims, 3) || !HasRank(do_dims, 4)) {
    return Invalid("FA3 backward received invalid ranks");
  }

  int64_t batch = q_dims[0];
  int64_t seqlen_q = q_dims[1];
  int64_t q_heads = q_dims[2];
  int64_t head_dim = q_dims[3];
  int64_t seqlen_k = k_dims[1];
  int64_t kv_heads = k_dims[2];
  int64_t value_dim = v_dims[3];

  if ((head_dim != 128 && head_dim != 256) || value_dim != head_dim ||
      batch <= 0 || q_heads <= 0 || kv_heads <= 0 || q_heads % kv_heads != 0 ||
      k_dims[0] != batch || k_dims[3] != head_dim || v_dims[0] != batch ||
      v_dims[1] != seqlen_k || v_dims[2] != kv_heads || o_dims[0] != batch ||
      o_dims[1] != seqlen_q || o_dims[2] != q_heads || o_dims[3] != value_dim ||
      do_dims[0] != batch || do_dims[1] != seqlen_q || do_dims[2] != q_heads ||
      do_dims[3] != value_dim || lse_dims[0] != batch ||
      lse_dims[1] != q_heads || lse_dims[2] != seqlen_q) {
    return Invalid("FA3 backward dimensions do not match the ABI");
  }

  if (causal && seqlen_q != seqlen_k) {
    return Invalid("causal FA3 backward requires equal Q/K sequence lengths");
  }

  auto num_sm = DeviceNumSms(device);
  if (!num_sm) {
    return num_sm.error();
  }

  int k_block_m = head_dim <= 128 ? (causal ? 64 : 80) : 64;
  int k_block_n = head_dim <= 128 ? 128 : 80;
  int seqlen_q_rounded =
      static_cast<int>((seqlen_q + k_block_m - 1) / k_block_m * k_block_m);
  int seqlen_k_rounded =
      static_cast<int>((seqlen_k + k_block_n - 1) / k_block_n * k_block_n);
  int q_blocks = static_cast<int>((seqlen_q + k_block_m - 1) / k_block_m);

  auto softmax_d_dims = softmax_d->dimensions();
  auto softmax_lse_log2_dims = softmax_lse_log2->dimensions();
  auto dq_accum_dims = dq_accum->dimensions();
  auto dq_semaphore_dims = dq_semaphore->dimensions();
  auto dk_accum_dims = dk_accum->dimensions();
  auto dv_accum_dims = dv_accum->dimensions();

  if (!HasRank(softmax_d_dims, 3) || !HasRank(softmax_lse_log2_dims, 3) ||
      !HasRank(dq_accum_dims, 4) || !HasRank(dq_semaphore_dims, 3) ||
      !HasRank(dk_accum_dims, 4) || !HasRank(dv_accum_dims, 4) ||
      softmax_d_dims[0] != batch || softmax_d_dims[1] != q_heads ||
      softmax_d_dims[2] != seqlen_q_rounded ||
      softmax_lse_log2_dims[0] != batch ||
      softmax_lse_log2_dims[1] != q_heads ||
      softmax_lse_log2_dims[2] != seqlen_q_rounded ||
      dq_accum_dims[0] != batch || dq_accum_dims[1] != q_heads ||
      dq_accum_dims[2] != seqlen_q_rounded || dq_accum_dims[3] != head_dim ||
      dq_semaphore_dims[0] != q_blocks || dq_semaphore_dims[1] != batch ||
      dq_semaphore_dims[2] != q_heads || dk_accum_dims[0] != batch ||
      dk_accum_dims[1] != kv_heads || dk_accum_dims[2] != seqlen_k_rounded ||
      dk_accum_dims[3] != head_dim || dv_accum_dims[0] != batch ||
      dv_accum_dims[1] != kv_heads || dv_accum_dims[2] != seqlen_k_rounded ||
      dv_accum_dims[3] != value_dim) {
    return Invalid("FA3 backward workspace dimensions do not match the ABI");
  }

  cudaStream_t stream = reinterpret_cast<cudaStream_t>(platform_stream);
  if (q_heads != kv_heads) {
    cudaError_t status = cudaMemsetAsync(
        dk_accum->untyped_data(), 0,
        batch * kv_heads * seqlen_k_rounded * head_dim * sizeof(float), stream);
    if (status != cudaSuccess) {
      return CudaError(status, "cudaMemsetAsync dK workspace");
    }

    status = cudaMemsetAsync(dv_accum->untyped_data(), 0,
                             batch * kv_heads * seqlen_k_rounded * value_dim *
                                 sizeof(float),
                             stream);
    if (status != cudaSuccess) {
      return CudaError(status, "cudaMemsetAsync dV workspace");
    }
  }

  Flash_bwd_params params{};
  params.q_ptr = q.untyped_data();
  params.k_ptr = k.untyped_data();
  params.v_ptr = v.untyped_data();
  params.o_ptr = output.untyped_data();
  params.softmax_lse_ptr = lse.untyped_data();
  params.do_ptr = doutput.untyped_data();
  params.dq_ptr = dq->untyped_data();
  params.dk_ptr = dk->untyped_data();
  params.dv_ptr = dv->untyped_data();

  params.q_batch_stride = seqlen_q * q_heads * head_dim;
  params.k_batch_stride = seqlen_k * kv_heads * head_dim;
  params.v_batch_stride = seqlen_k * kv_heads * value_dim;
  params.o_batch_stride = seqlen_q * q_heads * value_dim;
  params.do_batch_stride = seqlen_q * q_heads * value_dim;
  params.dq_batch_stride = seqlen_q * q_heads * head_dim;
  params.dk_batch_stride = seqlen_k * kv_heads * head_dim;
  params.dv_batch_stride = seqlen_k * kv_heads * value_dim;
  params.q_row_stride = q_heads * head_dim;
  params.k_row_stride = kv_heads * head_dim;
  params.v_row_stride = kv_heads * value_dim;
  params.o_row_stride = q_heads * value_dim;
  params.do_row_stride = q_heads * value_dim;
  params.dq_row_stride = q_heads * head_dim;
  params.dk_row_stride = kv_heads * head_dim;
  params.dv_row_stride = kv_heads * value_dim;
  params.q_head_stride = head_dim;
  params.k_head_stride = head_dim;
  params.v_head_stride = value_dim;
  params.o_head_stride = value_dim;
  params.do_head_stride = value_dim;
  params.dq_head_stride = head_dim;
  params.dk_head_stride = head_dim;
  params.dv_head_stride = value_dim;
  params.v_dim_stride = 1;

  params.b = static_cast<int>(batch);
  params.b_k = static_cast<int>(batch);
  params.h = static_cast<int>(q_heads);
  params.h_k = static_cast<int>(kv_heads);
  params.seqlen_q = static_cast<int>(seqlen_q);
  params.seqlen_k = static_cast<int>(seqlen_k);
  params.seqlen_q_rounded = seqlen_q_rounded;
  params.seqlen_k_rounded = seqlen_k_rounded;
  params.total_q = static_cast<int>(batch * seqlen_q);
  params.total_k = static_cast<int>(batch * seqlen_k);
  params.d = static_cast<int>(head_dim);
  params.d_rounded = static_cast<int>(head_dim);
  params.dv = static_cast<int>(value_dim);
  params.dv_rounded = static_cast<int>(value_dim);

  params.scale_softmax = softmax_scale;
  params.softcap = 0.0f;
  params.p_dropout = 1.0f;
  params.p_dropout_in_uint8_t = 255;
  params.rp_dropout = 1.0f;
  params.is_bf16 = std::is_same_v<KernelType, cutlass::bfloat16_t>;
  params.is_fp32 = false;
  params.is_e4m3 = false;
  params.is_causal = causal;
  params.is_local = false;
  params.window_size_left = static_cast<int>(seqlen_k - 1);
  params.window_size_right = causal ? 0 : static_cast<int>(seqlen_q - 1);
  params.attention_chunk = 0;
  params.arch = 90;
  params.num_sm = *num_sm;
  params.num_splits = 1;
  params.pack_gqa = false;
  params.deterministic = false;

  params.dsoftmax_sum = softmax_d->untyped_data();
  params.softmax_lse_log2_ptr = softmax_lse_log2->untyped_data();
  params.dq_accum_ptr = dq_accum->untyped_data();
  params.dq_semaphore = static_cast<int *>(dq_semaphore->untyped_data());
  params.dk_accum_ptr =
      q_heads != kv_heads ? dk_accum->untyped_data() : nullptr;
  params.dv_accum_ptr =
      q_heads != kv_heads ? dv_accum->untyped_data() : nullptr;

  if (head_dim == 128) {
    run_mha_bwd_<90, KernelType, 128, false>(params, stream);
  } else {
    run_mha_bwd_<90, KernelType, 256, false>(params, stream);
  }

  return ffi::Error::Success();
}

ffi::Error
Fa3BackwardBf16(CUstream stream, int32_t device, ffi::Buffer<ffi::BF16> q,
                ffi::Buffer<ffi::BF16> k, ffi::Buffer<ffi::BF16> v,
                ffi::Buffer<ffi::BF16> output, ffi::Buffer<ffi::F32> lse,
                ffi::Buffer<ffi::BF16> doutput, bool causal,
                float softmax_scale, ffi::Result<ffi::Buffer<ffi::BF16>> dq,
                ffi::Result<ffi::Buffer<ffi::BF16>> dk,
                ffi::Result<ffi::Buffer<ffi::BF16>> dv,
                ffi::Result<ffi::Buffer<ffi::F32>> softmax_d,
                ffi::Result<ffi::Buffer<ffi::F32>> lse_log2,
                ffi::Result<ffi::Buffer<ffi::F32>> dq_accum,
                ffi::Result<ffi::Buffer<ffi::S32>> dq_semaphore,
                ffi::Result<ffi::Buffer<ffi::F32>> dk_accum,
                ffi::Result<ffi::Buffer<ffi::F32>> dv_accum) {
  return Fa3BackwardTyped<ffi::BF16, cutlass::bfloat16_t>(
      stream, device, q, k, v, output, lse, doutput, causal, softmax_scale, dq,
      dk, dv, softmax_d, lse_log2, dq_accum, dq_semaphore, dk_accum, dv_accum);
}

ffi::Error Fa3BackwardF16(CUstream stream, int32_t device,
                          ffi::Buffer<ffi::F16> q, ffi::Buffer<ffi::F16> k,
                          ffi::Buffer<ffi::F16> v, ffi::Buffer<ffi::F16> output,
                          ffi::Buffer<ffi::F32> lse,
                          ffi::Buffer<ffi::F16> doutput, bool causal,
                          float softmax_scale,
                          ffi::Result<ffi::Buffer<ffi::F16>> dq,
                          ffi::Result<ffi::Buffer<ffi::F16>> dk,
                          ffi::Result<ffi::Buffer<ffi::F16>> dv,
                          ffi::Result<ffi::Buffer<ffi::F32>> softmax_d,
                          ffi::Result<ffi::Buffer<ffi::F32>> lse_log2,
                          ffi::Result<ffi::Buffer<ffi::F32>> dq_accum,
                          ffi::Result<ffi::Buffer<ffi::S32>> dq_semaphore,
                          ffi::Result<ffi::Buffer<ffi::F32>> dk_accum,
                          ffi::Result<ffi::Buffer<ffi::F32>> dv_accum) {
  return Fa3BackwardTyped<ffi::F16, cutlass::half_t>(
      stream, device, q, k, v, output, lse, doutput, causal, softmax_scale, dq,
      dk, dv, softmax_d, lse_log2, dq_accum, dq_semaphore, dk_accum, dv_accum);
}

class Fa3Partitioner final : public xla::CustomCallPartitioner {
public:
  bool
  IsCustomCallShardable(const xla::HloInstruction *instruction) const override {
    return instruction->operand_count() == 3 &&
           instruction->shape().IsTuple() &&
           instruction->shape().tuple_shapes_size() == 3;
  }

  std::optional<xla::HloSharding>
  InferShardingFromOperands(const xla::HloInstruction *) const override {
    return std::nullopt;
  }

  bool
  CanPropagateShardingToOperands(const xla::HloInstruction *) const override {
    return false;
  }

  absl::Status Partition(xla::spmd::SpmdPartitioningVisitor *partitioner,
                         xla::HloInstruction *hlo) const override {
    if (!IsCustomCallShardable(hlo)) {
      return absl::InvalidArgumentError(
          "FA3 custom call expects Q/K/V and tuple(O,LSE,workspace)");
    }

    auto q = partitioner->GetPartitionedHlo(hlo->operand(0));
    auto k = partitioner->GetPartitionedHlo(hlo->operand(1));
    auto v = partitioner->GetPartitionedHlo(hlo->operand(2));

    const xla::Shape &global_q = hlo->operand(0)->shape();
    const xla::Shape &global_k = hlo->operand(1)->shape();
    const xla::Shape &global_v = hlo->operand(2)->shape();
    const xla::Shape &local_q = q.hlo()->shape();
    const xla::Shape &local_k = k.hlo()->shape();
    const xla::Shape &local_v = v.hlo()->shape();

    if (local_q.dimensions(0) != global_q.dimensions(0) ||
        local_q.dimensions(1) != global_q.dimensions(1) ||
        local_q.dimensions(3) != global_q.dimensions(3) ||
        local_k.dimensions(0) != global_k.dimensions(0) ||
        local_k.dimensions(1) != global_k.dimensions(1) ||
        local_k.dimensions(3) != global_k.dimensions(3) ||
        local_v.dimensions(0) != global_v.dimensions(0) ||
        local_v.dimensions(1) != global_v.dimensions(1) ||
        local_v.dimensions(3) != global_v.dimensions(3)) {
      return absl::InvalidArgumentError(
          "FA3 only permits head-axis tensor parallelism");
    }

    if (local_k.dimensions(2) != local_v.dimensions(2) ||
        local_k.dimensions(2) <= 0 ||
        local_q.dimensions(2) % local_k.dimensions(2) != 0 ||
        global_q.dimensions(2) / global_k.dimensions(2) !=
            local_q.dimensions(2) / local_k.dimensions(2)) {
      return absl::InvalidArgumentError(
          "FA3 tensor parallelism must preserve complete KV groups");
    }

    xla::HloSharding output_sharding =
        hlo->sharding().NormalizeTupleSharding(hlo->shape());
    std::vector<xla::Shape> local_results;
    local_results.reserve(3);
    for (int i = 0; i < 3; ++i) {
      xla::HloSharding element_sharding =
          output_sharding.GetSubSharding(hlo->shape(), {i});
      local_results.push_back(xla::spmd::MakePartitionedShape(
          hlo->shape().tuple_shapes(i), element_sharding));
    }

    xla::Shape local_tuple = xla::ShapeUtil::MakeTupleShape(local_results);
    xla::HloInstruction *local_call = partitioner->builder()->AddInstruction(
        hlo->CloneWithNewOperands(local_tuple, {q.hlo(), k.hlo(), v.hlo()}));

    auto *original_custom_call =
        static_cast<xla::HloCustomCallInstruction *>(hlo);
    auto *local_custom_call =
        static_cast<xla::HloCustomCallInstruction *>(local_call);
    if (original_custom_call->layout_constrained()) {
      std::vector<xla::Shape> local_operand_layouts =
          original_custom_call->operand_shapes_with_layout();
      const std::array<const xla::Shape *, 3> local_operand_shapes = {
          &local_q, &local_k, &local_v};
      for (int operand = 0; operand < 3; ++operand) {
        for (int dim = 0;
             dim < static_cast<int>(
                       local_operand_layouts[operand].dimensions().size());
             ++dim) {
          local_operand_layouts[operand].set_dimensions(
              dim, local_operand_shapes[operand]->dimensions(dim));
        }
      }
      local_custom_call->set_operand_shapes_with_layout(
          std::move(local_operand_layouts));
    }
    local_call->set_sharding(output_sharding);

    xla::spmd::PartitionedHlo result(local_call, hlo->shape(),
                                     partitioner->MakePartitioningState());
    partitioner->SetPartitionedHlo(hlo, result.Reshard(output_sharding));
    return absl::OkStatus();
  }
};

class Fa3BackwardPartitioner final : public xla::CustomCallPartitioner {
public:
  bool
  IsCustomCallShardable(const xla::HloInstruction *instruction) const override {
    return instruction->operand_count() == 6 &&
           instruction->shape().IsTuple() &&
           instruction->shape().tuple_shapes_size() == 9;
  }

  std::optional<xla::HloSharding>
  InferShardingFromOperands(const xla::HloInstruction *) const override {
    return std::nullopt;
  }

  bool
  CanPropagateShardingToOperands(const xla::HloInstruction *) const override {
    return false;
  }

  absl::Status Partition(xla::spmd::SpmdPartitioningVisitor *partitioner,
                         xla::HloInstruction *hlo) const override {
    if (!IsCustomCallShardable(hlo)) {
      return absl::InvalidArgumentError(
          "FA3 backward expects Q/K/V/O/LSE/dO and gradients plus workspaces");
    }

    std::array<xla::spmd::PartitionedHlo, 6> operands = {
        partitioner->GetPartitionedHlo(hlo->operand(0)),
        partitioner->GetPartitionedHlo(hlo->operand(1)),
        partitioner->GetPartitionedHlo(hlo->operand(2)),
        partitioner->GetPartitionedHlo(hlo->operand(3)),
        partitioner->GetPartitionedHlo(hlo->operand(4)),
        partitioner->GetPartitionedHlo(hlo->operand(5))};

    const xla::Shape &global_q = hlo->operand(0)->shape();
    const xla::Shape &global_k = hlo->operand(1)->shape();
    const xla::Shape &global_v = hlo->operand(2)->shape();
    const xla::Shape &local_q = operands[0].hlo()->shape();
    const xla::Shape &local_k = operands[1].hlo()->shape();
    const xla::Shape &local_v = operands[2].hlo()->shape();

    if (local_q.dimensions(0) != global_q.dimensions(0) ||
        local_q.dimensions(1) != global_q.dimensions(1) ||
        local_q.dimensions(3) != global_q.dimensions(3) ||
        local_k.dimensions(0) != global_k.dimensions(0) ||
        local_k.dimensions(1) != global_k.dimensions(1) ||
        local_k.dimensions(3) != global_k.dimensions(3) ||
        local_v.dimensions(0) != global_v.dimensions(0) ||
        local_v.dimensions(1) != global_v.dimensions(1) ||
        local_v.dimensions(3) != global_v.dimensions(3) ||
        local_k.dimensions(2) != local_v.dimensions(2) ||
        local_k.dimensions(2) <= 0 ||
        local_q.dimensions(2) % local_k.dimensions(2) != 0 ||
        global_q.dimensions(2) / global_k.dimensions(2) !=
            local_q.dimensions(2) / local_k.dimensions(2)) {
      return absl::InvalidArgumentError(
          "FA3 backward only permits complete KV-group head sharding");
    }

    xla::HloSharding output_sharding =
        hlo->sharding().NormalizeTupleSharding(hlo->shape());
    std::vector<xla::Shape> local_results;
    local_results.reserve(9);
    for (int i = 0; i < 9; ++i) {
      xla::HloSharding element_sharding =
          output_sharding.GetSubSharding(hlo->shape(), {i});
      local_results.push_back(xla::spmd::MakePartitionedShape(
          hlo->shape().tuple_shapes(i), element_sharding));
    }

    std::vector<xla::HloInstruction *> local_operands;
    local_operands.reserve(6);
    for (auto &operand : operands) {
      local_operands.push_back(operand.hlo());
    }

    xla::Shape local_tuple = xla::ShapeUtil::MakeTupleShape(local_results);
    xla::HloInstruction *local_call = partitioner->builder()->AddInstruction(
        hlo->CloneWithNewOperands(local_tuple, local_operands));

    auto *original_custom_call =
        static_cast<xla::HloCustomCallInstruction *>(hlo);
    auto *local_custom_call =
        static_cast<xla::HloCustomCallInstruction *>(local_call);
    if (original_custom_call->layout_constrained()) {
      std::vector<xla::Shape> local_operand_layouts =
          original_custom_call->operand_shapes_with_layout();
      for (int operand = 0; operand < 6; ++operand) {
        const xla::Shape &local_shape = operands[operand].hlo()->shape();
        for (int dim = 0;
             dim < static_cast<int>(
                       local_operand_layouts[operand].dimensions().size());
             ++dim) {
          local_operand_layouts[operand].set_dimensions(
              dim, local_shape.dimensions(dim));
        }
      }
      local_custom_call->set_operand_shapes_with_layout(
          std::move(local_operand_layouts));
    }
    local_call->set_sharding(output_sharding);

    xla::spmd::PartitionedHlo result(local_call, hlo->shape(),
                                     partitioner->MakePartitioningState());
    partitioner->SetPartitionedHlo(hlo, result.Reshard(output_sharding));
    return absl::OkStatus();
  }
};

const bool kPartitionerRegistered = [] {
  xla::RegisterCustomCallPartitioner(kTarget,
                                     std::make_unique<Fa3Partitioner>());
  xla::RegisterCustomCallPartitioner(kTargetF16,
                                     std::make_unique<Fa3Partitioner>());
  xla::RegisterCustomCallPartitioner(
      kBackwardTarget, std::make_unique<Fa3BackwardPartitioner>());
  xla::RegisterCustomCallPartitioner(
      kBackwardTargetF16, std::make_unique<Fa3BackwardPartitioner>());
  return true;
}();

} // namespace

XLA_FFI_DEFINE_HANDLER_SYMBOL(exla_fa3_forward, Fa3ForwardBf16,
                              ffi::Ffi::Bind()
                                  .Ctx<ffi::PlatformStream<CUstream>>()
                                  .Ctx<ffi::DeviceOrdinal>()
                                  .Arg<ffi::Buffer<ffi::BF16>>()
                                  .Arg<ffi::Buffer<ffi::BF16>>()
                                  .Arg<ffi::Buffer<ffi::BF16>>()
                                  .Attr<bool>("causal")
                                  .Attr<float>("softmax_scale")
                                  .Ret<ffi::Buffer<ffi::BF16>>()
                                  .Ret<ffi::Buffer<ffi::F32>>()
                                  .Ret<ffi::Buffer<ffi::S32>>(),
                              {ffi::Traits::kCmdBufferCompatible});

XLA_FFI_REGISTER_HANDLER(ffi::GetXlaFfiApi(), kTarget, "CUDA",
                         exla_fa3_forward);

XLA_FFI_DEFINE_HANDLER_SYMBOL(exla_fa3_forward_f16, Fa3ForwardF16,
                              ffi::Ffi::Bind()
                                  .Ctx<ffi::PlatformStream<CUstream>>()
                                  .Ctx<ffi::DeviceOrdinal>()
                                  .Arg<ffi::Buffer<ffi::F16>>()
                                  .Arg<ffi::Buffer<ffi::F16>>()
                                  .Arg<ffi::Buffer<ffi::F16>>()
                                  .Attr<bool>("causal")
                                  .Attr<float>("softmax_scale")
                                  .Ret<ffi::Buffer<ffi::F16>>()
                                  .Ret<ffi::Buffer<ffi::F32>>()
                                  .Ret<ffi::Buffer<ffi::S32>>(),
                              {ffi::Traits::kCmdBufferCompatible});

XLA_FFI_REGISTER_HANDLER(ffi::GetXlaFfiApi(), kTargetF16, "CUDA",
                         exla_fa3_forward_f16);

XLA_FFI_DEFINE_HANDLER_SYMBOL(exla_fa3_backward, Fa3BackwardBf16,
                              ffi::Ffi::Bind()
                                  .Ctx<ffi::PlatformStream<CUstream>>()
                                  .Ctx<ffi::DeviceOrdinal>()
                                  .Arg<ffi::Buffer<ffi::BF16>>()
                                  .Arg<ffi::Buffer<ffi::BF16>>()
                                  .Arg<ffi::Buffer<ffi::BF16>>()
                                  .Arg<ffi::Buffer<ffi::BF16>>()
                                  .Arg<ffi::Buffer<ffi::F32>>()
                                  .Arg<ffi::Buffer<ffi::BF16>>()
                                  .Attr<bool>("causal")
                                  .Attr<float>("softmax_scale")
                                  .Ret<ffi::Buffer<ffi::BF16>>()
                                  .Ret<ffi::Buffer<ffi::BF16>>()
                                  .Ret<ffi::Buffer<ffi::BF16>>()
                                  .Ret<ffi::Buffer<ffi::F32>>()
                                  .Ret<ffi::Buffer<ffi::F32>>()
                                  .Ret<ffi::Buffer<ffi::F32>>()
                                  .Ret<ffi::Buffer<ffi::S32>>()
                                  .Ret<ffi::Buffer<ffi::F32>>()
                                  .Ret<ffi::Buffer<ffi::F32>>(),
                              {ffi::Traits::kCmdBufferCompatible});

XLA_FFI_REGISTER_HANDLER(ffi::GetXlaFfiApi(), kBackwardTarget, "CUDA",
                         exla_fa3_backward);

XLA_FFI_DEFINE_HANDLER_SYMBOL(exla_fa3_backward_f16, Fa3BackwardF16,
                              ffi::Ffi::Bind()
                                  .Ctx<ffi::PlatformStream<CUstream>>()
                                  .Ctx<ffi::DeviceOrdinal>()
                                  .Arg<ffi::Buffer<ffi::F16>>()
                                  .Arg<ffi::Buffer<ffi::F16>>()
                                  .Arg<ffi::Buffer<ffi::F16>>()
                                  .Arg<ffi::Buffer<ffi::F16>>()
                                  .Arg<ffi::Buffer<ffi::F32>>()
                                  .Arg<ffi::Buffer<ffi::F16>>()
                                  .Attr<bool>("causal")
                                  .Attr<float>("softmax_scale")
                                  .Ret<ffi::Buffer<ffi::F16>>()
                                  .Ret<ffi::Buffer<ffi::F16>>()
                                  .Ret<ffi::Buffer<ffi::F16>>()
                                  .Ret<ffi::Buffer<ffi::F32>>()
                                  .Ret<ffi::Buffer<ffi::F32>>()
                                  .Ret<ffi::Buffer<ffi::F32>>()
                                  .Ret<ffi::Buffer<ffi::S32>>()
                                  .Ret<ffi::Buffer<ffi::F32>>()
                                  .Ret<ffi::Buffer<ffi::F32>>(),
                              {ffi::Traits::kCmdBufferCompatible});

XLA_FFI_REGISTER_HANDLER(ffi::GetXlaFfiApi(), kBackwardTargetF16, "CUDA",
                         exla_fa3_backward_f16);

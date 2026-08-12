# Laguna FP8 on SM70

Laguna checkpoints are trained for bfloat16 execution, but V100 serving uses float16 activations. The model can
produce finite expert values above the float16 range in late sparse layers, so a float16 residual stream is not a
valid fallback for this architecture. The SM70 path keeps the inexpensive inner products in float16 while preserving
the model's output range at block boundaries.

## Precision contract

When a Laguna model runs with float16 model dtype, the architecture requests this explicit mixed-dtype contract:

- the embedding output is promoted when it enters the float32 transformer residual stream;
- attention QKV, gates, expert gate/up products, and normalized activations remain float16;
- attention output projection, dense and shared-expert down projections, and routed-expert final outputs are float32;
- tensor-parallel reduction and the shared-plus-routed expert combination happen in float32; and
- the final RMS normalization returns float16 for the output head.

The generic `FusedMoEConfig.out_dtype` and `FusedMoE(output_dtype=...)` interfaces describe only the final expert
output. Intermediate workspaces continue to use `in_dtype`. Likewise, `RowParallelLinear(reduce_output_dtype=...)`
casts each rank-local result before the all-reduce. Both options default to the prior same-dtype behavior, and
same-dtype-only fused reducers fail closed when a caller requests mixed output.

`LagunaRMSNorm` owns the float32 residual addition and returns a float16 normalized activation. Other architectures,
and Laguna executions that do not use float16 model dtype, retain the standard normalization and projection paths.

## Native FP8 MoE tables

TurboMind grouped GEMM consumes one `{pointer, leading_dimension}` row per expert. These persistent tables are
allocated as PyTorch tensors on the weight device and filled directly on the current stream. They must not use a
second `cudaMallocAsync` allocation followed by a copy: near model capacity that allocator can fail outside
PyTorch's reserved pool, and launching the fill kernel with the unchecked null result corrupts the CUDA context.

The owning custom operation validates tensor devices and expert bounds, fills exactly one 16-byte row per expert,
and performs a CUDA launch check before returning the tensors that the layer retains. The native SM70 FP8 route keeps
the prepared FP8 weights and float16 inner workspaces; only its routed reduction writes the requested float32 final
output. Native shortcuts that require matching input and output types remain disabled for this mixed-dtype contract.

## Verification

The contract is covered at each boundary: MoE configuration propagation, mixed workspace allocation, Triton and
CUDA reductions, float16-to-float32 unpermute, cast-before-all-reduce, Laguna residual behavior, and unchanged
defaults for other models. SM70 CUDA tests retain 48 layers of 256-expert W13 and W2 pointer tables, verify pointer
and stride values on multiple devices, and guard the source allocations against out-of-bounds writes.

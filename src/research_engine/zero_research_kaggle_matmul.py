"""Minimal reviewed Triton matmul used only by the offline Kaggle template."""

from __future__ import annotations


_KERNEL = None


def _kernel():
    global _KERNEL
    if _KERNEL is not None:
        return _KERNEL
    import triton
    import triton.language as tl

    @triton.jit
    def kernel(
        a_ptr, b_ptr, c_ptr,
        M, N, K,
        stride_am, stride_ak, stride_bk, stride_bn, stride_cm, stride_cn,
        BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr,
        BLOCK_SIZE_K: tl.constexpr, GROUP_SIZE_M: tl.constexpr,
    ):
        pid = tl.program_id(axis=0)
        num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
        num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
        num_pid_in_group = GROUP_SIZE_M * num_pid_n
        group_id = pid // num_pid_in_group
        first_pid_m = group_id * GROUP_SIZE_M
        group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
        pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
        pid_n = (pid % num_pid_in_group) // group_size_m
        offs_am = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)) % M
        offs_bn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) % N
        offs_k = tl.arange(0, BLOCK_SIZE_K)
        a_ptrs = a_ptr + offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak
        b_ptrs = b_ptr + offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn
        accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
        for k_offset in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
            remaining = K - k_offset * BLOCK_SIZE_K
            a = tl.load(a_ptrs, mask=offs_k[None, :] < remaining, other=0.0)
            b = tl.load(b_ptrs, mask=offs_k[:, None] < remaining, other=0.0)
            accumulator = tl.dot(a, b, accumulator)
            a_ptrs += BLOCK_SIZE_K * stride_ak
            b_ptrs += BLOCK_SIZE_K * stride_bk
        offs_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
        offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
        c_ptrs = c_ptr + stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]
        tl.store(c_ptrs, accumulator.to(c_ptr.dtype.element_ty), mask=(offs_cm[:, None] < M) & (offs_cn[None, :] < N))

    _KERNEL = kernel
    return kernel


def matmul(a, b, config: dict[str, int]):
    import torch
    import triton

    if a.shape[1] != b.shape[0] or not a.is_contiguous() or not b.is_contiguous():
        raise ValueError("Kaggle template matmul requires compatible contiguous tensors")
    m, k = a.shape; _k, n = b.shape
    output = torch.empty((m, n), device=a.device, dtype=torch.float16)
    grid = lambda meta: (triton.cdiv(m, meta["BLOCK_SIZE_M"]) * triton.cdiv(n, meta["BLOCK_SIZE_N"]),)
    _kernel()[grid](a, b, output, m, n, k, a.stride(0), a.stride(1), b.stride(0), b.stride(1), output.stride(0), output.stride(1), BLOCK_SIZE_M=config["BLOCK_SIZE_M"], BLOCK_SIZE_N=config["BLOCK_SIZE_N"], BLOCK_SIZE_K=config["BLOCK_SIZE_K"], GROUP_SIZE_M=config["GROUP_SIZE_M"], num_warps=config["num_warps"], num_stages=config["num_stages"])
    return output

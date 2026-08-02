// tensor.rs
//
// Two responsibilities here: turning a compressed on-disk tensor (Q8_0 or
// plain F32) into a flat Vec<f32> we can do math on, and multiplying two
// such tensors together. Everything downstream (attention, MLP, etc.) is
// built out of these two operations.

use std::fs::File;
use std::io::{Read, Seek, SeekFrom};
use crate::gguf::TensorInfo;

const Q8_0_BLOCK_SIZE: usize = 32; // weights per block
const Q8_0_BLOCK_BYTES: usize = 2 + Q8_0_BLOCK_SIZE; // 1 f16 scale + 32 int8 values

// GGML dtype ids we currently support. Others (Q4_K_M etc.) will panic for now.
const GGML_TYPE_F32: u32 = 0;
const GGML_TYPE_Q8_0: u32 = 8;

// Half-precision (f16) to full f32 conversion, done manually via bit
// manipulation since f16 has no native Rust type in std. f16 layout is
// 1 sign bit, 5 exponent bits, 10 mantissa bits.
fn f16_to_f32(bits: u16) -> f32 {
    let sign = (bits >> 15) & 0x1;
    let exponent = (bits >> 10) & 0x1F;
    let mantissa = bits & 0x3FF;

    let value: f32 = if exponent == 0 {
        // subnormal number
        (mantissa as f32) * 2f32.powi(-24)
    } else if exponent == 0x1F {
        if mantissa == 0 { f32::INFINITY } else { f32::NAN }
    } else {
        let e = exponent as i32 - 15; // remove f16 bias
        (1.0 + mantissa as f32 / 1024.0) * 2f32.powi(e)
    };

    if sign == 1 { -value } else { value }
}

pub fn total_elements(dims: &[u64]) -> usize {
    dims.iter().product::<u64>() as usize
}

// Reads a tensor's raw bytes from disk and returns it as a flat, row-major
// Vec<f32>, regardless of how it was stored on disk.
pub fn load_tensor(path: &str, tensor_data_offset: u64, info: &TensorInfo) -> Vec<f32> {
    let n_elements = total_elements(&info.dims);
    let mut file = File::open(path).expect("failed to open GGUF file");

    let abs_offset = tensor_data_offset + info.offset;
    file.seek(SeekFrom::Start(abs_offset)).expect("seek failed");

    match info.dtype {
        GGML_TYPE_F32 => {
            let mut buf = vec![0u8; n_elements * 4];
            file.read_exact(&mut buf).expect("read failed");
            buf.chunks_exact(4)
                .map(|b| f32::from_le_bytes(b.try_into().unwrap()))
                .collect()
        }
        GGML_TYPE_Q8_0 => {
            assert_eq!(n_elements % Q8_0_BLOCK_SIZE, 0, "Q8_0 tensor size must be a multiple of 32");
            let n_blocks = n_elements / Q8_0_BLOCK_SIZE;
            let mut buf = vec![0u8; n_blocks * Q8_0_BLOCK_BYTES];
            file.read_exact(&mut buf).expect("read failed");

            let mut out = Vec::with_capacity(n_elements);
            for block in buf.chunks_exact(Q8_0_BLOCK_BYTES) {
                let scale_bits = u16::from_le_bytes([block[0], block[1]]);
                let scale = f16_to_f32(scale_bits);
                for &byte in &block[2..2 + Q8_0_BLOCK_SIZE] {
                    let q = byte as i8; // reinterpret as signed
                    out.push(q as f32 * scale);
                }
            }
            out
        }
        other => panic!("unsupported dtype: {other} (only F32=0 and Q8_0=8 implemented so far)"),
    }
}

// Raw FFI binding to the CBLAS standard's sgemm ("single-precision general
// matrix multiply") function, provided by Apple's Accelerate framework.
// This is the same interface llama.cpp itself can call into -- a decades-old,
// heavily optimized standard for this exact operation.
#[allow(non_camel_case_types)]
type CBLAS_ORDER = i32;
#[allow(non_camel_case_types)]
type CBLAS_TRANSPOSE = i32;

const CBLAS_ROW_MAJOR: CBLAS_ORDER = 101;
const CBLAS_NO_TRANS: CBLAS_TRANSPOSE = 111;

    unsafe extern "C" {
    fn cblas_sgemm(
        order: CBLAS_ORDER,
        transa: CBLAS_TRANSPOSE,
        transb: CBLAS_TRANSPOSE,
        m: i32,
        n: i32,
        k: i32,
        alpha: f32,
        a: *const f32,
        lda: i32,
        b: *const f32,
        ldb: i32,
        beta: f32,
        c: *mut f32,
        ldc: i32,
    );
}

// Matrix multiply: a is [m x k], b is [k x n], result is [m x n].
//
// Delegates to Accelerate's cblas_sgemm rather than a hand-written loop.
// The hand-written, cache-aware version (matmul_scalar below) already
// showed batching can scale once memory access is fixed; BLAS goes further,
// using SIMD instructions to process several floats per CPU cycle and
// blocking the computation to fit the processor's cache hierarchy far more
// precisely than a simple loop reorder can.
//
// alpha=1, beta=0 means: compute C = A * B exactly, ignoring whatever
// garbage is currently in C's memory rather than accumulating into it.
pub fn matmul(a: &[f32], m: usize, k: usize, b: &[f32], n: usize) -> Vec<f32> {
    assert_eq!(a.len(), m * k, "matrix a has wrong length for given dims");
    assert_eq!(b.len(), k * n, "matrix b has wrong length for given dims");

    let mut out = vec![0f32; m * n];

    unsafe {
        cblas_sgemm(
            CBLAS_ROW_MAJOR,
            CBLAS_NO_TRANS,
            CBLAS_NO_TRANS,
            m as i32,
            n as i32,
            k as i32,
            1.0,
            a.as_ptr(),
            k as i32,
            b.as_ptr(),
            n as i32,
            0.0,
            out.as_mut_ptr(),
            n as i32,
        );
    }

    out
}

// The hand-written, cache-aware matmul from Milestone 7 stage 1. Kept as a
// named reference and a correctness cross-check against BLAS -- both
// should produce identical results, since they compute the same operation.
pub fn matmul_scalar(a: &[f32], m: usize, k: usize, b: &[f32], n: usize) -> Vec<f32> {
    assert_eq!(a.len(), m * k, "matrix a has wrong length for given dims");
    assert_eq!(b.len(), k * n, "matrix b has wrong length for given dims");

    let mut out = vec![0f32; m * n];

    for i in 0..m {
        let a_row = &a[i * k..(i + 1) * k];
        let out_row = &mut out[i * n..(i + 1) * n];

        for p in 0..k {
            let a_val = a_row[p];
            if a_val == 0.0 {
                continue;
            }
            let b_row = &b[p * n..(p + 1) * n];

            for j in 0..n {
                out_row[j] += a_val * b_row[j];
            }
        }
    }

    out
}
// build.rs
//
// Links Apple's Accelerate framework, which provides a professionally
// optimized BLAS (Basic Linear Algebra Subprograms) implementation --
// SIMD-vectorized, cache-blocked, and in some cases multi-threaded matrix
// multiplication. This is what tensor.rs calls into via cblas_sgemm instead
// of the hand-written loop.
fn main() {
    #[cfg(target_os = "macos")]
    println!("cargo:rustc-link-lib=framework=Accelerate");
}
//! x86-64 kernel slot. Currently aliases the scalar tables; the real AVX2 /
//! SSSE3 kernels with a runtime pick replace this file (same four
//! signatures).

pub(crate) use super::scalar::{decode_16, decode_23, encode_16, encode_23};

//! NEON kernel slot. aarch64 always has NEON, so the parent module calls
//! these unconditionally. Currently aliases the scalar tables; the real
//! vector kernel replaces this file (same four signatures).

pub(crate) use super::scalar::{decode_16, decode_23, encode_16, encode_23};

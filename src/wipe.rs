//! Secure-erasure helpers for buffers that held key material.
//!
//! A plain `buf.fill(0)` on a buffer that is never read again is a dead
//! store, and `-O3` is entitled to remove it — exactly what a wipe must not
//! be. These write through [`core::ptr::write_volatile`], which the
//! optimizer may not elide, and seal the stores with a
//! [`compiler_fence`](core::sync::atomic::compiler_fence) so they cannot be
//! reordered past whatever happens after the wipe.
//!
//! This module is the crate's only hand-rolled wipe and the only place
//! `unsafe` appears in the scalar core; the `zeroize` *feature* gates its
//! use at the call sites.

use core::sync::atomic::{Ordering, compiler_fence};

/// Overwrite `xs` with zeros in a way the optimizer cannot elide.
pub(crate) fn secure_wipe_u32(xs: &mut [u32]) {
    for x in xs {
        // SAFETY: `x` is derived from the live, exclusively-borrowed slice
        // `xs`, so it is valid, aligned, and writable for the duration of
        // this call — that is exactly what the safe `&mut [u32]` signature
        // guarantees.
        unsafe { core::ptr::write_volatile(x, 0) };
    }
    compiler_fence(Ordering::SeqCst);
}

/// Overwrite `xs` with zeros in a way the optimizer cannot elide.
pub(crate) fn secure_wipe_bytes(xs: &mut [u8]) {
    for x in xs {
        // SAFETY: same invariant as `secure_wipe_u32` — the exclusive slice
        // borrow makes every element pointer valid, aligned, and writable.
        unsafe { core::ptr::write_volatile(x, 0) };
    }
    compiler_fence(Ordering::SeqCst);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn wipe_zeroes_every_element() {
        let mut words = [0xDEADBEEFu32; 37];
        secure_wipe_u32(&mut words);
        assert!(words.iter().all(|&w| w == 0));

        let mut bytes = [0xA5u8; 64];
        secure_wipe_bytes(&mut bytes);
        assert!(bytes.iter().all(|&b| b == 0));
    }

    #[test]
    fn wipe_tolerates_empty_slices() {
        secure_wipe_u32(&mut []);
        secure_wipe_bytes(&mut []);
    }
}

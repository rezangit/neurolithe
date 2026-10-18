//! The one place an embedding crosses the storage boundary: `[f32]` ↔ the
//! float32 blob a sqlite-vec `vec0` table stores, plus the dimension check
//! every writer applies before a vector reaches a store.
//!
//! sqlite-vec reads a float32 blob as raw native-endian `float`s. Every target
//! NeuroLithe builds for is little-endian, so the blob is also the portable
//! little-endian layout that older stores and fixtures were written with.

use anyhow::{Result, bail, ensure};

/// Bytes per stored vector element (float32).
const F32_BYTES: usize = std::mem::size_of::<f32>();

/// View an embedding as the float32 blob sqlite-vec expects (zero-copy).
pub fn as_blob(embedding: &[f32]) -> &[u8] {
    // SAFETY: `f32` has no padding or invalid bit patterns, and `u8` has
    // alignment 1; the byte slice borrows the same memory for the same
    // lifetime with the exact byte length.
    unsafe {
        std::slice::from_raw_parts(
            embedding.as_ptr() as *const u8,
            std::mem::size_of_val(embedding),
        )
    }
}

/// Decode a sqlite-vec float32 blob back into an embedding. Errors on a blob
/// whose length is not a whole number of float32 elements (a corrupt row).
pub fn from_blob(blob: &[u8]) -> Result<Vec<f32>> {
    ensure!(
        blob.len().is_multiple_of(F32_BYTES),
        "corrupt vector blob: {} bytes is not a whole number of float32 values",
        blob.len()
    );
    Ok(blob
        .chunks_exact(F32_BYTES)
        .map(|b| f32::from_ne_bytes([b[0], b[1], b[2], b[3]]))
        .collect())
}

/// Reject an embedding whose length is not the store's `expected` dimension —
/// a wrong-dimension vector cannot be indexed (or would corrupt the index).
/// `what` names the vector in the error, e.g. `"embedder (bge-small)"`.
pub fn check_dim(embedding: &[f32], expected: usize, what: &str) -> Result<()> {
    if embedding.len() != expected {
        bail!(
            "{what} returned {} dimensions, expected {expected}",
            embedding.len()
        );
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn blob_round_trips() {
        let v = [1.0f32, -0.5, 0.0, f32::MAX];
        let blob = as_blob(&v);
        assert_eq!(blob.len(), 16);
        assert_eq!(from_blob(blob).unwrap(), v);
    }

    /// The blob layout matches what stores were written with: little-endian
    /// float32 (`to_le_bytes`), as sqlite-vec reads it on every supported target.
    #[cfg(target_endian = "little")]
    #[test]
    fn blob_is_little_endian_float32() {
        let v = [0.25f32, 3.0];
        let le: Vec<u8> = v.iter().flat_map(|f| f.to_le_bytes()).collect();
        assert_eq!(as_blob(&v), le.as_slice());
    }

    #[test]
    fn misaligned_blob_is_an_error() {
        assert!(from_blob(&[0, 0, 128]).is_err());
        assert_eq!(from_blob(&[]).unwrap(), Vec::<f32>::new());
    }

    #[test]
    fn check_dim_names_the_vector() {
        assert!(check_dim(&[0.0; 4], 4, "x").is_ok());
        let err = check_dim(&[0.0; 6], 4, "embedder (m)")
            .unwrap_err()
            .to_string();
        assert_eq!(err, "embedder (m) returned 6 dimensions, expected 4");
    }
}

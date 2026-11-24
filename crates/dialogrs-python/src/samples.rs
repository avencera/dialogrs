//! Converting Python audio buffers to the 16 kHz mono samples dialogrs takes

use pyo3::buffer::PyUntypedBuffer;
use pyo3::exceptions::{PyTypeError, PyValueError};
use pyo3::prelude::*;

/// Copy a 1-D float32 buffer, such as a NumPy array, or a sequence of floats
///
/// Samples are always copied, so the GIL can be released without the caller's
/// buffer changing underneath a run. The copy costs milliseconds per hour of
/// audio, far less than transcribing it
pub(crate) fn extract(samples: &Bound<'_, PyAny>) -> PyResult<Vec<f32>> {
    let Ok(buffer) = PyUntypedBuffer::get(samples) else {
        return samples.extract::<Vec<f32>>().map_err(|_| {
            PyTypeError::new_err("samples must be a float32 buffer or a sequence of floats")
        });
    };

    // the format is checked here because PyO3 0.30's own float32 check accepts
    // the non-native byte order and rejects an explicit native one
    let format = buffer.format().to_bytes();
    let Some(byte_order) = float32_byte_order(format).filter(|_| buffer.item_size() == 4) else {
        return Err(PyTypeError::new_err(format!(
            "samples must be float32, got buffer format '{}'; convert with numpy.asarray(samples, dtype=numpy.float32)",
            String::from_utf8_lossy(format)
        )));
    };
    if buffer.dimensions() != 1 {
        return Err(PyValueError::new_err(format!(
            "samples must be a 1-D array of mono audio, got {} dimensions",
            buffer.dimensions()
        )));
    }

    // an exporter may hand out an empty buffer as a null pointer, which no slice
    // may be built from
    if buffer.item_count() == 0 {
        return Ok(Vec::new());
    }

    let aligned = buffer.buf_ptr().cast::<f32>().is_aligned();
    if byte_order == ByteOrder::Native && buffer.is_c_contiguous() && aligned {
        // copied through raw pointers rather than a `&[f32]`, since another thread
        // that released the GIL, such as a NumPy operation, may write to the buffer
        // while it is read, and a Rust shared slice promises that nothing does
        let count = buffer.item_count();
        let mut samples = Vec::<f32>::with_capacity(count);
        // SAFETY: the buffer is 1-D, C-contiguous, aligned, native-endian float32
        // with `count` items, the export keeps its memory alive for the copy, and
        // the destination has room for `count` items
        unsafe {
            std::ptr::copy_nonoverlapping(
                buffer.buf_ptr().cast::<f32>(),
                samples.as_mut_ptr(),
                count,
            );
            samples.set_len(count);
        }
        return Ok(samples);
    }

    // strided, unaligned, or byte-swapped buffers are read one item at a time
    Ok((0..buffer.item_count())
        .map(|index| {
            // SAFETY: `index` is within the 1-D shape and each item is 4 bytes
            let bits = unsafe { buffer.get_ptr(&[index]).cast::<u32>().read_unaligned() };
            match byte_order {
                ByteOrder::Native => f32::from_bits(bits),
                ByteOrder::Swapped => f32::from_bits(bits.swap_bytes()),
            }
        })
        .collect())
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum ByteOrder {
    Native,
    Swapped,
}

/// Byte order of a struct-module format string for one float32, or `None` for
/// any other type
fn float32_byte_order(format: &[u8]) -> Option<ByteOrder> {
    let (prefix, code) = match format {
        [code] => (b'@', *code),
        [prefix, code] => (*prefix, *code),
        _ => return None,
    };
    if code != b'f' {
        return None;
    }

    let little = match prefix {
        b'@' | b'=' => return Some(ByteOrder::Native),
        b'<' => true,
        b'>' | b'!' => false,
        _ => return None,
    };
    if little == cfg!(target_endian = "little") {
        Some(ByteOrder::Native)
    } else {
        Some(ByteOrder::Swapped)
    }
}

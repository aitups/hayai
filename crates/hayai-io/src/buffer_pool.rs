/// A zeroed, **page-aligned** byte buffer (4 KiB alignment).
///
/// Plain `Vec<u8>` has no alignment guarantee, which blocks:
/// * `O_DIRECT` (Linux) — needs 512/4096-byte aligned buffers + offsets,
/// * zero-copy `CL_MEM_USE_HOST_PTR` — drivers are forced to make hidden copies
///   of unaligned host memory, breaking zero-copy.
pub struct AlignedBuffer {
    ptr: *mut u8,
    len: usize,
}

impl AlignedBuffer {
    /// Page size used for all Hayai streaming buffers.
    pub const ALIGN: usize = 4096;

    /// Allocate a zeroed, page-aligned buffer of `len` bytes.
    pub fn zeroed(len: usize) -> Self {
        let layout = std::alloc::Layout::from_size_align(len.max(1), Self::ALIGN)
            .unwrap_or_else(|_| std::alloc::Layout::from_size_align(1, Self::ALIGN).unwrap());
        let ptr = unsafe { std::alloc::alloc_zeroed(layout) };
        if ptr.is_null() {
            std::alloc::handle_alloc_error(layout);
        }
        Self { ptr, len }
    }

    pub fn as_ptr(&self) -> *const u8 {
        self.ptr
    }

    pub fn as_mut_ptr(&mut self) -> *mut u8 {
        self.ptr
    }

    pub fn len(&self) -> usize {
        self.len
    }

    pub fn as_slice(&self) -> &[u8] {
        unsafe { std::slice::from_raw_parts(self.ptr, self.len) }
    }

    pub fn as_mut_slice(&mut self) -> &mut [u8] {
        unsafe { std::slice::from_raw_parts_mut(self.ptr, self.len) }
    }
}

impl std::ops::Deref for AlignedBuffer {
    type Target = [u8];

    fn deref(&self) -> &[u8] {
        self.as_slice()
    }
}

impl std::ops::DerefMut for AlignedBuffer {
    fn deref_mut(&mut self) -> &mut [u8] {
        self.as_mut_slice()
    }
}

impl Drop for AlignedBuffer {
    fn drop(&mut self) {
        if !self.ptr.is_null() {
            let layout = unsafe {
                std::alloc::Layout::from_size_align_unchecked(self.len.max(1), Self::ALIGN)
            };
            unsafe { std::alloc::dealloc(self.ptr, layout) };
            self.ptr = std::ptr::null_mut();
        }
    }
}

/// Double Buffering (Ping-Pong) Memory Pool for deterministic streaming without reallocation.
///
/// Contains two fixed-size byte buffers:
///  - `active`:   the buffer currently being consumed by compute (GPU kernel or CPU matmul).
///  - `prefetch`: the buffer currently being filled by the I/O thread (next layer).
///
/// Call `swap()` once compute on the active buffer is complete and the prefetch fill is done.
pub struct PingPongBuffer {
    buffer_a: AlignedBuffer,
    buffer_b: AlignedBuffer,
    active_is_a: bool,
    pub layer_size_bytes: usize,
}

impl PingPongBuffer {
    /// Allocates two page-aligned buffers of `layer_size_bytes` each.
    pub fn new(layer_size_bytes: usize) -> Self {
        Self {
            buffer_a: AlignedBuffer::zeroed(layer_size_bytes),
            buffer_b: AlignedBuffer::zeroed(layer_size_bytes),
            active_is_a: true,
            layer_size_bytes,
        }
    }

    /// Gets a read-only slice of the active buffer (current layer being computed).
    pub fn active(&self) -> &[u8] {
        if self.active_is_a { &self.buffer_a } else { &self.buffer_b }
    }

    /// Gets a mutable slice of the active buffer.
    pub fn active_mut(&mut self) -> &mut [u8] {
        if self.active_is_a { &mut self.buffer_a } else { &mut self.buffer_b }
    }

    /// Gets a mutable slice of the prefetch buffer (being filled by I/O thread with layer N+1).
    pub fn prefetch_mut(&mut self) -> &mut [u8] {
        if self.active_is_a { &mut self.buffer_b } else { &mut self.buffer_a }
    }

    /// Gets a read-only slice of the prefetch buffer.
    pub fn prefetch(&self) -> &[u8] {
        if self.active_is_a { &self.buffer_b } else { &self.buffer_a }
    }

    /// Atomically swaps active ↔ prefetch roles. Call once per layer boundary.
    pub fn swap(&mut self) {
        self.active_is_a = !self.active_is_a;
    }

    /// Borrow active (read) and prefetch (write) at once for overlapped I/O+compute.
    pub fn active_and_prefetch_mut(&mut self) -> (&[u8], &mut [u8]) {
        if self.active_is_a {
            (&self.buffer_a, &mut self.buffer_b)
        } else {
            (&self.buffer_b, &mut self.buffer_a)
        }
    }

    /// Returns total allocated bytes (both buffers).
    pub fn total_allocated_bytes(&self) -> usize {
        self.buffer_a.len() + self.buffer_b.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn aligned_buffer_is_page_aligned() {
        for len in [1usize, 4095, 4096, 4097, 2531 * 1024] {
            let buf = AlignedBuffer::zeroed(len);
            assert_eq!(buf.as_ptr() as usize % AlignedBuffer::ALIGN, 0);
            assert_eq!(buf.len(), len);
            assert!(buf.as_slice().iter().all(|&b| b == 0));
        }
    }

    #[test]
    fn aligned_buffer_roundtrip() {
        let mut buf = AlignedBuffer::zeroed(1024);
        buf.as_mut_slice()[..4].copy_from_slice(&[1, 2, 3, 4]);
        assert_eq!(&buf.as_slice()[..4], &[1, 2, 3, 4]);
    }

    #[test]
    fn ping_pong_slots_are_aligned() {
        let pp = PingPongBuffer::new(8192);
        assert_eq!(pp.total_allocated_bytes(), 16384);
        assert_eq!(pp.active().as_ptr() as usize % AlignedBuffer::ALIGN, 0);
        assert_eq!(pp.prefetch().as_ptr() as usize % AlignedBuffer::ALIGN, 0);
    }
}

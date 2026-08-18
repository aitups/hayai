/// Double Buffering (Ping-Pong) Memory Pool for deterministic streaming without reallocation.
///
/// Contains two fixed-size byte buffers:
///  - `active`:   the buffer currently being consumed by compute (GPU kernel or CPU matmul).
///  - `prefetch`: the buffer currently being filled by the I/O thread (next layer).
///
/// Call `swap()` once compute on the active buffer is complete and the prefetch fill is done.
pub struct PingPongBuffer {
    buffer_a: Vec<u8>,
    buffer_b: Vec<u8>,
    active_is_a: bool,
    pub layer_size_bytes: usize,
}

impl PingPongBuffer {
    /// Allocates two buffers of `layer_size_bytes` each.
    pub fn new(layer_size_bytes: usize) -> Self {
        Self {
            buffer_a: vec![0u8; layer_size_bytes],
            buffer_b: vec![0u8; layer_size_bytes],
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

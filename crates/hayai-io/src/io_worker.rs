//! Dedicated background I/O worker (Phase H1).
//!
//! Reuses a single OS thread across prefetch jobs instead of spawning a new
//! thread per block/layer. Jobs run sequentially; [`Self::run`] returns a channel
//! the caller can block on at the exact point the result is needed.

use std::sync::mpsc::{channel, Receiver, Sender};
use std::thread::JoinHandle;

type Job = Box<dyn FnOnce() + Send + 'static>;

/// One background worker thread executing `FnOnce` jobs in FIFO order.
pub struct IoWorker {
    tx: Option<Sender<Job>>,
    join: Option<JoinHandle<()>>,
}

impl IoWorker {
    /// Spawn the worker thread.
    pub fn new(name: &str) -> Self {
        let (tx, rx) = channel::<Job>();
        let join = std::thread::Builder::new()
            .name(name.to_string())
            .spawn(move || {
                while let Ok(job) = rx.recv() {
                    job();
                }
            })
            .expect("spawn IoWorker thread");
        Self {
            tx: Some(tx),
            join: Some(join),
        }
    }

    /// Enqueue a fire-and-forget job. Returns `false` if the worker is shut down.
    pub fn submit(&self, job: impl FnOnce() + Send + 'static) -> bool {
        match &self.tx {
            Some(tx) => tx.send(Box::new(job)).is_ok(),
            None => false,
        }
    }

    /// Run `job` on the worker and return a receiver for its result. The caller
    /// blocks on `recv()` only when the result is actually needed, so the worker's
    /// I/O overlaps the caller's compute between submit and recv.
    pub fn run<T: Send + 'static>(&self, job: impl FnOnce() -> T + Send + 'static) -> Option<Receiver<T>> {
        let (tx, rx) = channel::<T>();
        let ok = self.submit(move || {
            let _ = tx.send(job());
        });
        ok.then_some(rx)
    }
}

impl Drop for IoWorker {
    fn drop(&mut self) {
        // Close the channel (ends the worker loop) and join the thread.
        self.tx.take();
        if let Some(h) = self.join.take() {
            let _ = h.join();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Arc;
    use std::sync::atomic::{AtomicUsize, Ordering};

    #[test]
    fn worker_runs_jobs_in_order() {
        let w = IoWorker::new("hayai-io-test");
        let counter = Arc::new(AtomicUsize::new(0));
        for i in 0..4 {
            let c = counter.clone();
            assert!(w.submit(move || {
                c.fetch_add(i + 1, Ordering::SeqCst);
            }));
        }
        // Wait until the worker drained the queue.
        let rx = w.run(|| 0u8).unwrap();
        let _ = rx.recv();
        assert_eq!(counter.load(Ordering::SeqCst), 10);
    }

    #[test]
    fn worker_run_returns_result() {
        let w = IoWorker::new("hayai-io-test-2");
        let rx = w.run(|| 6 * 7).unwrap();
        assert_eq!(rx.recv().unwrap(), 42);
    }
}

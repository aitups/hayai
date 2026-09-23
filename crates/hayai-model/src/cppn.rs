//! Global sparse FFN ("Vía B"): a single CPPN genome + threshold decodes, for
//! every layer and block, which `(input, output)` connections are alive.
//!
//! Mirrors `saor_domain::cppn` and the `cppn_decode.cl` kernel: genome layout
//! `w0[16*9] | b0[16] | w1[16*16] | b1[16] | w2[2*16] | b2[2]`, where the 9th
//! input coordinate is `y_layer`. Active connections are pruned to a [`CsrSparse`]
//! so the streaming runtime can execute them through the existing CSR FFN path.

use crate::sparse_dag::try_sparse_dag_to_csr;
use crate::weights::CsrSparse;
use rayon::prelude::*;

/// Hidden width of the CPPN MLP.
pub const CPPN_HIDDEN: usize = 16;
/// Input dimensionality of the CPPN (includes `y_layer`).
pub const CPPN_INPUT_DIM: usize = 9;

/// Evaluate the CPPN at input `v`, returning `(w, l)` (weight gate, sigmoid
/// activation). Mirrors `saor_domain::cppn::eval`.
pub fn cppn_eval(genome: &[f32], v: &[f32; CPPN_INPUT_DIM]) -> (f32, f32) {
    let h = CPPN_HIDDEN;
    let off_b0 = h * CPPN_INPUT_DIM;
    let off_w1 = off_b0 + h;
    let off_b1 = off_w1 + h * h;
    let off_w2 = off_b1 + h;
    let off_b2 = off_w2 + 2 * h;
    let mut h0 = [0.0f32; CPPN_HIDDEN];
    for o in 0..h {
        let mut acc = genome[off_b0 + o];
        for k in 0..CPPN_INPUT_DIM {
            acc += genome[o * CPPN_INPUT_DIM + k] * v[k];
        }
        h0[o] = acc.tanh();
    }
    let mut acc_w = genome[off_b2];
    let mut acc_l = genome[off_b2 + 1];
    for o in 0..h {
        let mut h1 = genome[off_b1 + o];
        for k in 0..h {
            h1 += genome[off_w1 + o * h + k] * h0[k];
        }
        h1 = h1.sin();
        acc_w += genome[off_w2 + o] * h1;
        acc_l += genome[off_w2 + h + o] * h1;
    }
    (acc_w, 1.0 / (1.0 + (-acc_l).exp()))
}

/// Minimum genome length required by [`cppn_eval`].
pub fn genome_len() -> usize {
    CPPN_HIDDEN * CPPN_INPUT_DIM + CPPN_HIDDEN + CPPN_HIDDEN * CPPN_HIDDEN + CPPN_HIDDEN + 2 * CPPN_HIDDEN + 2
}

/// Depth coordinate of layer `layer` of `n_layers` (band centre, in `[-1, 1]`).
pub fn layer_coord(layer: usize, n_layers: usize) -> f32 {
    if n_layers <= 1 {
        0.0
    } else {
        -1.0 + 2.0 * (layer as f32 + 0.5) / n_layers as f32
    }
}

/// Active mask (`conn = i*d_out + j`, LSB-first) of a layer, parallelized over
/// input rows. `active[conn]` is true when `l_ij > tau`.
pub fn layer_active_mask(
    genome: &[f32],
    d_in: usize,
    d_out: usize,
    tau: f32,
    y_layer: f32,
) -> Vec<bool> {
    if genome.len() < genome_len() {
        return vec![false; d_in * d_out];
    }
    (0..d_in)
        .into_par_iter()
        .flat_map_iter(|i| {
            let y_i = if d_in > 1 {
                -1.0 + 2.0 * i as f32 / (d_in - 1) as f32
            } else {
                0.0
            };
            (0..d_out).map(move |j| {
                let y_j = if d_out > 1 {
                    -1.0 + 2.0 * j as f32 / (d_out - 1) as f32
                } else {
                    0.0
                };
                let v: [f32; CPPN_INPUT_DIM] = [
                    -1.0,
                    y_i,
                    1.0,
                    y_j,
                    2.0,
                    y_j - y_i,
                    (std::f32::consts::PI * y_i).sin(),
                    (std::f32::consts::PI * y_j).cos(),
                    y_layer,
                ];
                let (_, l) = cppn_eval(genome, &v);
                l > tau
            })
        })
        .collect()
}

/// Convert an active mask + dense row-major (`j*d_in + i`) weights to CSR, using
/// the teacher weights only where the connection is alive.
pub fn mask_to_csr(
    active: &[bool],
    dense_j_major: &[f32],
    d_in: usize,
    d_out: usize,
) -> CsrSparse {
    let total = d_in * d_out;
    let mut bits = vec![0u8; total.div_ceil(8)];
    let mut weights = Vec::new();
    for i in 0..d_in {
        for j in 0..d_out {
            let conn = i * d_out + j;
            if conn < active.len() && active[conn] {
                bits[conn / 8] |= 1 << (conn % 8);
                weights.push(dense_j_major.get(j * d_in + i).copied().unwrap_or(0.0));
            }
        }
    }
    let (row_ptr, col_idx, vals) =
        try_sparse_dag_to_csr(&bits, &weights, d_in, d_out).unwrap_or_else(|_| {
            // Shapes are consistent by construction; fall back to empty CSR.
            (vec![0; d_out + 1], Vec::new(), Vec::new())
        });
    CsrSparse {
        row_ptr,
        col_idx,
        vals,
        d_in,
        d_out,
    }
}

/// Decode one layer's sparse topology from the genome and prune `dense_j_major`.
pub fn sparse_layer_csr(
    genome: &[f32],
    tau: f32,
    y_layer: f32,
    dense_j_major: &[f32],
    d_in: usize,
    d_out: usize,
) -> CsrSparse {
    let active = layer_active_mask(genome, d_in, d_out, tau, y_layer);
    mask_to_csr(&active, dense_j_major, d_in, d_out)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::sparse_dag::spmm_dense_masked;

    fn genome_zeros() -> Vec<f32> {
        vec![0.0f32; genome_len()]
    }

    #[test]
    fn tau_low_keeps_all_tau_high_keeps_none() {
        let g = genome_zeros();
        let dense: Vec<f32> = (0..(4 * 8)).map(|i| i as f32 * 0.1).collect();
        let all = sparse_layer_csr(&g, -1.0, 0.0, &dense, 4, 8);
        assert_eq!(all.vals.len(), 4 * 8, "tau=-1 must keep every connection");
        let none = sparse_layer_csr(&g, 2.0, 0.0, &dense, 4, 8);
        assert_eq!(none.vals.len(), 0, "tau=2 must drop every connection");
    }

    #[test]
    fn sparse_layer_matches_dense_masked() {
        // A non-trivial genome: pseudorandom weights.
        let mut g = genome_zeros();
        for (i, x) in g.iter_mut().enumerate() {
            *x = ((i as f32 * 12.9898).sin() * 43758.547).fract() * 2.0 - 1.0;
        }
        let d_in = 8;
        let d_out = 4;
        let dense: Vec<f32> = (0..(d_in * d_out)).map(|i| (i as f32) * 0.25 - 2.0).collect();
        let active = layer_active_mask(&g, d_in, d_out, 0.5, 0.25);
        let adj: Vec<u8> = {
            let mut bits = vec![0u8; (d_in * d_out).div_ceil(8)];
            for (conn, &a) in active.iter().enumerate() {
                if a {
                    bits[conn / 8] |= 1 << (conn % 8);
                }
            }
            bits
        };
        // Weights in i-major order (only active) for the reference dense-masked call.
        let mut weights = Vec::new();
        for i in 0..d_in {
            for j in 0..d_out {
                if active[i * d_out + j] {
                    weights.push(dense[j * d_in + i]);
                }
            }
        }
        let x: Vec<f32> = (0..d_in).map(|i| i as f32 * 0.1).collect();
        let expected = spmm_dense_masked(&x, &adj, &weights, d_in, d_out);
        let csr = sparse_layer_csr(&g, 0.5, 0.25, &dense, d_in, d_out);
        let got = crate::spmm_csr_cpu(&x, &csr.row_ptr, &csr.col_idx, &csr.vals, d_in, d_out);
        assert_eq!(expected.len(), got.len());
        for (a, b) in expected.iter().zip(got.iter()) {
            assert!((a - b).abs() < 1e-4, "cppn CSR vs dense mismatch: {a} != {b}");
        }
    }

    #[test]
    fn short_genome_is_fail_closed() {
        let csr = sparse_layer_csr(&[0.0; 4], 0.5, 0.0, &[1.0; 12], 4, 3);
        assert_eq!(csr.vals.len(), 0);
    }
}

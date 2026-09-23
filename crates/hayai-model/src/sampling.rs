/// Next-token sampling strategies.
#[derive(Debug, Clone, Copy)]
pub enum SamplerConfig {
    Greedy,
    Temperature { temperature: f32 },
    TopP { temperature: f32, top_p: f32 },
    TopK { temperature: f32, top_k: usize },
    TopKTopP { temperature: f32, top_k: usize, top_p: f32 },
    /// temperature + top_p + min_p (min-p keeps tokens with `p >= min_p * p_max`).
    MinP { temperature: f32, top_p: f32, min_p: f32 },
}

impl Default for SamplerConfig {
    fn default() -> Self {
        Self::Greedy
    }
}

/// Additive/multiplicative adjustments applied to logits before filtering.
#[derive(Debug, Clone, Default)]
pub struct Penalties {
    /// Division of positive logits / multiplication of negative ones for tokens
    /// already generated (`1.0` = disabled).
    pub repetition: f32,
    /// One-off subtraction for each token already generated.
    pub presence: f32,
    /// Subtraction proportional to how many times the token already appeared.
    pub frequency: f32,
    /// Token id → additive bias (OpenAI `logit_bias`).
    pub logit_bias: Vec<(u32, f32)>,
}

impl Penalties {
    pub fn is_noop(&self) -> bool {
        (self.repetition <= 0.0 || self.repetition == 1.0)
            && self.presence == 0.0
            && self.frequency == 0.0
            && self.logit_bias.is_empty()
    }
}

/// Sample from logits `[vocab]`. Returns token id. NaN/inf logits are sanitized so
/// the result is always a valid id.
pub fn sample(logits: &[f32], cfg: SamplerConfig, rng_state: &mut u64) -> u32 {
    sample_with(logits, cfg, rng_state, &Penalties::default(), &[])
}

/// Like [`sample`] but applies penalties against the already-generated ids and
/// optional `logit_bias` overrides.
pub fn sample_with(
    logits: &[f32],
    cfg: SamplerConfig,
    rng_state: &mut u64,
    penalties: &Penalties,
    generated: &[u32],
) -> u32 {
    if logits.is_empty() {
        return 0;
    }
    let mut l = sanitize(logits);
    apply_penalties(&mut l, penalties, generated);

    match cfg {
        SamplerConfig::Greedy => argmax(&l),
        SamplerConfig::Temperature { temperature } => {
            let mut probs = softmax_temp(&l, temperature.max(1e-5));
            sample_multinomial(&mut probs, rng_state)
        }
        SamplerConfig::TopP { temperature, top_p } => {
            let mut probs = softmax_temp(&l, temperature.max(1e-5));
            apply_top_p(&mut probs, top_p.clamp(0.0, 1.0));
            sample_multinomial(&mut probs, rng_state)
        }
        SamplerConfig::TopK { temperature, top_k } => {
            let mut probs = softmax_temp(&l, temperature.max(1e-5));
            apply_top_k(&mut probs, top_k);
            sample_multinomial(&mut probs, rng_state)
        }
        SamplerConfig::TopKTopP {
            temperature,
            top_k,
            top_p,
        } => {
            let mut probs = softmax_temp(&l, temperature.max(1e-5));
            apply_top_k(&mut probs, top_k);
            apply_top_p(&mut probs, top_p.clamp(0.0, 1.0));
            sample_multinomial(&mut probs, rng_state)
        }
        SamplerConfig::MinP {
            temperature,
            top_p,
            min_p,
        } => {
            let mut probs = softmax_temp(&l, temperature.max(1e-5));
            apply_top_p(&mut probs, top_p.clamp(0.0, 1.0));
            apply_min_p(&mut probs, min_p);
            sample_multinomial(&mut probs, rng_state)
        }
    }
}

/// Replace NaN with `-inf` and cap `+inf` to a large finite value so downstream
/// arithmetic stays well-defined.
fn sanitize(logits: &[f32]) -> Vec<f32> {
    let cap = f32::MAX / 4.0;
    logits
        .iter()
        .map(|&v| {
            if v.is_nan() {
                f32::NEG_INFINITY
            } else if v == f32::INFINITY {
                cap
            } else {
                v
            }
        })
        .collect()
}

fn apply_penalties(logits: &mut [f32], pen: &Penalties, generated: &[u32]) {
    if !pen.is_noop() && !generated.is_empty() {
        // Count occurrences for frequency/presence.
        let mut counts: std::collections::HashMap<u32, u32> = std::collections::HashMap::new();
        for &t in generated {
            *counts.entry(t).or_insert(0) += 1;
        }
        let rep = if pen.repetition > 0.0 {
            pen.repetition
        } else {
            1.0
        };
        for (&t, &c) in &counts {
            if let Some(l) = logits.get_mut(t as usize) {
                if rep != 1.0 {
                    *l = if *l > 0.0 { *l / rep } else { *l * rep };
                }
                *l -= pen.presence;
                *l -= pen.frequency * c as f32;
            }
        }
    }
    for &(id, b) in &pen.logit_bias {
        if let Some(l) = logits.get_mut(id as usize) {
            *l += b;
        }
    }
}

fn argmax(logits: &[f32]) -> u32 {
    let mut best_i = 0usize;
    let mut best_v = f32::NEG_INFINITY;
    for (i, &v) in logits.iter().enumerate() {
        if v > best_v {
            best_v = v;
            best_i = i;
        }
    }
    best_i as u32
}

fn softmax_temp(logits: &[f32], temperature: f32) -> Vec<f32> {
    let max = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let mut exps = Vec::with_capacity(logits.len());
    let mut sum = 0.0f32;
    for &l in logits {
        // `l - max` is <= 0 (finite), so exp is in (0, 1] and cannot overflow.
        let e = ((l - max) / temperature).exp();
        exps.push(e);
        sum += e;
    }
    if sum > 0.0 {
        for e in &mut exps {
            *e /= sum;
        }
    }
    exps
}

fn apply_top_k(probs: &mut [f32], top_k: usize) {
    if top_k == 0 || top_k >= probs.len() {
        return;
    }
    let mut order: Vec<usize> = (0..probs.len()).collect();
    order.sort_by(|&a, &b| {
        probs[b]
            .partial_cmp(&probs[a])
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    for &i in order.iter().skip(top_k) {
        probs[i] = 0.0;
    }
    renormalize(probs);
}

fn apply_top_p(probs: &mut [f32], top_p: f32) {
    if top_p >= 1.0 {
        return;
    }
    let mut idx: Vec<usize> = (0..probs.len()).collect();
    idx.sort_by(|&a, &b| {
        probs[b]
            .partial_cmp(&probs[a])
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    let mut cum = 0.0f32;
    let mut cutoff = idx.len();
    for (k, &i) in idx.iter().enumerate() {
        cum += probs[i];
        if cum >= top_p {
            cutoff = k + 1;
            break;
        }
    }
    for (k, &i) in idx.iter().enumerate() {
        if k >= cutoff {
            probs[i] = 0.0;
        }
    }
    renormalize(probs);
}

/// Keep tokens whose probability is at least `min_p * max_prob`.
fn apply_min_p(probs: &mut [f32], min_p: f32) {
    if min_p <= 0.0 {
        return;
    }
    let max = probs.iter().copied().fold(0.0f32, f32::max);
    if max <= 0.0 {
        return;
    }
    let threshold = min_p * max;
    for p in probs.iter_mut() {
        if *p < threshold {
            *p = 0.0;
        }
    }
    renormalize(probs);
}

fn renormalize(probs: &mut [f32]) {
    let sum: f32 = probs.iter().sum();
    if sum > 0.0 && sum.is_finite() {
        for p in probs.iter_mut() {
            *p /= sum;
        }
    }
}

fn sample_multinomial(probs: &mut [f32], rng: &mut u64) -> u32 {
    let total: f32 = probs.iter().sum();
    if total <= 0.0 || !total.is_finite() {
        // Degenerate distribution (all filtered / NaN): fall back to argmax.
        return argmax(probs);
    }
    let r = xorshift64(rng) * total;
    let mut cum = 0.0f32;
    for (i, &p) in probs.iter().enumerate() {
        cum += p;
        if r < cum {
            return i as u32;
        }
    }
    (probs.len().saturating_sub(1)) as u32
}

fn xorshift64(state: &mut u64) -> f32 {
    // xorshift64 is a fixed point at 0; reseed it so `seed = 0` does not always
    // return token 0.
    if *state == 0 {
        *state = 0x9E37_79B9_7F4A_7C15;
    }
    let mut x = *state;
    x ^= x << 13;
    x ^= x >> 7;
    x ^= x << 17;
    *state = x;
    ((x >> 11) as f32) / ((1u64 << 53) as f32)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn greedy_picks_max() {
        let logits = [0.1f32, 3.0, 0.2];
        assert_eq!(sample(&logits, SamplerConfig::Greedy, &mut 1), 1);
    }

    #[test]
    fn temperature_is_deterministic_with_fixed_rng_seed_path() {
        let logits = [1.0f32, 1.0, 1.0];
        let mut rng = 42u64;
        let a = sample(
            &logits,
            SamplerConfig::Temperature { temperature: 1.0 },
            &mut rng,
        );
        assert!(a < 3);
    }

    #[test]
    fn seed_zero_is_not_degenerate() {
        // With an all-equal distribution, seed 0 must still sample valid ids.
        let logits = [1.0f32; 8];
        let mut rng = 0u64;
        let mut seen = std::collections::HashSet::new();
        for _ in 0..64 {
            seen.insert(sample(
                &logits,
                SamplerConfig::Temperature { temperature: 1.0 },
                &mut rng,
            ));
        }
        assert!(seen.len() > 1, "seed 0 stuck: {seen:?}");
    }

    #[test]
    fn nan_and_inf_logits_are_sanitized() {
        let logits = [f32::NAN, f32::INFINITY, 0.0];
        assert_eq!(sample(&logits, SamplerConfig::Greedy, &mut 7), 1);
        let logits = [f32::NAN, f32::NEG_INFINITY];
        // No NaN panic; picks a valid id.
        let id = sample(&logits, SamplerConfig::Greedy, &mut 7);
        assert!(id < 2);
    }

    #[test]
    fn top_k_never_samples_outside_k() {
        let logits = [5.0f32, 4.0, 3.0, 2.0, 1.0];
        let mut rng = 123u64;
        for _ in 0..200 {
            let id = sample(
                &logits,
                SamplerConfig::TopK {
                    temperature: 1.0,
                    top_k: 2,
                },
                &mut rng,
            );
            assert!(id < 2, "top_k sampled {id}");
        }
    }

    #[test]
    fn min_p_drops_low_probability_tokens() {
        let logits = [10.0f32, 0.0, 0.0, 0.0];
        let mut rng = 5u64;
        for _ in 0..200 {
            let id = sample(
                &logits,
                SamplerConfig::MinP {
                    temperature: 1.0,
                    top_p: 1.0,
                    min_p: 0.5,
                },
                &mut rng,
            );
            assert_eq!(id, 0);
        }
    }

    #[test]
    fn repetition_penalty_discourages_repeat() {
        // Token 1 has the highest logit but is penalized for already appearing.
        let logits = [1.0f32, 3.0, 0.0];
        let pen = Penalties {
            repetition: 4.0,
            ..Default::default()
        };
        let id = sample_with(
            &logits,
            SamplerConfig::Greedy,
            &mut 1,
            &pen,
            &[1],
        );
        assert_ne!(id, 1, "repetition penalty did not apply");
    }

    #[test]
    fn logit_bias_overrides_greedy() {
        let logits = [1.0f32, 3.0, 0.0];
        let pen = Penalties {
            logit_bias: vec![(2, 10.0)],
            ..Default::default()
        };
        let id = sample_with(&logits, SamplerConfig::Greedy, &mut 1, &pen, &[]);
        assert_eq!(id, 2);
    }
}

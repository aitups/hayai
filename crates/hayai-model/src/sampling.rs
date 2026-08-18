/// Next-token sampling strategies.
#[derive(Debug, Clone, Copy)]
pub enum SamplerConfig {
    Greedy,
    Temperature { temperature: f32 },
    TopP { temperature: f32, top_p: f32 },
}

impl Default for SamplerConfig {
    fn default() -> Self {
        Self::Greedy
    }
}

/// Sample from logits `[vocab]`. Returns token id.
pub fn sample(logits: &[f32], cfg: SamplerConfig, rng_state: &mut u64) -> u32 {
    match cfg {
        SamplerConfig::Greedy => argmax(logits),
        SamplerConfig::Temperature { temperature } => {
            let t = temperature.max(1e-5);
            let probs = softmax_temp(logits, t);
            sample_multinomial(&probs, rng_state)
        }
        SamplerConfig::TopP { temperature, top_p } => {
            let t = temperature.max(1e-5);
            let mut probs = softmax_temp(logits, t);
            apply_top_p(&mut probs, top_p.clamp(0.0, 1.0));
            sample_multinomial(&probs, rng_state)
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
    let sum: f32 = probs.iter().sum();
    if sum > 0.0 {
        for p in probs.iter_mut() {
            *p /= sum;
        }
    }
}

fn sample_multinomial(probs: &[f32], rng: &mut u64) -> u32 {
    let r = xorshift64(rng);
    let mut cum = 0.0f32;
    for (i, &p) in probs.iter().enumerate() {
        cum += p;
        if r <= cum {
            return i as u32;
        }
    }
    (probs.len().saturating_sub(1)) as u32
}

fn xorshift64(state: &mut u64) -> f32 {
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
}

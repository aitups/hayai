fn main() {
    use hayai_core::deltanet::DeltaNetLayerWeights;
    use hayai_cpu::rms_norm;
    let mut cat = hayai_model::GgufCatalog::open("models/Qwen_Qwen3.5-4B-Q4_K_M.gguf").unwrap();
    let w = DeltaNetLayerWeights::load(&mut cat, 0).unwrap();
    let h = 2560usize;
    let mut x = vec![0.0f32; h];
    cat.read_embed_row("token_embd.weight", 760, h, &mut x).unwrap();
    let attn_norm = cat.dequant_f32("blk.0.attn_norm.weight").unwrap();
    rms_norm(&mut x, &attn_norm, 1e-6);

    let mut mixed = vec![0.0f32; w.qkv.nrows];
    w.qkv.gemv(&x, &mut mixed).unwrap();
    let k = w.conv_k; let cd = w.conv_dim;
    let mut conv_out = vec![0.0f32; cd];
    for c in 0..cd {
        let acc = w.conv1d[c * k + (k - 1)] * mixed[c];
        conv_out[c] = acc / (1.0 + (-acc).exp());
    }
    let key_dim = w.n_k_heads * w.head_k;
    let value_dim = w.n_v_heads * w.head_v;
    let mut q = conv_out[..key_dim].to_vec();
    let mut kk = conv_out[key_dim..key_dim*2].to_vec();
    let v = conv_out[key_dim*2..key_dim*2+value_dim].to_vec();
    // l2
    for h_i in 0..w.n_k_heads {
        let base = h_i * w.head_k;
        let mut ms = 0.0f32;
        for i in 0..w.head_k { ms += q[base+i]*q[base+i]; }
        let inv = 1.0/(ms + w.eps).sqrt();
        for i in 0..w.head_k { q[base+i] *= inv; }
        let mut ms = 0.0f32;
        for i in 0..w.head_k { ms += kk[base+i]*kk[base+i]; }
        let inv = 1.0/(ms + w.eps).sqrt();
        for i in 0..w.head_k { kk[base+i] *= inv; }
    }
    let q_scale = 1.0/(w.head_k as f32).sqrt();
    for e in q.iter_mut() { *e *= q_scale; }
    let mut alpha_h = vec![0.0f32; w.n_v_heads];
    let mut beta_h = vec![0.0f32; w.n_v_heads];
    w.alpha.gemv(&x, &mut alpha_h).unwrap();
    w.beta.gemv(&x, &mut beta_h).unwrap();
    println!("alpha first4={:?}", &alpha_h[..4]);
    println!("beta first4={:?}", &beta_h[..4]);
    println!("q0 first4={:?}", &q[..4]);
    println!("k0 first4={:?}", &kk[..4]);
    println!("v0 first4={:?}", &v[..4]);

    // one head recurrent from zero state
    let hv = w.head_v; let hk = w.head_k; let k_repeat = w.n_v_heads / w.n_k_heads;
    let mut o = vec![0.0f32; value_dim];
    let mut state = vec![0.0f32; w.n_v_heads * hk * hv];
    for vh in 0..w.n_v_heads {
        let kh = vh / k_repeat;
        let a = w.a[vh];
        let soft = { let t = w.dt_bias[vh] + alpha_h[vh]; if t > 20.0 { t } else { (1.0+t.exp()).ln() } };
        let decay = (a * soft).exp();
        let b = 1.0/(1.0+(-beta_h[vh]).exp());
        let s = &mut state[vh*hk*hv..(vh+1)*hk*hv];
        let qh = &q[kh*hk..kh*hk+hk];
        let khv = &kk[kh*hk..kh*hk+hk];
        let vv = &v[vh*hv..(vh+1)*hv];
        for e in s.iter_mut() { *e *= decay; }
        let mut kv_mem = vec![0.0f32; hv];
        for ki in 0..hk {
            let ks = khv[ki]; let row = ki*hv;
            for vi in 0..hv { kv_mem[vi] += s[row+vi]*ks; }
        }
        for ki in 0..hk {
            let ks = khv[ki]; let row = ki*hv;
            for vi in 0..hv {
                let delta = (vv[vi]-kv_mem[vi])*b;
                s[row+vi] += ks*delta;
            }
        }
        for vi in 0..hv {
            let mut acc = 0.0;
            for ki in 0..hk { acc += s[ki*hv+vi]*qh[ki]; }
            o[vh*hv+vi] = acc;
        }
        if vh == 0 {
            println!("vh0 decay={decay:.6} b={b:.6} soft={soft:.6} o0 first4={:?}", &o[..4]);
        }
    }
    let mut ms=0.0f32; for &v in &o { ms+=v*v; }
    println!("o_pre_norm_rms={:.6}", (ms/o.len() as f32).sqrt());

    let mut z = vec![0.0f32; value_dim];
    w.gate.gemv(&x, &mut z).unwrap();
    for vh in 0..w.n_v_heads {
        let base = vh*hv;
        let oh = &mut o[base..base+hv];
        let mut ms = 0.0f32;
        for &v in oh.iter() { ms += v*v; }
        let inv = 1.0/(ms/hv as f32 + w.eps).sqrt();
        for i in 0..hv {
            let n = w.norm[i];
            let silu = z[base+i]/(1.0+(-z[base+i]).exp());
            oh[i] = oh[i]*inv*n*silu;
        }
    }
    let mut ms=0.0f32; for &v in &o { ms+=v*v; }
    println!("o_post_gate_rms={:.6} first4={:?}", (ms/o.len() as f32).sqrt(), &o[..4]);
    let mut proj = vec![0.0f32; h];
    w.out.gemv(&o, &mut proj).unwrap();
    let mut ms=0.0f32; for &v in &proj { ms+=v*v; }
    println!("proj_rms={:.6} first8={:?}", (ms/h as f32).sqrt(), &proj[..8]);
}

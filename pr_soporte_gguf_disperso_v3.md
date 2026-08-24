# Pull Request: Ejecución de capas FFN dispersas (GGUF de `saor`) sin densificar — v3

**Target Repository:** `github.com/aitups/hayai`
**Branches:** `feature/ffn-irregular-dag` -> `main` (a crear)
**Crates afectados:** `hayai-core`, `hayai-kernels`, `hayai-opencl` (y lectura vía `hayai-model` / `hayai-io`, sin tipos nuevos)
**Productor del formato:** workspace `saor` (directorio hermano), crate `saor-streamer`

---

## 0. Objetivo (revisado)

Ejecutar, dentro del planificador inteligente actual de `hayai` (`build_exec_plan`
→ `StreamingUnit`), un GGUF cuyos bloques FFN vienen en **representación dispersa**,
**sin re-materializar una matriz densa con padding de ceros**.

No hay cambio automático a "modo hardware mínimo": la capa dispersa es un
`LayerOpKind` más, planificado y ejecutado como cualquier otro op.

> Nota sobre la versión v2: la PR anterior (`pr_soporte_gguf_disperso_v2.md`) estaba
> escrita contra una estructura de código inexistente (`TensorRole`, `compile_layer`,
> `engine.rs`, `RustStreamer` en `hayai-io`, etc.) y con un kernel que no coincidía
> con el formato real. Esta v3 documenta el formato **real** que produce `saor` y lo
> integra con las APIs **reales** de `hayai`.

---

## 1. Formato real del GGUF disperso (lo que produce `saor`)

`saor-streamer/src/gguf_sparse.rs` emite un GGUF v3 con:

**Metadatos (prefijo `saor.*`):**

| Clave | Tipo GGUF | Acceso en `hayai` |
|---|---|---|
| `saor.d_in` | UINT64 (10) | `cat.meta_u32/u64("saor.d_in")` |
| `saor.d_out` | UINT64 (10) | `cat.meta_u32/u64("saor.d_out")` |
| `saor.tau` | FLOAT32 (6) | `cat.meta_f32("saor.tau")` |
| `saor.genome` | ARRAY(FLOAT32) (9) | `cat.metadata["saor.genome"].as_f32_array()` |
| `saor.sparse` | BOOL (7) | `cat.metadata["saor.sparse"]` → `MetadataValue::Bool` |

**Tensores (exactamente 2, sin prefijo `blk.N.`):**

| Nombre | Tipo GGML | Contenido |
|---|---|---|
| `ffn_dag_adjacency` | I8 (1 byte/elemento) | bit-tensor LSB-first, `conn = i*d_out + j` |
| `ffn_dag_weights` | F32 | pesos activos en orden i-mayor (solo conexiones vivas) |

El parser GGUF de `hayai` (`parse_header_bytes` + `read_value`) ya soporta todos los
tipos de metadatos anteriores (UINT64=10, FLOAT32=6, ARRAY=9, BOOL=7). Los tensores
se indexan en `GgufCatalog.tensor_index` como cualquier otro.

**Cómputo:** el DAG no se consume como "bitmask + pesos densos", sino convirtiendo
`(adjacency, weights)` a **CSR** (`row_ptr`, `col_idx`, `vals`) y ejecutando un
SpMM CSR en OpenCL (ver `saor-domain/src/topology.rs::Topology::to_csr` y
`saor-opencl/kernels/spmm.cl`).

---

## 2. Reconciliaciones previas (bloqueantes)

### R1 — Bug de `ggml_type` en `saor` (obligatorio)

`saor-streamer/src/gguf_sparse.rs` declara:

```rust
pub const GGML_TYPE_I8: i32 = 16;
```

Según el estándar GGML, **16 = `IQ2_XXS`** e **`I8` = 24**. `hayai` ya lo mapea así
(`gguf_types.rs`: `IQ2_XXS = 16`, `I8 = 24`). Con el valor 16, `hayai` interpretaría
la adyacencia como `IQ2_XXS` y `tensor_nbytes` daría un tamaño incorrecto
(rompiendo `read_tensor_into` y el streaming).

**Acción:** corregir en `saor` a `pub const GGML_TYPE_I8: i32 = 24;`. Mientras tanto,
`hayai` puede mitigarlo leyendo la adyacencia por `dims[0]` (nº de bytes) en vez de
confiar en `tensor_nbytes` (ver §3.D).

### R2 — `general.architecture` ausente

El GGUF de `saor` no trae `general.architecture`; `build_exec_plan` lo tratará como
`"unknown"`, lo cual es aceptable (el motor es agnóstico). No requiere acción, solo
confirmación de que el plan no exige esa clave.

### R3 — Alcance: un único bloque FFN

Hoy `saor consolidate` emite **un solo bloque disperso** (no un modelo completo con
`blk.N.`). El soporte en `hayai` debe, por tanto, arrancar ejecutando un bloque
disperso aislado (SpMM), y dejar preparada la extensión a bloques por capa.

---

## 3. Cambios por crate

### A. `hayai-core/src/exec_plan.rs` — registrar el op disperso

#### A1. Variantes nuevas de `LayerOpKind`

Añadir al enum existente (que ya tiene ~40 variantes):

```rust
pub enum LayerOpKind {
    // ...variantes existentes (TokenEmbed, AttnQ, FfnGate, FfnUp, FfnDown,
    // Router, ExpertGate, ...)...
    /// Bit-tensor de adyacencia del FFN disperso (`ffn_dag_adjacency`).
    FfnDagAdjacency,
    /// Pesos activos del FFN disperso (`ffn_dag_weights`).
    FfnDagWeights,
}
```

#### A2. Clasificación en `classify_tensor_impl`

Insertar una fase **antes** del FFN denso (Phase 6). Los nombres `ffn_dag_*` no
colisionan con `ffn_gate/up/down/norm`, pero conviene anclarlos explícitamente:

```rust
    // ── Phase 5c: FFN disperso (DAG irregular, GGUF de saor) — antes del FFN denso. ─
    if n.contains("ffn_dag_adjacency") {
        return Ok(FfnDagAdjacency);
    }
    if n.contains("ffn_dag_weights") {
        return Ok(FfnDagWeights);
    }
```

#### A3. Binding en `op_binding`

```rust
        FfnDagAdjacency | FfnDagWeights => OpBinding::GPU_ASYNC,
```

#### A4. Conversión host `to_csr`

Replicar `Topology::to_csr` de `saor` (bit-tensor LSB-first + pesos i-mayor → CSR
j-mayor, filas = salidas `j`):

```rust
/// Convierte (adjacency, weights) a CSR. `conn = i*d_out + j` es el índice de
/// conexión en el bit-tensor; `weights` está en orden i-mayor (solo conexiones vivas).
pub fn sparse_dag_to_csr(
    adjacency: &[u8],
    weights: &[f32],
    d_in: usize,
    d_out: usize,
) -> (Vec<i32>, Vec<i32>, Vec<f32>) {
    let total = d_in * d_out;
    // Peso por conexión `conn`, para iterar en orden j-mayor sin desordenar valores.
    let mut weight_by_conn = vec![0.0f32; total];
    let mut w_idx = 0usize;
    for i in 0..d_in {
        for j in 0..d_out {
            let conn = i * d_out + j;
            if conn < total && (adjacency[conn / 8] & (1 << (conn % 8))) != 0 {
                weight_by_conn[conn] = weights[w_idx];
                w_idx += 1;
            }
        }
    }
    let mut row_ptr = vec![0i32; d_out + 1];
    let mut col_idx = Vec::new();
    let mut vals = Vec::new();
    for j in 0..d_out {
        for i in 0..d_in {
            let conn = i * d_out + j;
            if conn < total && (adjacency[conn / 8] & (1 << (conn % 8))) != 0 {
                col_idx.push(i as i32);
                vals.push(weight_by_conn[conn]);
            }
        }
        row_ptr[j + 1] = col_idx.len() as i32;
    }
    (row_ptr, col_idx, vals)
}
```

#### A5. Agrupación en `build_exec_plan`

Los dos tensores se clasifican y entran en un `StreamingUnit` (hoy con
`block_id = None`, porque el GGUF de `saor` no tiene capas). `d_in`/`d_out` se leen
de los metadatos `saor.d_in`/`saor.d_out` para dimensionar el SpMM. La extensión a
bloques por capa (`blk.N.ffn_dag_*`) se hace añadiendo el prefijo al clasificador y
reutilizando `block_id`.

### B. `hayai-kernels` — kernel SpMM CSR

#### B1. `crates/hayai-kernels/kernels/spmm_csr.cl`

Espejo del kernel validado de `saor` (OpenCL C 1.2, un work-item por `(b, j)`):

```c
__kernel void spmm_csr(
    __global const float* x,     // B * d_in
    __global const int* row_ptr, // d_out + 1
    __global const int* col_idx, // nnz
    __global const float* vals,  // nnz
    const int d_in,
    const int d_out,
    __global float* y)           // B * d_out
{
    const int gid = get_global_id(0);
    const int b = gid / d_out;
    const int j = gid % d_out;
    float acc = 0.0f;
    for (int k = row_ptr[j]; k < row_ptr[j + 1]; k++) {
        acc += x[b * d_in + col_idx[k]] * vals[k];
    }
    y[b * d_out + j] = acc;
}
```

#### B2. `crates/hayai-kernels/src/lib.rs`

```rust
pub const SPMM_CSR_CL: &str = include_str!("../kernels/spmm_csr.cl");

pub fn opencl_program_source() -> String {
    let mut src = String::with_capacity(
        LUT_MATMUL_CL.len() + GGML_GEMV_Q4_CL.len() + SPMM_CSR_CL.len() + 8,
    );
    src.push_str(LUT_MATMUL_CL);
    src.push('\n');
    src.push_str(GGML_GEMV_Q4_CL);
    src.push('\n');
    src.push_str(SPMM_CSR_CL);
    src
}
```

Y añadir `("spmm_csr.cl", SPMM_CSR_CL)` al array del test
`kernel_sources_never_dynamically_index_vector_components` (garantiza el subconjunto
OpenCL C 1.2 obligatorio).

### C. `hayai-opencl` — registro y dispatch

#### C1. `src/context.rs`

Añadir el campo y crearlo junto a los demás kernels:

```rust
pub struct OpenClEngine {
    // ...campos existentes (lut_kernel, gemv_q4_0, ...)...
    pub spmm_csr: Kernel,
}
```

```rust
let spmm_csr = Kernel::create(&program, "spmm_csr")
    .map_err(|e| OpenClError::ClError(format!("spmm_csr kernel: {e}")))?;
```

#### C2. `src/compute.rs`

Dispatch con el patrón real `ExecuteKernel` (no `enqueue_nd_range_kernel` directo):

```rust
pub fn dispatch_spmm_csr(
    &self,
    x: &Buffer<f32>,
    row_ptr: &Buffer<i32>,
    col_idx: &Buffer<i32>,
    vals: &Buffer<f32>,
    d_in: usize,
    d_out: usize,
    y: &mut Buffer<f32>,
    batch: usize,
) -> Result<(), OpenClError> {
    let global = batch * d_out;
    unsafe {
        ExecuteKernel::new(&self.spmm_csr)
            .set_arg(x)
            .set_arg(row_ptr)
            .set_arg(col_idx)
            .set_arg(vals)
            .set_arg(&(d_in as cl_int))
            .set_arg(&(d_out as cl_int))
            .set_arg(y)
            .set_global_work_size(global)
            .enqueue_nd_range(&self.queue)
            .map_err(|e| OpenClError::ClError(format!("enqueue spmm_csr: {e}")))?;
    }
    Ok(())
}
```

> Nota: si `col_idx`/`vals` están vacíos (topología con τ muy alto), OpenCL no admite
> buffers de tamaño 0; devolver `y = 0` directamente en ese caso.

Los buffers se alojan con `memory.rs` (SVM o pinned), igual que el resto de ops.

### D. `hayai-model` / `hayai-io` — lectura (sin tipos nuevos)

No hace falta ningún tipo nuevo. Para leer los dos tensores:

```rust
let mut cat = GgufCatalog::open(&path)?;
let adj_info = cat.tensor("ffn_dag_adjacency")?.clone();
let w_info   = cat.tensor("ffn_dag_weights")?.clone();

// Mitigación de R1 (hasta que saor emita I8=24): la adyacencia mide `dims[0]` bytes.
let adj_len = adj_info.dims.first().copied().unwrap_or(0) as usize;
let mut adjacency = vec![0u8; adj_len];
cat.read_tensor_into("ffn_dag_adjacency", &mut adjacency)?;

let mut weights_bytes = vec![0u8; tensor_nbytes(&w_info)?];
cat.read_tensor_into("ffn_dag_weights", &mut weights_bytes)?;
// reinterpretar weights_bytes como &[f32] (chunks_exact(4) / bytemuck)
```

`read_tensor_into` ya usa `WeightIo::read_at` de forma determinista (io_uring en
Linux, File en Windows).

---

## 4. Ruta de ejecución (data flow)

1. `GgufCatalog::open` parsea cabecera + metadatos `saor.*` + índice de tensores.
2. `build_exec_plan` clasifica `ffn_dag_adjacency` / `ffn_dag_weights` en
   `FfnDagAdjacency` / `FfnDagWeights` y los agrupa en un `StreamingUnit`.
3. En el forward del bloque disperso:
   - Leer `ffn_dag_adjacency` + `ffn_dag_weights` (host).
   - `sparse_dag_to_csr(adjacency, weights, d_in, d_out)` → `(row_ptr, col_idx, vals)`.
   - Subir `x`, `row_ptr`, `col_idx`, `vals` y ejecutar `spmm_csr` en OpenCL.
   - Leer `y` (B × d_out).

---

## 5. Tests

En `hayai-core` y `hayai-model`:

1. **Clasificación** — `classify_tensor_impl("ffn_dag_adjacency", false) == FfnDagAdjacency`
   y `classify_tensor_impl("ffn_dag_weights", false) == FfnDagWeights`.
2. **Desempaquetado / `sparse_dag_to_csr`** — replicar el caso de `Topology::to_csr`
   (d_in=4, d_out=3, todo activo): `row_ptr[j+1]-row_ptr[j] == 4`, y `vals[k]` coincide
   con el peso de la conexión `i*3+j`.
3. **Equivalencia CSR vs densa** — `spmm_csr` (CPU de referencia o OpenCL si hay
   dispositivo) vs. `dense_row_major` enmascarada, tolerancia `<= 1e-5` en FP32.
4. **Bit-tensor LSB-first** — `0b10010110` → bits `[0,1,1,0,1,0,0,1]` (LSB-first).

Eliminados respecto a v2: `test_vram_flat_limit` (exige un modelo 30B inexistente) y
`test_cka_precision` (el filtro CKA vive en `saor`, no en `hayai`).

---

## 6. Criterios de aceptación (honestos)

- Un GGUF de `saor` (salida de `saor-engine consolidate`) **abre y planifica** en
  `hayai` sin error `UnknownLayerOp`.
- La salida del SpMM CSR coincide con la referencia densa enmascarada dentro de
  `1e-5` (FP32).
- No se materializa ninguna matriz densa `d_in × d_out`: el pico de memoria del
  bloque disperso es `O(nnz + B·(d_in+d_out))`, no `O(d_in·d_out)`.
- (Target pendiente de benchmark, no gate) el streaming por capa mantiene la
  ventana dispersa vía `ExecPlan::max_unit_bytes` / `StreamingMemoryBudget`.

---

## 7. Fuera de alcance / notas

- **MoE existente:** `hayai` ya tiene streaming disperso para MoE (`ExpertUnit`,
  `max_expert_bytes`, slicing de expertos fusionados). Si en el futuro los bloques
  dispersos se quieren encuadrar como "un experto por bloque", evaluar reutilizar
  esa maquinaria en lugar de un op paralelo.
- **Modelo completo 30B:** la captura de hooks del modelo real es la Fase 5 de
  `saor` y sigue pendiente del GGUF base; este PR cubre el runtime del bloque
  disperso ya consolidado.
- **Cuantización de pesos activos:** `saor` guarda `ffn_dag_weights` en F32; el
  empaquetado exacto (IQ4/Q4_K) se define cuando esta PR fije el esquema.






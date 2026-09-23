# Calibración de hardware y objetivos de utilización

Hayai planifica para **75–80 % de utilización de RAM y de cómputo**. El suelo
numérico de tok/s no es absoluto: emerge del hardware medido y del tamaño del
modelo que se streamea por token. Este fichero define el protocolo y guarda los
perfiles medidos por equipo.

## Protocolo

```bash
# Mide RAM host, disco (leyendo el GGUF) y BW efectiva de GEMV por dispositivo,
# e imprime los objetivos @75 % y @80 %.
cargo run --release -p hayai-cli -- calibrate \
  --model models/<modelo>.gguf \
  --out bench_results/hw_profile.json
```

- **RAM host**: copia secuencial de 32 MiB ×4 (lectura+escritura).
- **Disco**: lectura secuencial por el mismo `hayai_io` que usa inferencia,
  muestreando hasta 64 MiB del fichero.
- **Dispositivo**: GEMV F32 sobre una matriz de 16 MiB (incluye subida
  host→device), que es la forma real del streaming de FFN.
- `bytes/token` = tamaño del GGUF (cota superior de streaming no residente).
  El planner de Fase 1 lo refinará por familia/modo (residente → 0 I/O).

Objetivo: `tok/s @U = effective_stream_BW × U / bytes_per_token`, con
`U ∈ [0.75, 0.80]`. `effective_stream_BW = min(disco, mejor acelerador)`.

## Perfil — equipo de desarrollo (Windows, 2026)

Host: CPU + Intel UHD iGPU + NVIDIA RTX 4050 Laptop. Modelo de medida:
SmolLM2-135M-Instruct-Q4_K_M (100.6 MiB).

| Métrica | Valor |
| --- | --- |
| RAM host | ~22.4 GB/s |
| Disco (caché) | ~5.7 GB/s |
| RTX 4050 (GEMV efectivo) | ~6.0 GB/s |
| Intel UHD (GEMV efectivo) | ~3.4 GB/s |
| `effective_stream_BW` | ~5.7 GB/s |
| Objetivo @75 % | ~40.7 tok/s |
| Objetivo @80 % | ~43.4 tok/s |

> Nota: el decode actual en `minimal` está muy por debajo (≈1.6 tok/s) y en
> residente ≈5–6 tok/s; la brecha con el objetivo es el trabajo de Fase 1
> (planner de reparto, prefetch de expertos, prefill batcheado, solape I/O).

## Añadir un equipo

1. Ejecutar el comando `calibrate` en el equipo (BC250, GB10, …) con un modelo
   representativo.
2. Añadir una sección con la tabla de métricas y el modelo usado.
3. Guardar el JSON en `bench_results/hw_profile.json` (gitignored) y resumir
   aquí los valores para que el planner y el agente de despliegue los conozcan.

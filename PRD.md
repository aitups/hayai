# PRD: Motor de Inferencia LLM por Streaming (OpenCL/Rust)

## 1. Visión General

**Objetivo:** Construir un motor de inferencia de Modelos de Lenguaje Grande (LLMs) diseñado específicamente para entornos restringidos en memoria (Edge, APUs, hardware de consumo).
**Diferenciador:** En lugar de depender del mapeo de memoria reactivo del sistema operativo (`mmap`), el motor utilizará **streaming determinista de pesos comprimidos** desde disco, ocultando la latencia de I/O mediante ejecución asíncrona, orquestación heterogénea (CPU+GPU) y cálculo directo sobre datos comprimidos usando OpenCL 3.0.

---

## 2. Pila Tecnológica (Tech Stack)

* **Lenguaje Core:** Rust (seguridad de memoria, concurrencia asíncrona, control de bajo nivel).
* **Backend Computacional:** OpenCL 3.0 (vía el crate `opencl3`).
* **I/O Asíncrono:** `tokio-uring` / `io_uring` (Linux) para prefetching agresivo.
* **Aceleración CPU:** `std::simd` (Portable SIMD) para vectorización en operaciones dependientes de CPU.

---

## 3. Arquitectura del Sistema y Requisitos Clave

### 3.1. Pipeline de Memoria e I/O (Prefetching)

* **Requisito:** La latencia de disco no debe bloquear los ciclos de cómputo.
* **Implementación:**
* Diseño de **Double Buffering (Ping-Pong)** para activaciones y pesos.
* Uso de **Pinned Memory** (`CL_MEM_ALLOC_HOST_PTR`) para transferencias DMA rápidas en GPUs dedicadas.
* Uso de **SVM (Shared Virtual Memory)** en modo *Zero-Copy* para APUs/Sistemas unificados.
* Mientras la Capa $N$ se ejecuta, `io_uring` carga la Capa $N+1$ directamente en el buffer de prefetch.



### 3.2. Estrategia de Cómputo: Compresión y Ejecución

* **Requisito:** Minimizar el ancho de banda necesario en el bus (PCIe/RAM) y operar sin descomprimir la matriz entera.
* **Implementación:**
* Cuantización soportada: Formatos empaquetados (ej. 4-bit) con diccionarios de valores (LUT).
* **Kernels OpenCL Custom:** Diseño de kernels (en OpenCL C) que implementen multiplicaciones matriciales mediante diccionarios (LUT MatMul / Shift & Add) usando la memoria local (`__local`) del Work-Group para cargar la LUT.



### 3.3. Orquestación Heterogénea (Macro-Pipelining)

* **Requisito:** Mantener la CPU y la GPU al 100% de utilización de forma simultánea.
* **Implementación:**
* **GPU (OpenCL):** Se encarga exclusivamente de las redes Feed-Forward (FFN), procesando matrices gigantes que llegan en streaming.
* **CPU (Rust/SIMD):** Se encarga de calcular la Atención, procesar el RoPE (Rotary Position Embeddings) y gestionar el KV Cache.
* **Sincronización:** Eventos de OpenCL (`cl_event`) combinados con el runtime asíncrono de Rust para solapar la FFN de la capa actual con la Atención de la capa siguiente.



### 3.4. Gestión de Estado Acotado (KV Cache)

* **Requisito:** El consumo de RAM del estado debe ser predecible y tener un límite estricto, sin importar la longitud de la generación.
* **Implementación:**
* Estrategia mixta: **Attention Sinks** (retener los primeros 4-5 tokens permanentemente) + **Ventana Deslizante** (Sliding Window) para el contexto reciente.
* El KV Cache se almacenará cuantizado en INT8 (ej. tipo KIVI), manteniendo solo un pequeño buffer temporal en FP16/FP32 para el token actual.



---

## 4. Fases de Implementación (Roadmap)

| Fase | Hito Principal | Entregable Técnico |
| --- | --- | --- |
| **Fase 1: Cimientos (I/O & OpenCL)** | Pipeline de Streaming "Dummy" | Un bucle en Rust que lee archivos ficticios desde disco con `io_uring` y los sube a la GPU vía OpenCL 3.0 SVM/Pinned Memory mediante buffers Ping-Pong, midiendo ancho de banda. |
| **Fase 2: Kernels de Cómputo** | Inferencia de un MLP simple | Desarrollo del Kernel de OpenCL para LUT MatMul. Validación de que el resultado de la multiplicación empaquetada coincide con un cálculo en FP32 estándar. |
| **Fase 3: Orquestación y Estado** | Ejecución Heterogénea | Implementación de Atención en CPU (SIMD) + gestión del buffer circular (KV Cache INT8). Sincronización asíncrona CPU -> GPU. |
| **Fase 4: Integración de Modelo** | Carga de un LLM real | Parseo de un formato de modelo existente (ej. GGUF) extrayendo los tensores y pasándolos por tu pipeline de streaming hasta generar texto. |

---
use thiserror::Error;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[allow(non_camel_case_types)]
#[repr(u32)]
pub enum GgmlType {
    F32 = 0,
    F16 = 1,
    Q4_0 = 2,
    Q4_1 = 3,
    Q2_K = 10,
    Q3_K = 11,
    Q4_K = 12,
    Q5_K = 13,
    Q6_K = 14,
    Q5_0 = 6,
    Q5_1 = 7,
    Q8_0 = 8,
    Q8_1 = 9,
    /// IQ / TQ family (GGML type ids — GEMV wired incrementally).
    IQ2_XXS = 16,
    IQ2_XS = 17,
    IQ3_XXS = 18,
    IQ1_S = 19,
    IQ4_NL = 20,
    IQ3_S = 21,
    IQ2_S = 22,
    IQ4_XS = 23,
    IQ1_M = 29,
    TQ1_0 = 34,
    TQ2_0 = 35,
    Q8_K = 15,
    I8 = 24,
    I16 = 25,
    I32 = 26,
    I64 = 27,
    F64 = 28,
    BF16 = 30,
    Other(u32),
}

impl GgmlType {
    pub fn from_u32(v: u32) -> Self {
        match v {
            0 => Self::F32,
            1 => Self::F16,
            2 => Self::Q4_0,
            3 => Self::Q4_1,
            6 => Self::Q5_0,
            7 => Self::Q5_1,
            8 => Self::Q8_0,
            9 => Self::Q8_1,
            10 => Self::Q2_K,
            11 => Self::Q3_K,
            12 => Self::Q4_K,
            13 => Self::Q5_K,
            14 => Self::Q6_K,
            15 => Self::Q8_K,
            16 => Self::IQ2_XXS,
            17 => Self::IQ2_XS,
            18 => Self::IQ3_XXS,
            19 => Self::IQ1_S,
            20 => Self::IQ4_NL,
            21 => Self::IQ3_S,
            22 => Self::IQ2_S,
            23 => Self::IQ4_XS,
            29 => Self::IQ1_M,
            34 => Self::TQ1_0,
            35 => Self::TQ2_0,
            24 => Self::I8,
            25 => Self::I16,
            26 => Self::I32,
            27 => Self::I64,
            28 => Self::F64,
            30 => Self::BF16,
            other => Self::Other(other),
        }
    }

    pub fn block_size(self) -> Option<usize> {
        match self {
            Self::Q4_0 => Some(18),
            Self::Q4_1 => Some(20),
            Self::Q8_0 => Some(34),
            Self::Q2_K => Some(84),
            Self::Q3_K => Some(110),
            Self::Q4_K => Some(144),
            Self::Q5_0 => Some(22),
            Self::Q5_1 => Some(24),
            Self::Q5_K => Some(176),
            Self::Q6_K => Some(210),
            Self::Q8_K => Some(256), // d(4)+qs(256) approx — refine with GEMV
            Self::IQ4_NL => Some(18), // d(2)+qs(16) for QK4_NL=32
            Self::IQ2_XXS => Some(66), // d(2)+qs(64)
            Self::IQ2_XS => Some(74), // d(2)+qs(64)+scales(8)
            Self::IQ2_S => Some(82), // d(2)+qs(64)+qh(8)+scales(8)
            Self::IQ3_XXS => Some(98), // d(2)+qs(96)
            Self::IQ3_S => Some(110),
            Self::IQ1_S => Some(50),
            Self::IQ1_M => Some(56),
            Self::IQ4_XS => Some(136), // d(2)+scales_h(2)+scales_l(4)+qs(128)
            Self::TQ1_0 => Some(54),
            Self::TQ2_0 => Some(66),
            _ => None,
        }
    }

    pub fn type_size_elements(self) -> Option<usize> {
        match self {
            Self::Q4_0 | Self::Q4_1 | Self::Q5_0 | Self::Q5_1 | Self::IQ4_NL => Some(32),
            Self::Q8_0 => Some(32),
            Self::Q2_K
            | Self::Q3_K
            | Self::Q4_K
            | Self::Q5_K
            | Self::Q6_K
            | Self::Q8_K
            | Self::IQ2_XXS
            | Self::IQ2_XS
            | Self::IQ2_S
            | Self::IQ3_XXS
            | Self::IQ3_S
            | Self::IQ1_S
            | Self::IQ1_M
            | Self::IQ4_XS
            | Self::TQ1_0
            | Self::TQ2_0 => Some(256),
            Self::F32 | Self::I32 => Some(1),
            Self::F16 | Self::BF16 | Self::I16 => Some(1),
            Self::F64 | Self::I64 => Some(1),
            Self::I8 => Some(1),
            _ => None,
        }
    }
}

#[derive(Debug, Clone)]
pub enum MetadataValue {
    U8(u8),
    I8(i8),
    U16(u16),
    I16(i16),
    U32(u32),
    I32(i32),
    F32(f32),
    U64(u64),
    I64(i64),
    F64(f64),
    Bool(bool),
    String(String),
    Array(Vec<MetadataValue>),
}

impl MetadataValue {
    pub fn as_u32(&self) -> Option<u32> {
        match self {
            Self::U32(v) => Some(*v),
            Self::U64(v) => Some(*v as u32),
            Self::I32(v) if *v >= 0 => Some(*v as u32),
            Self::U16(v) => Some(*v as u32),
            Self::U8(v) => Some(*v as u32),
            _ => None,
        }
    }

    pub fn as_u64(&self) -> Option<u64> {
        match self {
            Self::U64(v) => Some(*v),
            Self::U32(v) => Some(*v as u64),
            Self::I64(v) if *v >= 0 => Some(*v as u64),
            Self::I32(v) if *v >= 0 => Some(*v as u64),
            _ => None,
        }
    }

    pub fn as_f32(&self) -> Option<f32> {
        match self {
            Self::F32(v) => Some(*v),
            Self::F64(v) => Some(*v as f32),
            _ => None,
        }
    }

    pub fn as_str(&self) -> Option<&str> {
        match self {
            Self::String(s) => Some(s),
            _ => None,
        }
    }

    pub fn as_string_array(&self) -> Option<Vec<String>> {
        match self {
            Self::Array(items) => {
                let mut out = Vec::with_capacity(items.len());
                for it in items {
                    out.push(it.as_str()?.to_string());
                }
                Some(out)
            }
            _ => None,
        }
    }

    pub fn as_f32_array(&self) -> Option<Vec<f32>> {
        match self {
            Self::Array(items) => {
                let mut out = Vec::with_capacity(items.len());
                for it in items {
                    out.push(it.as_f32()?);
                }
                Some(out)
            }
            _ => None,
        }
    }

    pub fn as_bool_array(&self) -> Option<Vec<bool>> {
        match self {
            Self::Array(items) => {
                let mut out = Vec::with_capacity(items.len());
                for it in items {
                    match it {
                        Self::Bool(b) => out.push(*b),
                        Self::U8(v) => out.push(*v != 0),
                        Self::U32(v) => out.push(*v != 0),
                        Self::I32(v) => out.push(*v != 0),
                        _ => return None,
                    }
                }
                Some(out)
            }
            _ => None,
        }
    }
}

#[derive(Debug, Clone)]
pub struct TensorInfo {
    pub name: String,
    pub dims: Vec<u64>,
    pub ggml_type: GgmlType,
    pub type_id: u32,
    /// Offset relative to start of tensor data section.
    pub offset: u64,
}

impl TensorInfo {
    pub fn n_elements(&self) -> u64 {
        self.dims.iter().product()
    }

    pub fn nrows(&self) -> usize {
        if self.dims.len() >= 2 {
            self.dims[1] as usize
        } else {
            1
        }
    }

    pub fn ncols(&self) -> usize {
        self.dims.first().copied().unwrap_or(1) as usize
    }
}

#[derive(Error, Debug)]
pub enum GgufError {
    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),
    #[error("invalid GGUF magic")]
    BadMagic,
    #[error("unsupported GGUF version {0}")]
    BadVersion(u32),
    #[error("truncated / malformed GGUF: {0}")]
    Truncated(&'static str),
    #[error("missing metadata key: {0}")]
    MissingKey(String),
    #[error("tensor not found: {0}")]
    MissingTensor(String),
    #[error("unsupported ggml type {0:?} for tensor {1}")]
    UnsupportedType(GgmlType, String),
    #[error("{0}")]
    Msg(String),
}

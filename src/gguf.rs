// gguf.rs
//
// A GGUF file starts with a fixed-size header (magic number, version, and two
// counts), followed by a list of metadata key/value pairs, followed by a list
// of tensor descriptors (name, shape, data type, and where its raw bytes live
// later in the file). This module reads just that structure — the labels and
// shapes — not the actual weight numbers yet. Think of it as reading a table
// of contents before opening any chapters.

use std::fs::File;
use std::io::{self, Read, Seek, BufReader};

// GGUF's tagged-union type system for metadata values. The u32 stored in the
// file tells us which variant to expect right after it.
#[derive(Debug, Clone)]
pub enum GgufValue {
    U8(u8),
    I8(i8),
    U16(u16),
    I16(i16),
    U32(u32),
    I32(i32),
    F32(f32),
    Bool(bool),
    String(String),
    Array(Vec<GgufValue>),
    U64(u64),
    I64(i64),
    F64(f64),
}

#[derive(Debug)]
pub struct TensorInfo {
    pub name: String,
    pub dims: Vec<u64>,
    pub dtype: u32,   // raw GGML type id (e.g. Q8_0, F32) — we decode names later
    pub offset: u64,  // byte offset into the tensor-data section
}

#[derive(Debug)]
pub struct GgufFile {
    pub version: u32,
    pub metadata: Vec<(String, GgufValue)>,
    pub tensors: Vec<TensorInfo>,
    pub tensor_data_offset: u64, // absolute byte offset in the file where tensor bytes begin
}

// Thin wrapper around a file reader so we don't repeat "read N bytes, convert
// to a number" everywhere. Every read advances the cursor automatically.
struct Reader {
    inner: BufReader<File>,
}

impl Reader {
    fn new(path: &str) -> io::Result<Self> {
        let file = File::open(path)?;
        Ok(Reader { inner: BufReader::new(file) })
    }

    fn read_exact_bytes(&mut self, n: usize) -> io::Result<Vec<u8>> {
        let mut buf = vec![0u8; n];
        self.inner.read_exact(&mut buf)?;
        Ok(buf)
    }

    fn current_pos(&mut self) -> io::Result<u64> {
        self.inner.stream_position()
    }

    // GGUF stores all numbers little-endian, so we always use from_le_bytes.
    fn read_u32(&mut self) -> io::Result<u32> {
        let b = self.read_exact_bytes(4)?;
        Ok(u32::from_le_bytes(b.try_into().unwrap()))
    }

    fn read_u64(&mut self) -> io::Result<u64> {
        let b = self.read_exact_bytes(8)?;
        Ok(u64::from_le_bytes(b.try_into().unwrap()))
    }

    fn read_i32(&mut self) -> io::Result<i32> {
        let b = self.read_exact_bytes(4)?;
        Ok(i32::from_le_bytes(b.try_into().unwrap()))
    }

    fn read_i64(&mut self) -> io::Result<i64> {
        let b = self.read_exact_bytes(8)?;
        Ok(i64::from_le_bytes(b.try_into().unwrap()))
    }

    fn read_f32(&mut self) -> io::Result<f32> {
        let b = self.read_exact_bytes(4)?;
        Ok(f32::from_le_bytes(b.try_into().unwrap()))
    }

    fn read_f64(&mut self) -> io::Result<f64> {
        let b = self.read_exact_bytes(8)?;
        Ok(f64::from_le_bytes(b.try_into().unwrap()))
    }

    fn read_u8(&mut self) -> io::Result<u8> {
        let b = self.read_exact_bytes(1)?;
        Ok(b[0])
    }

    fn read_bool(&mut self) -> io::Result<bool> {
        Ok(self.read_u8()? != 0)
    }

    // Every GGUF string is [length: u64][raw utf8 bytes], no null terminator.
    fn read_string(&mut self) -> io::Result<String> {
        let len = self.read_u64()? as usize;
        let bytes = self.read_exact_bytes(len)?;
        Ok(String::from_utf8(bytes).expect("invalid utf8 in GGUF string"))
    }

    // Reads one metadata value given its type id. Recurses for arrays, since
    // an array's elements are themselves typed values.
    fn read_value(&mut self, value_type: u32) -> io::Result<GgufValue> {
        Ok(match value_type {
            0 => GgufValue::U8(self.read_u8()?),
            1 => GgufValue::I8(self.read_u8()? as i8),
            2 => GgufValue::U16({
                let b = self.read_exact_bytes(2)?;
                u16::from_le_bytes(b.try_into().unwrap())
            }),
            3 => GgufValue::I16({
                let b = self.read_exact_bytes(2)?;
                i16::from_le_bytes(b.try_into().unwrap())
            }),
            4 => GgufValue::U32(self.read_u32()?),
            5 => GgufValue::I32(self.read_i32()?),
            6 => GgufValue::F32(self.read_f32()?),
            7 => GgufValue::Bool(self.read_bool()?),
            8 => GgufValue::String(self.read_string()?),
            9 => {
                // Array: [element_type: u32][count: u64][elements...]
                let elem_type = self.read_u32()?;
                let count = self.read_u64()?;
                let mut items = Vec::with_capacity(count as usize);
                for _ in 0..count {
                    items.push(self.read_value(elem_type)?);
                }
                GgufValue::Array(items)
            }
            10 => GgufValue::U64(self.read_u64()?),
            11 => GgufValue::I64(self.read_i64()?),
            12 => GgufValue::F64(self.read_f64()?),
            other => panic!("unknown GGUF value type id: {other}"),
        })
    }
}

pub fn parse(path: &str) -> io::Result<GgufFile> {
    let mut r = Reader::new(path)?;

    // --- Fixed 24-byte header ---
    let magic = r.read_exact_bytes(4)?;
    assert_eq!(&magic, b"GGUF", "not a valid GGUF file (bad magic number)");

    let version = r.read_u32()?;
    let tensor_count = r.read_u64()?;
    let metadata_count = r.read_u64()?;

    // --- Metadata key/value pairs ---
    let mut metadata = Vec::with_capacity(metadata_count as usize);
    for _ in 0..metadata_count {
        let key = r.read_string()?;
        let value_type = r.read_u32()?;
        let value = r.read_value(value_type)?;
        metadata.push((key, value));
    }

    // --- Tensor descriptors ---
    // Format per tensor: [name][n_dims: u32][dims: u64 * n_dims][dtype: u32][offset: u64]
    let mut tensors = Vec::with_capacity(tensor_count as usize);
    for _ in 0..tensor_count {
        let name = r.read_string()?;
        let n_dims = r.read_u32()?;
        let mut dims = Vec::with_capacity(n_dims as usize);
        for _ in 0..n_dims {
            dims.push(r.read_u64()?);
        }
        let dtype = r.read_u32()?;
        let offset = r.read_u64()?;
        tensors.push(TensorInfo { name, dims, dtype, offset });
    }

    // GGUF pads the tensor-data section to an alignment boundary (default 32),
    // configurable via the "general.alignment" metadata key.
    let alignment = metadata.iter()
        .find(|(k, _)| k == "general.alignment")
        .and_then(|(_, v)| if let GgufValue::U32(n) = v { Some(*n as u64) } else { None })
        .unwrap_or(32);

    let pos = r.current_pos()?;
    let tensor_data_offset = pos.div_ceil(alignment) * alignment;

    Ok(GgufFile { version, metadata, tensors, tensor_data_offset })
}
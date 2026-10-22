use std::{
    fmt,
    fs::File,
    io::{self, BufReader, Cursor, Read, Seek, SeekFrom},
    path::Path,
    string::{String, ToString},
    vec::Vec,
};

use p3_field::PrimeCharacteristicRing;
use serde::{Deserialize, Serialize};

use crate::{engine::F, R1csShape, R1csWitness, SparseMatEntry, SparseMatrix, SpartanWhirError};

pub const KOALABEAR_MODULUS: u32 = 2_130_706_433;
pub const MAX_CIRCOM_FILE_BYTES: usize = 1 << 30;
pub const MAX_CIRCOM_SECTIONS: usize = 64;
pub const MAX_CIRCOM_DIMENSION: usize = 1 << 24;
pub const MAX_CIRCOM_TERMS: usize = 1 << 24;
const R1CS_MAGIC: &[u8; 4] = b"r1cs";
const WTNS_MAGIC: &[u8; 4] = b"wtns";

pub type ImportedCircuit = (R1csShape<F>, R1csWitness<F>, Vec<F>);
pub type ImportedWitness = (R1csWitness<F>, Vec<F>);

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct CircomR1cs {
    pub shape: R1csShape<F>,
    pub public_outputs: usize,
    pub public_inputs: usize,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CircomAdapterError {
    Io(String),
    UnexpectedEof,
    InvalidMagic(&'static str),
    UnsupportedVersion {
        format: &'static str,
        version: u32,
    },
    MissingSection(u32),
    DuplicateSection(u32),
    UnsupportedSection {
        id: u32,
    },
    InvalidFieldSize {
        expected: u32,
        actual: u32,
    },
    InvalidModulus {
        expected: u32,
        actual: Vec<u8>,
    },
    InvalidFieldElement,
    InvalidWitnessLength {
        expected: usize,
        actual: usize,
    },
    ResourceLimitExceeded {
        resource: &'static str,
        limit: u64,
        actual: u64,
    },
    AllocationFailed(&'static str),
    InvalidShape,
    UnsatisfiedConstraint {
        row: usize,
    },
}

impl fmt::Display for CircomAdapterError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Io(e) => write!(f, "I/O error: {e}"),
            Self::UnexpectedEof => write!(f, "unexpected end of file"),
            Self::InvalidMagic(format) => write!(f, "invalid {format} magic"),
            Self::UnsupportedVersion { format, version } => {
                write!(f, "unsupported {format} version {version}")
            }
            Self::MissingSection(id) => write!(f, "missing section {id}"),
            Self::DuplicateSection(id) => write!(f, "duplicate section {id}"),
            Self::UnsupportedSection { id: 4 | 5 } => {
                write!(f, "custom-gate R1CS sections are not supported")
            }
            Self::UnsupportedSection { id } => write!(f, "unsupported R1CS section {id}"),
            Self::InvalidFieldSize { expected, actual } => {
                write!(f, "invalid field size: expected {expected}, got {actual}")
            }
            Self::InvalidModulus { .. } => write!(f, "invalid modulus"),
            Self::InvalidFieldElement => write!(f, "field element is not canonical KoalaBear"),
            Self::InvalidWitnessLength { expected, actual } => {
                write!(
                    f,
                    "invalid witness length: expected {expected}, got {actual}"
                )
            }
            Self::ResourceLimitExceeded {
                resource,
                limit,
                actual,
            } => write!(
                f,
                "Circom {resource} exceeds limit: maximum {limit}, got {actual}"
            ),
            Self::AllocationFailed(resource) => {
                write!(f, "failed to allocate memory for Circom {resource}")
            }
            Self::InvalidShape => write!(f, "invalid R1CS shape"),
            Self::UnsatisfiedConstraint { row } => write!(f, "unsatisfied constraint at row {row}"),
        }
    }
}

impl std::error::Error for CircomAdapterError {}

impl From<std::io::Error> for CircomAdapterError {
    fn from(value: std::io::Error) -> Self {
        Self::Io(value.to_string())
    }
}

fn ensure_limit(
    resource: &'static str,
    actual: usize,
    limit: usize,
) -> Result<(), CircomAdapterError> {
    if actual > limit {
        return Err(CircomAdapterError::ResourceLimitExceeded {
            resource,
            limit: limit as u64,
            actual: actual as u64,
        });
    }
    Ok(())
}

fn validate_file_size(bytes: &[u8], resource: &'static str) -> Result<(), CircomAdapterError> {
    ensure_limit(resource, bytes.len(), MAX_CIRCOM_FILE_BYTES)
}

fn read_path_bounded(
    path: impl AsRef<Path>,
    resource: &'static str,
) -> Result<Vec<u8>, CircomAdapterError> {
    let file = File::open(path)?;
    let mut bytes = Vec::new();
    file.take(MAX_CIRCOM_FILE_BYTES as u64 + 1)
        .read_to_end(&mut bytes)?;
    validate_file_size(&bytes, resource)?;
    Ok(bytes)
}

fn try_vec_with_capacity<T>(
    capacity: usize,
    resource: &'static str,
) -> Result<Vec<T>, CircomAdapterError> {
    let mut values = Vec::new();
    values
        .try_reserve_exact(capacity)
        .map_err(|_| CircomAdapterError::AllocationFailed(resource))?;
    Ok(values)
}

/// Import a KoalaBear Circom `.r1cs` and `.wtns` pair.
///
/// Import validates the file headers, remaps Circom wires into Spartan-WHIR's
/// `[witness | constant_one | public_outputs || public_inputs]` layout, and
/// runs a constraint-satisfaction check against the imported witness.
pub fn import_paths(
    r1cs_path: impl AsRef<Path>,
    wtns_path: impl AsRef<Path>,
) -> Result<ImportedCircuit, CircomAdapterError> {
    let r1cs = import_r1cs_path(r1cs_path)?;
    let (witness, public_inputs) = import_witness_path(&r1cs.shape, wtns_path)?;
    Ok((r1cs.shape, witness, public_inputs))
}

pub fn import_bytes(r1cs: &[u8], wtns: &[u8]) -> Result<ImportedCircuit, CircomAdapterError> {
    validate_file_size(r1cs, "R1CS file")?;
    validate_file_size(wtns, "witness file")?;
    let r1cs = import_r1cs_bytes(r1cs)?;
    let witness = parse_wtns(wtns)?;
    let (witness, public_inputs) = import_witness_values(&r1cs.shape, witness)?;
    Ok((r1cs.shape, witness, public_inputs))
}

/// Import a KoalaBear R1CS file with two passes into the final sparse matrices.
pub fn import_r1cs_path(r1cs_path: impl AsRef<Path>) -> Result<CircomR1cs, CircomAdapterError> {
    let file = File::open(r1cs_path)?;
    let length = file.metadata()?.len();
    ensure_limit(
        "R1CS file",
        length.min(MAX_CIRCOM_FILE_BYTES as u64 + 1) as usize,
        MAX_CIRCOM_FILE_BYTES,
    )?;
    let mut reader = BufReader::with_capacity(1 << 20, file);
    import_r1cs_reader(&mut reader, length)
}

pub fn import_r1cs_bytes(r1cs: &[u8]) -> Result<CircomR1cs, CircomAdapterError> {
    validate_file_size(r1cs, "R1CS file")?;
    import_r1cs_reader(&mut Cursor::new(r1cs), r1cs.len() as u64)
}

pub fn import_witness_path(
    shape: &R1csShape<F>,
    wtns_path: impl AsRef<Path>,
) -> Result<ImportedWitness, CircomAdapterError> {
    let wtns = read_path_bounded(wtns_path, "witness file")?;
    import_witness_bytes(shape, &wtns)
}

pub fn import_witness_bytes(
    shape: &R1csShape<F>,
    wtns: &[u8],
) -> Result<ImportedWitness, CircomAdapterError> {
    validate_file_size(wtns, "witness file")?;
    let witness_values = parse_wtns(wtns)?;
    import_witness_values(shape, witness_values)
}

pub fn import_witness_bytes_with_layout(
    validation_shape: &R1csShape<F>,
    num_vars: usize,
    num_io: usize,
    wtns: &[u8],
) -> Result<ImportedWitness, CircomAdapterError> {
    validate_file_size(wtns, "witness file")?;
    let witness_values = parse_wtns(wtns)?;
    import_witness_values_with_layout(validation_shape, num_vars, num_io, witness_values)
}

pub fn import_witness_values(
    shape: &R1csShape<F>,
    witness_values: Vec<F>,
) -> Result<ImportedWitness, CircomAdapterError> {
    import_witness_values_with_layout(shape, shape.num_vars, shape.num_io, witness_values)
}

pub fn import_witness_values_with_layout(
    validation_shape: &R1csShape<F>,
    num_vars: usize,
    num_io: usize,
    witness_values: Vec<F>,
) -> Result<ImportedWitness, CircomAdapterError> {
    let (witness, public_inputs) =
        split_witness_values_with_layout(num_vars, num_io, witness_values)?;
    validate_satisfaction(validation_shape, &witness, &public_inputs)?;
    Ok((witness, public_inputs))
}

fn split_witness_values_with_layout(
    num_vars: usize,
    num_io: usize,
    witness_values: Vec<F>,
) -> Result<ImportedWitness, CircomAdapterError> {
    let expected = num_vars
        .checked_add(num_io)
        .and_then(|n| n.checked_add(1))
        .ok_or(CircomAdapterError::InvalidShape)?;
    if witness_values.len() != expected {
        return Err(CircomAdapterError::InvalidWitnessLength {
            expected,
            actual: witness_values.len(),
        });
    }
    let one_plus_io = num_io
        .checked_add(1)
        .ok_or(CircomAdapterError::InvalidShape)?;
    let mut public_inputs = try_vec_with_capacity(num_io, "public inputs")?;
    public_inputs.extend_from_slice(&witness_values[1..one_plus_io]);
    let witness_len = witness_values.len() - one_plus_io;
    let mut witness_values_out = try_vec_with_capacity(witness_len, "witness values")?;
    witness_values_out.extend_from_slice(&witness_values[one_plus_io..]);
    let witness = R1csWitness {
        w: witness_values_out,
    };
    Ok((witness, public_inputs))
}

fn import_r1cs_reader(
    reader: &mut (impl Read + Seek),
    length: u64,
) -> Result<CircomR1cs, CircomAdapterError> {
    if read_r1cs_bytes::<4>(reader)? != *R1CS_MAGIC {
        return Err(CircomAdapterError::InvalidMagic("r1cs"));
    }
    let version = read_r1cs_u32(reader)?;
    if version != 1 {
        return Err(CircomAdapterError::UnsupportedVersion {
            format: "r1cs",
            version,
        });
    }
    let sections = read_r1cs_sections(reader, length)?;
    for section in &sections {
        match section.id {
            1 | 2 | 3 => {}
            id => return Err(CircomAdapterError::UnsupportedSection { id }),
        }
    }
    let header_section = r1cs_section(&sections, 1)?;
    reader.seek(SeekFrom::Start(header_section.start))?;
    let mut header_bytes = [0u8; 36];
    let header_length = header_section.length.min(header_bytes.len() as u64) as usize;
    read_r1cs_exact(reader, &mut header_bytes[..header_length])?;
    let header = parse_r1cs_header(&header_bytes[..header_length])?;
    let constraint_section = r1cs_section(&sections, 2)?;

    reader.seek(SeekFrom::Start(constraint_section.start))?;
    let counts = {
        let mut constraints = reader.take(constraint_section.length);
        if header.number_of_constraints as u64 > constraints.limit() / 12 {
            return Err(CircomAdapterError::UnexpectedEof);
        }
        let mut counts = [0usize; 3];
        let mut total_terms = 0usize;
        for _ in 0..header.number_of_constraints {
            for count in &mut counts {
                let terms = read_r1cs_u32(&mut constraints)? as usize;
                add_r1cs_terms(&mut total_terms, terms)?;
                if terms as u64 > constraints.limit() / 8 {
                    return Err(CircomAdapterError::UnexpectedEof);
                }
                *count += terms;
                for _ in 0..terms {
                    let term = read_r1cs_bytes::<8>(&mut constraints)?;
                    let _ = parse_field(&term[4..])?;
                }
            }
        }
        counts
    };

    let num_io = header
        .public_outputs
        .checked_add(header.public_inputs)
        .ok_or(CircomAdapterError::InvalidShape)?;
    let one_plus_io = num_io
        .checked_add(1)
        .ok_or(CircomAdapterError::InvalidShape)?;
    let non_private_wires = one_plus_io
        .checked_add(header.private_inputs)
        .ok_or(CircomAdapterError::InvalidShape)?;
    if non_private_wires > header.total_wires {
        return Err(CircomAdapterError::InvalidShape);
    }
    let num_vars = header
        .total_wires
        .checked_sub(one_plus_io)
        .ok_or(CircomAdapterError::InvalidShape)?;
    let num_cols = num_vars
        .checked_add(one_plus_io)
        .ok_or(CircomAdapterError::InvalidShape)?;

    let mut entries: [Vec<SparseMatEntry<F>>; 3] = [
        try_vec_with_capacity(counts[0], "matrix terms")?,
        try_vec_with_capacity(counts[1], "matrix terms")?,
        try_vec_with_capacity(counts[2], "matrix terms")?,
    ];
    reader.seek(SeekFrom::Start(constraint_section.start))?;
    let mut constraints = reader.take(constraint_section.length);
    let mut total_terms = 0usize;
    for row in 0..header.number_of_constraints {
        for (matrix, count) in entries.iter_mut().zip(counts) {
            let terms = read_r1cs_u32(&mut constraints)? as usize;
            add_r1cs_terms(&mut total_terms, terms)?;
            if terms > count.saturating_sub(matrix.len()) {
                return Err(CircomAdapterError::InvalidShape);
            }
            if terms as u64 > constraints.limit() / 8 {
                return Err(CircomAdapterError::UnexpectedEof);
            }
            for _ in 0..terms {
                let term = read_r1cs_bytes::<8>(&mut constraints)?;
                let wire = u32::from_le_bytes(term[..4].try_into().unwrap()) as usize;
                let coefficient = parse_field(&term[4..])?;
                if wire >= header.total_wires {
                    return Err(CircomAdapterError::InvalidShape);
                }
                let col = if wire == 0 {
                    num_vars
                } else if wire <= num_io {
                    num_vars + wire
                } else {
                    wire - num_io - 1
                };
                matrix.push(SparseMatEntry {
                    row,
                    col,
                    val: coefficient,
                });
            }
        }
    }
    for (matrix, count) in entries.iter().zip(counts) {
        if matrix.len() != count {
            return Err(CircomAdapterError::InvalidShape);
        }
    }
    let [a, b, c] = entries.map(|entries| SparseMatrix {
        num_rows: header.number_of_constraints,
        num_cols,
        entries,
    });
    let shape = R1csShape {
        num_cons: header.number_of_constraints,
        num_vars,
        num_io,
        a,
        b,
        c,
    };

    Ok(CircomR1cs {
        shape,
        public_outputs: header.public_outputs,
        public_inputs: header.public_inputs,
    })
}

#[derive(Clone, Copy)]
struct R1csSection {
    id: u32,
    start: u64,
    length: u64,
}

fn read_r1cs_sections(
    reader: &mut (impl Read + Seek),
    length: u64,
) -> Result<Vec<R1csSection>, CircomAdapterError> {
    let count = read_r1cs_u32(reader)? as usize;
    ensure_limit("sections", count, MAX_CIRCOM_SECTIONS)?;
    if count as u64 > length.saturating_sub(reader.stream_position()?) / 12 {
        return Err(CircomAdapterError::UnexpectedEof);
    }
    let mut sections = try_vec_with_capacity(count, "sections")?;
    for _ in 0..count {
        let id = read_r1cs_u32(reader)?;
        if sections
            .iter()
            .any(|section: &R1csSection| section.id == id)
        {
            return Err(CircomAdapterError::DuplicateSection(id));
        }
        let section_length = u64::from_le_bytes(read_r1cs_bytes(reader)?);
        let section_size = usize::try_from(section_length).map_err(|_| {
            CircomAdapterError::ResourceLimitExceeded {
                resource: "section bytes",
                limit: MAX_CIRCOM_FILE_BYTES as u64,
                actual: section_length,
            }
        })?;
        ensure_limit("section bytes", section_size, MAX_CIRCOM_FILE_BYTES)?;
        let start = reader.stream_position()?;
        let end = start
            .checked_add(section_length)
            .ok_or(CircomAdapterError::UnexpectedEof)?;
        if end > length {
            return Err(CircomAdapterError::UnexpectedEof);
        }
        sections.push(R1csSection {
            id,
            start,
            length: section_length,
        });
        reader.seek(SeekFrom::Start(end))?;
    }
    Ok(sections)
}

fn r1cs_section(sections: &[R1csSection], id: u32) -> Result<R1csSection, CircomAdapterError> {
    sections
        .iter()
        .find(|section| section.id == id)
        .copied()
        .ok_or(CircomAdapterError::MissingSection(id))
}

fn add_r1cs_terms(total_terms: &mut usize, terms: usize) -> Result<(), CircomAdapterError> {
    let next_total =
        total_terms
            .checked_add(terms)
            .ok_or(CircomAdapterError::ResourceLimitExceeded {
                resource: "linear-combination terms",
                limit: MAX_CIRCOM_TERMS as u64,
                actual: u64::MAX,
            })?;
    ensure_limit("linear-combination terms", next_total, MAX_CIRCOM_TERMS)?;
    *total_terms = next_total;
    Ok(())
}

fn read_r1cs_exact(reader: &mut impl Read, bytes: &mut [u8]) -> Result<(), CircomAdapterError> {
    reader.read_exact(bytes).map_err(|error| {
        if error.kind() == io::ErrorKind::UnexpectedEof {
            CircomAdapterError::UnexpectedEof
        } else {
            error.into()
        }
    })
}

fn read_r1cs_bytes<const N: usize>(reader: &mut impl Read) -> Result<[u8; N], CircomAdapterError> {
    let mut bytes = [0u8; N];
    read_r1cs_exact(reader, &mut bytes)?;
    Ok(bytes)
}

fn read_r1cs_u32(reader: &mut impl Read) -> Result<u32, CircomAdapterError> {
    Ok(u32::from_le_bytes(read_r1cs_bytes(reader)?))
}

/// Validate an imported Circom witness against a Spartan-WHIR R1CS shape.
///
/// `shape` may be the padded shape stored in a proving key. The witness may be
/// unpadded; it is zero-padded to `shape.num_vars` before matrix evaluation.
pub fn validate_satisfaction(
    shape: &R1csShape<F>,
    witness: &R1csWitness<F>,
    public_inputs: &[F],
) -> Result<(), CircomAdapterError> {
    let mut z = shape.witness_to_mle(&witness.w).map_err(|err| match err {
        SpartanWhirError::InvalidWitnessLength => CircomAdapterError::InvalidWitnessLength {
            expected: shape.num_vars,
            actual: witness.w.len(),
        },
        _ => CircomAdapterError::InvalidShape,
    })?;
    z.push(F::ONE);
    z.extend_from_slice(public_inputs);
    let (az, bz, cz) = shape.multiply_vec(&z).map_err(|err| match err {
        SpartanWhirError::InvalidWitnessLength => CircomAdapterError::InvalidShape,
        _ => CircomAdapterError::InvalidShape,
    })?;
    for row in 0..shape.num_cons {
        if az[row] * bz[row] != cz[row] {
            return Err(CircomAdapterError::UnsatisfiedConstraint { row });
        }
    }
    Ok(())
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct R1csHeader {
    total_wires: usize,
    public_outputs: usize,
    public_inputs: usize,
    private_inputs: usize,
    number_of_constraints: usize,
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct Section<'a> {
    id: u32,
    data: &'a [u8],
}

fn parse_r1cs_header(bytes: &[u8]) -> Result<R1csHeader, CircomAdapterError> {
    let mut reader = Reader::new(bytes);
    let field_size = reader.u32()?;
    if field_size != 4 {
        return Err(CircomAdapterError::InvalidFieldSize {
            expected: 4,
            actual: field_size,
        });
    }
    validate_modulus(reader.take(4)?)?;
    let total_wires = reader.u32()? as usize;
    let public_outputs = reader.u32()? as usize;
    let public_inputs = reader.u32()? as usize;
    let private_inputs = reader.u32()? as usize;
    let number_of_labels = reader.u64()?;
    let number_of_constraints = reader.u32()? as usize;
    ensure_limit("wires", total_wires, MAX_CIRCOM_DIMENSION)?;
    ensure_limit("public outputs", public_outputs, MAX_CIRCOM_DIMENSION)?;
    ensure_limit("public inputs", public_inputs, MAX_CIRCOM_DIMENSION)?;
    ensure_limit("private inputs", private_inputs, MAX_CIRCOM_DIMENSION)?;
    if number_of_labels > MAX_CIRCOM_DIMENSION as u64 {
        return Err(CircomAdapterError::ResourceLimitExceeded {
            resource: "labels",
            limit: MAX_CIRCOM_DIMENSION as u64,
            actual: number_of_labels,
        });
    }
    ensure_limit("constraints", number_of_constraints, MAX_CIRCOM_DIMENSION)?;
    Ok(R1csHeader {
        total_wires,
        public_outputs,
        public_inputs,
        private_inputs,
        number_of_constraints,
    })
}

fn parse_wtns(bytes: &[u8]) -> Result<Vec<F>, CircomAdapterError> {
    let mut reader = Reader::new(bytes);
    if reader.take(4)? != WTNS_MAGIC {
        return Err(CircomAdapterError::InvalidMagic("wtns"));
    }
    let version = reader.u32()?;
    if version != 2 {
        return Err(CircomAdapterError::UnsupportedVersion {
            format: "wtns",
            version,
        });
    }
    let sections = read_sections(&mut reader)?;
    let header = section(&sections, 1)?;
    let witness = section(&sections, 2)?;

    let mut header_reader = Reader::new(header);
    let n8 = header_reader.u32()?;
    if n8 != 4 {
        return Err(CircomAdapterError::InvalidFieldSize {
            expected: 4,
            actual: n8,
        });
    }
    validate_modulus(header_reader.take(4)?)?;
    let n_witness = header_reader.u32()? as usize;
    ensure_limit("witness values", n_witness, MAX_CIRCOM_DIMENSION)?;
    let expected_witness_bytes =
        n_witness
            .checked_mul(4)
            .ok_or(CircomAdapterError::ResourceLimitExceeded {
                resource: "witness bytes",
                limit: MAX_CIRCOM_FILE_BYTES as u64,
                actual: u64::MAX,
            })?;
    if witness.len() != expected_witness_bytes {
        return Err(CircomAdapterError::InvalidWitnessLength {
            expected: n_witness,
            actual: witness.len() / 4,
        });
    }

    let mut values = try_vec_with_capacity(n_witness, "witness values")?;
    for bytes in witness.chunks_exact(4) {
        values.push(parse_field(bytes)?);
    }
    Ok(values)
}

fn read_sections<'a>(reader: &mut Reader<'a>) -> Result<Vec<Section<'a>>, CircomAdapterError> {
    let n_sections = reader.u32()? as usize;
    ensure_limit("sections", n_sections, MAX_CIRCOM_SECTIONS)?;
    if n_sections > reader.remaining() / 12 {
        return Err(CircomAdapterError::UnexpectedEof);
    }
    let mut sections = try_vec_with_capacity(n_sections, "sections")?;
    for _ in 0..n_sections {
        let id = reader.u32()?;
        if sections
            .iter()
            .any(|section: &Section<'_>| section.id == id)
        {
            return Err(CircomAdapterError::DuplicateSection(id));
        }
        let len_u64 = reader.u64()?;
        let len =
            usize::try_from(len_u64).map_err(|_| CircomAdapterError::ResourceLimitExceeded {
                resource: "section bytes",
                limit: MAX_CIRCOM_FILE_BYTES as u64,
                actual: len_u64,
            })?;
        ensure_limit("section bytes", len, MAX_CIRCOM_FILE_BYTES)?;
        let data = reader.take(len)?;
        sections.push(Section { id, data });
    }
    Ok(sections)
}

fn section<'a>(sections: &'a [Section<'a>], id: u32) -> Result<&'a [u8], CircomAdapterError> {
    sections
        .iter()
        .find(|section| section.id == id)
        .map(|section| section.data)
        .ok_or(CircomAdapterError::MissingSection(id))
}

fn validate_modulus(bytes: &[u8]) -> Result<(), CircomAdapterError> {
    if bytes == KOALABEAR_MODULUS.to_le_bytes() {
        Ok(())
    } else {
        Err(CircomAdapterError::InvalidModulus {
            expected: KOALABEAR_MODULUS,
            actual: bytes.to_vec(),
        })
    }
}

fn parse_field(bytes: &[u8]) -> Result<F, CircomAdapterError> {
    let raw = u32::from_le_bytes(
        bytes
            .try_into()
            .map_err(|_| CircomAdapterError::UnexpectedEof)?,
    );
    if raw >= KOALABEAR_MODULUS {
        return Err(CircomAdapterError::InvalidFieldElement);
    }
    Ok(F::from_u32(raw))
}

struct Reader<'a> {
    bytes: &'a [u8],
    pos: usize,
}

impl<'a> Reader<'a> {
    fn new(bytes: &'a [u8]) -> Self {
        Self { bytes, pos: 0 }
    }

    fn take(&mut self, len: usize) -> Result<&'a [u8], CircomAdapterError> {
        let end = self
            .pos
            .checked_add(len)
            .ok_or(CircomAdapterError::UnexpectedEof)?;
        if end > self.bytes.len() {
            return Err(CircomAdapterError::UnexpectedEof);
        }
        let out = &self.bytes[self.pos..end];
        self.pos = end;
        Ok(out)
    }

    fn remaining(&self) -> usize {
        self.bytes.len() - self.pos
    }

    fn u32(&mut self) -> Result<u32, CircomAdapterError> {
        let bytes: [u8; 4] = self
            .take(4)?
            .try_into()
            .map_err(|_| CircomAdapterError::UnexpectedEof)?;
        Ok(u32::from_le_bytes(bytes))
    }

    fn u64(&mut self) -> Result<u64, CircomAdapterError> {
        let bytes: [u8; 8] = self
            .take(8)?
            .try_into()
            .map_err(|_| CircomAdapterError::UnexpectedEof)?;
        Ok(u64::from_le_bytes(bytes))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn u32_le(out: &mut Vec<u8>, value: u32) {
        out.extend_from_slice(&value.to_le_bytes());
    }

    fn u64_le(out: &mut Vec<u8>, value: u64) {
        out.extend_from_slice(&value.to_le_bytes());
    }

    fn field(out: &mut Vec<u8>, value: u32) {
        out.extend_from_slice(&value.to_le_bytes());
    }

    fn section(out: &mut Vec<u8>, id: u32, data: &[u8]) {
        u32_le(out, id);
        u64_le(out, data.len() as u64);
        out.extend_from_slice(data);
    }

    fn lc(out: &mut Vec<u8>, terms: &[(u32, u32)]) {
        u32_le(out, terms.len() as u32);
        for (wire, coeff) in terms {
            u32_le(out, *wire);
            field(out, *coeff);
        }
    }

    fn sample_r1cs(custom_gates: bool, modulus: u32) -> Vec<u8> {
        sample_r1cs_with_constraints(custom_gates, modulus, 2)
    }

    fn sample_r1cs_with_constraints(
        custom_gates: bool,
        modulus: u32,
        constraints_count: u32,
    ) -> Vec<u8> {
        let mut header = Vec::new();
        u32_le(&mut header, 4);
        field(&mut header, modulus);
        u32_le(&mut header, 5); // [1, output, public input, private input, internal]
        u32_le(&mut header, 1);
        u32_le(&mut header, 1);
        u32_le(&mut header, 1);
        u64_le(&mut header, 5);
        u32_le(&mut header, constraints_count);

        let mut constraints = Vec::new();
        lc(&mut constraints, &[(3, 1)]);
        lc(&mut constraints, &[(2, 1)]);
        lc(&mut constraints, &[(4, 1)]);
        lc(&mut constraints, &[(4, 1), (0, 1)]);
        lc(&mut constraints, &[(0, 1)]);
        lc(&mut constraints, &[(1, 1)]);
        if constraints_count == 3 {
            lc(&mut constraints, &[(0, 1)]);
            lc(&mut constraints, &[(0, 1)]);
            lc(&mut constraints, &[(0, 1)]);
        }

        let mut r1cs = Vec::new();
        r1cs.extend_from_slice(R1CS_MAGIC);
        u32_le(&mut r1cs, 1);
        u32_le(&mut r1cs, if custom_gates { 4 } else { 3 });
        section(&mut r1cs, 1, &header);
        section(&mut r1cs, 2, &constraints);
        section(&mut r1cs, 3, &[]);
        if custom_gates {
            section(&mut r1cs, 4, &[]);
        }
        r1cs
    }

    fn sample_wtns(values: &[u32], n8: u32, modulus: u32) -> Vec<u8> {
        let mut header = Vec::new();
        u32_le(&mut header, n8);
        let mut modulus_bytes = [0u8; 8];
        modulus_bytes[..4].copy_from_slice(&modulus.to_le_bytes());
        header.extend_from_slice(&modulus_bytes[..n8 as usize]);
        u32_le(&mut header, values.len() as u32);

        let mut body = Vec::new();
        for value in values {
            let mut bytes = [0u8; 8];
            bytes[..4].copy_from_slice(&value.to_le_bytes());
            body.extend_from_slice(&bytes[..n8 as usize]);
        }

        let mut wtns = Vec::new();
        wtns.extend_from_slice(WTNS_MAGIC);
        u32_le(&mut wtns, 2);
        u32_le(&mut wtns, 2);
        section(&mut wtns, 1, &header);
        section(&mut wtns, 2, &body);
        wtns
    }

    #[test]
    fn imports_unpadded_layout() {
        let r1cs = sample_r1cs(false, KOALABEAR_MODULUS);
        let wtns = sample_wtns(&[1, 36, 5, 7, 35], 4, KOALABEAR_MODULUS);

        let (shape, witness, public_inputs) = import_bytes(&r1cs, &wtns).unwrap();

        assert_eq!(public_inputs, vec![F::from_u32(36), F::from_u32(5)]);
        assert_eq!(witness.w, vec![F::from_u32(7), F::from_u32(35)]);
        assert_eq!(shape.num_cons, 2);
        assert_eq!(shape.num_vars, 2);
        assert_eq!(shape.num_io, 2);
    }

    #[test]
    fn imported_shape_can_be_regular_padded_by_protocol_setup_path() {
        let r1cs = sample_r1cs_with_constraints(false, KOALABEAR_MODULUS, 3);
        let wtns = sample_wtns(&[1, 36, 5, 7, 35], 4, KOALABEAR_MODULUS);

        let (shape, witness, public_inputs) = import_bytes(&r1cs, &wtns).unwrap();
        assert_eq!(shape.num_cons, 3);
        assert_eq!(shape.num_vars, witness.w.len());

        let padded = shape.pad_regular().unwrap();
        assert_eq!(padded.num_cons, 4);
        assert_eq!(padded.num_vars, 4);

        let mut z = padded.witness_to_mle(&witness.w).unwrap();
        z.push(F::ONE);
        z.extend_from_slice(&public_inputs);
        let (az, bz, cz) = padded.multiply_vec(&z).unwrap();
        for row in 0..padded.num_cons {
            assert_eq!(az[row] * bz[row], cz[row]);
        }
    }

    #[test]
    fn accepts_boundary_coefficients() {
        assert_eq!(parse_field(&0u32.to_le_bytes()).unwrap(), F::from_u32(0));
        assert_eq!(parse_field(&1u32.to_le_bytes()).unwrap(), F::from_u32(1));
        assert_eq!(
            parse_field(&(KOALABEAR_MODULUS - 1).to_le_bytes()).unwrap(),
            F::from_u32(KOALABEAR_MODULUS - 1)
        );
        assert_eq!(
            parse_field(&KOALABEAR_MODULUS.to_le_bytes()),
            Err(CircomAdapterError::InvalidFieldElement)
        );
    }

    #[test]
    fn accepts_omitted_label_map_count() {
        let mut r1cs = sample_r1cs(false, KOALABEAR_MODULUS);
        let labels_offset = 48;
        r1cs[labels_offset..labels_offset + 8].fill(0);
        let wtns = sample_wtns(&[1, 36, 5, 7, 35], 4, KOALABEAR_MODULUS);

        let (shape, _, _) = import_bytes(&r1cs, &wtns).unwrap();
        assert_eq!(shape.num_vars, 2);
    }

    #[test]
    fn rejects_wrong_r1cs_modulus() {
        let r1cs = sample_r1cs(false, KOALABEAR_MODULUS - 2);
        let wtns = sample_wtns(&[1, 36, 5, 7, 35], 4, KOALABEAR_MODULUS);
        assert!(matches!(
            import_bytes(&r1cs, &wtns),
            Err(CircomAdapterError::InvalidModulus { .. })
        ));
    }

    #[test]
    fn rejects_wrong_wtns_width_and_modulus() {
        let r1cs = sample_r1cs(false, KOALABEAR_MODULUS);
        let wrong_width = sample_wtns(&[1, 36, 5, 7, 35], 8, KOALABEAR_MODULUS);
        assert!(matches!(
            import_bytes(&r1cs, &wrong_width),
            Err(CircomAdapterError::InvalidFieldSize { actual: 8, .. })
        ));

        let wrong_modulus = sample_wtns(&[1, 36, 5, 7, 35], 4, KOALABEAR_MODULUS - 1);
        assert!(matches!(
            import_bytes(&r1cs, &wrong_modulus),
            Err(CircomAdapterError::InvalidModulus { .. })
        ));
    }

    #[test]
    fn rejects_malformed_wtns_and_mismatched_lengths() {
        let r1cs = sample_r1cs(false, KOALABEAR_MODULUS);
        assert!(matches!(
            import_bytes(&r1cs, b"wtn"),
            Err(CircomAdapterError::UnexpectedEof)
        ));

        let short = sample_wtns(&[1, 36, 5, 7], 4, KOALABEAR_MODULUS);
        assert!(matches!(
            import_bytes(&r1cs, &short),
            Err(CircomAdapterError::InvalidWitnessLength {
                expected: 5,
                actual: 4
            })
        ));
    }

    #[test]
    fn rejects_custom_gate_r1cs() {
        let r1cs = sample_r1cs(true, KOALABEAR_MODULUS);
        let wtns = sample_wtns(&[1, 36, 5, 7, 35], 4, KOALABEAR_MODULUS);
        assert!(matches!(
            import_bytes(&r1cs, &wtns),
            Err(CircomAdapterError::UnsupportedSection { id: 4 })
        ));
    }

    #[test]
    fn rejects_unknown_r1cs_section_without_calling_it_custom_gates() {
        let mut r1cs = sample_r1cs(false, KOALABEAR_MODULUS);
        r1cs[8..12].copy_from_slice(&4u32.to_le_bytes());
        section(&mut r1cs, 9, &[]);
        let wtns = sample_wtns(&[1, 36, 5, 7, 35], 4, KOALABEAR_MODULUS);

        assert!(matches!(
            import_bytes(&r1cs, &wtns),
            Err(CircomAdapterError::UnsupportedSection { id: 9 })
        ));
    }

    #[test]
    fn rejects_oversized_section_count_before_allocation() {
        let mut bytes = Vec::new();
        u32_le(&mut bytes, (MAX_CIRCOM_SECTIONS + 1) as u32);
        let mut reader = Reader::new(&bytes);

        assert_eq!(
            read_sections(&mut reader),
            Err(CircomAdapterError::ResourceLimitExceeded {
                resource: "sections",
                limit: MAX_CIRCOM_SECTIONS as u64,
                actual: (MAX_CIRCOM_SECTIONS + 1) as u64,
            })
        );
    }

    #[test]
    fn rejects_oversized_r1cs_dimensions_before_allocation() {
        let mut header = Vec::new();
        u32_le(&mut header, 4);
        field(&mut header, KOALABEAR_MODULUS);
        u32_le(&mut header, 5);
        u32_le(&mut header, 1);
        u32_le(&mut header, 1);
        u32_le(&mut header, 1);
        u64_le(&mut header, 5);
        u32_le(&mut header, (MAX_CIRCOM_DIMENSION + 1) as u32);

        assert_eq!(
            parse_r1cs_header(&header),
            Err(CircomAdapterError::ResourceLimitExceeded {
                resource: "constraints",
                limit: MAX_CIRCOM_DIMENSION as u64,
                actual: (MAX_CIRCOM_DIMENSION + 1) as u64,
            })
        );
    }

    #[test]
    fn rejects_oversized_linear_combination_before_allocation() {
        let mut bytes = sample_r1cs(false, KOALABEAR_MODULUS);
        bytes[72..76].copy_from_slice(&((MAX_CIRCOM_TERMS + 1) as u32).to_le_bytes());

        assert_eq!(
            import_r1cs_bytes(&bytes),
            Err(CircomAdapterError::ResourceLimitExceeded {
                resource: "linear-combination terms",
                limit: MAX_CIRCOM_TERMS as u64,
                actual: (MAX_CIRCOM_TERMS + 1) as u64,
            })
        );
    }

    #[test]
    fn rejects_oversized_witness_count_before_allocation() {
        let mut header = Vec::new();
        u32_le(&mut header, 4);
        field(&mut header, KOALABEAR_MODULUS);
        u32_le(&mut header, (MAX_CIRCOM_DIMENSION + 1) as u32);

        let mut wtns = Vec::new();
        wtns.extend_from_slice(WTNS_MAGIC);
        u32_le(&mut wtns, 2);
        u32_le(&mut wtns, 2);
        section(&mut wtns, 1, &header);
        section(&mut wtns, 2, &[]);

        assert_eq!(
            parse_wtns(&wtns),
            Err(CircomAdapterError::ResourceLimitExceeded {
                resource: "witness values",
                limit: MAX_CIRCOM_DIMENSION as u64,
                actual: (MAX_CIRCOM_DIMENSION + 1) as u64,
            })
        );
    }
}

#[cfg(test)]
mod streaming_import_tests {
    use super::*;
    use std::{
        fs::{self, OpenOptions},
        io::{self, BufReader, Read, Seek, SeekFrom, Write},
        path::{Path, PathBuf},
        sync::atomic::{AtomicU64, Ordering},
    };

    struct TemporaryFile(PathBuf);

    impl TemporaryFile {
        fn new(bytes: &[u8]) -> Self {
            static NEXT: AtomicU64 = AtomicU64::new(0);
            let path = std::env::temp_dir().join(format!(
                "spartan-whir-import-{}-{}",
                std::process::id(),
                NEXT.fetch_add(1, Ordering::Relaxed)
            ));
            let mut file = OpenOptions::new()
                .write(true)
                .create_new(true)
                .open(&path)
                .unwrap();
            file.write_all(bytes).unwrap();
            Self(path)
        }

        fn path(&self) -> &Path {
            &self.0
        }
    }

    impl Drop for TemporaryFile {
        fn drop(&mut self) {
            let _ = fs::remove_file(&self.0);
        }
    }

    fn fixture_sections() -> Vec<(u32, Vec<u8>)> {
        let mut header = Vec::new();
        for value in [4u32, KOALABEAR_MODULUS, 5, 1, 1, 1] {
            header.extend(value.to_le_bytes());
        }
        header.extend(5u64.to_le_bytes());
        header.extend(2u32.to_le_bytes());
        let mut constraints = Vec::new();
        for terms in [
            vec![(3u32, 1u32), (0, 0), (3, 2)],
            vec![(4, 1)],
            vec![(1, 1)],
            vec![(2, 1)],
            vec![(0, 1)],
            vec![(2, 1)],
        ] {
            constraints.extend((terms.len() as u32).to_le_bytes());
            for (wire, coefficient) in terms {
                constraints.extend(wire.to_le_bytes());
                constraints.extend(coefficient.to_le_bytes());
            }
        }
        let labels = (0..5u64).flat_map(u64::to_le_bytes).collect();
        vec![(1, header), (2, constraints), (3, labels)]
    }

    fn encode(sections: &[(u32, Vec<u8>)]) -> Vec<u8> {
        let mut bytes = b"r1cs".to_vec();
        bytes.extend(1u32.to_le_bytes());
        bytes.extend((sections.len() as u32).to_le_bytes());
        for (id, contents) in sections {
            bytes.extend(id.to_le_bytes());
            bytes.extend((contents.len() as u64).to_le_bytes());
            bytes.extend(contents);
        }
        bytes
    }

    fn entry(row: usize, col: usize, coefficient: u32) -> SparseMatEntry<F> {
        SparseMatEntry {
            row,
            col,
            val: F::from_u32(coefficient),
        }
    }

    fn assert_both_error(bytes: &[u8], expected: CircomAdapterError) {
        let file = TemporaryFile::new(bytes);
        assert_eq!(import_r1cs_bytes(bytes), Err(expected.clone()));
        assert_eq!(import_r1cs_path(file.path()), Err(expected));
    }

    #[test]
    fn path_and_bytes_preserve_sections_entries_and_public_metadata() {
        let original = fixture_sections();
        let expected = import_r1cs_bytes(&encode(&original)).unwrap();
        assert_eq!(expected.public_outputs, 1);
        assert_eq!(expected.public_inputs, 1);
        assert_eq!(
            (
                expected.shape.num_cons,
                expected.shape.num_vars,
                expected.shape.num_io
            ),
            (2, 2, 2)
        );
        assert_eq!(
            expected.shape.a.entries,
            vec![
                entry(0, 0, 1),
                entry(0, 2, 0),
                entry(0, 0, 2),
                entry(1, 4, 1)
            ]
        );
        assert_eq!(
            expected.shape.b.entries,
            vec![entry(0, 1, 1), entry(1, 2, 1)]
        );
        assert_eq!(
            expected.shape.c.entries,
            vec![entry(0, 3, 1), entry(1, 4, 1)]
        );
        expected.shape.validate().unwrap();

        for order in [
            [0, 1, 2],
            [0, 2, 1],
            [1, 0, 2],
            [1, 2, 0],
            [2, 0, 1],
            [2, 1, 0],
        ] {
            let sections: Vec<_> = order.iter().map(|&i| original[i].clone()).collect();
            let bytes = encode(&sections);
            let file = TemporaryFile::new(&bytes);
            let from_bytes = import_r1cs_bytes(&bytes).unwrap();
            let from_path = import_r1cs_path(file.path()).unwrap();
            assert_eq!(from_bytes, expected, "order {order:?}");
            assert_eq!(from_path, expected, "order {order:?}");
            validate_satisfaction(
                &from_path.shape,
                &R1csWitness {
                    w: vec![F::ONE, F::from_u32(2)],
                },
                &[F::from_u32(6), F::from_u32(5)],
            )
            .unwrap();
        }
    }

    #[test]
    fn pair_import_uses_the_same_layout_for_paths_and_bytes() {
        let relation = encode(&fixture_sections());
        let mut witness = b"wtns".to_vec();
        witness.extend(2u32.to_le_bytes());
        witness.extend(2u32.to_le_bytes());
        let mut header = Vec::new();
        for value in [4u32, KOALABEAR_MODULUS, 5] {
            header.extend(value.to_le_bytes());
        }
        let values: Vec<_> = [1u32, 6, 5, 1, 2]
            .into_iter()
            .flat_map(u32::to_le_bytes)
            .collect();
        for (id, bytes) in [(1u32, header), (2, values)] {
            witness.extend(id.to_le_bytes());
            witness.extend((bytes.len() as u64).to_le_bytes());
            witness.extend(bytes);
        }
        let relation_file = TemporaryFile::new(&relation);
        let witness_file = TemporaryFile::new(&witness);
        let expected = import_bytes(&relation, &witness).unwrap();
        assert_eq!(
            import_paths(relation_file.path(), witness_file.path()).unwrap(),
            expected
        );
        assert_eq!(expected.1.w, vec![F::ONE, F::from_u32(2)]);
        assert_eq!(expected.2, vec![F::from_u32(6), F::from_u32(5)]);
    }

    #[test]
    fn preserves_optional_label_maps_and_ignored_trailing_bytes() {
        let sections = fixture_sections();
        let expected = import_r1cs_bytes(&encode(&sections)).unwrap();
        let mut cases = vec![sections[..2].to_vec()];
        let mut empty_map = sections.clone();
        empty_map[2].1.clear();
        cases.push(empty_map);
        let mut arbitrary_map = sections.clone();
        arbitrary_map[2].1 = vec![7, 8, 9];
        cases.push(arbitrary_map);
        let mut trailing = sections;
        trailing[0].1.extend([0xaa, 0xbb]);
        trailing[1].1.extend([0xcc, 0xdd]);
        cases.push(trailing);
        for sections in cases {
            let mut bytes = encode(&sections);
            bytes.extend([0xee, 0xff]);
            let file = TemporaryFile::new(&bytes);
            assert_eq!(import_r1cs_bytes(&bytes).unwrap(), expected);
            assert_eq!(import_r1cs_path(file.path()).unwrap(), expected);
        }
    }

    #[test]
    fn streamed_coefficients_keep_canonical_boundaries_and_error_precedence() {
        let mut sections = fixture_sections();
        sections[1].1[8..12].copy_from_slice(&(KOALABEAR_MODULUS - 1).to_le_bytes());
        let bytes = encode(&sections);
        let file = TemporaryFile::new(&bytes);
        let imported = import_r1cs_bytes(&bytes).unwrap();
        assert_eq!(
            imported.shape.a.entries[0].val,
            F::from_u32(KOALABEAR_MODULUS - 1)
        );
        assert_eq!(import_r1cs_path(file.path()).unwrap(), imported);

        // Wire remapping follows parsing every coefficient in the relation.
        // A later malformed coefficient retains precedence over an invalid wire.
        sections[1].1[4..8].copy_from_slice(&5u32.to_le_bytes());
        sections[1].1[24..28].copy_from_slice(&KOALABEAR_MODULUS.to_le_bytes());
        assert_both_error(&encode(&sections), CircomAdapterError::InvalidFieldElement);
        sections[1].1[24..28].copy_from_slice(&2u32.to_le_bytes());
        assert_both_error(&encode(&sections), CircomAdapterError::InvalidShape);
    }

    #[test]
    fn every_truncated_relation_prefix_retains_unexpected_eof() {
        let bytes = encode(&fixture_sections());
        for length in 0..bytes.len() {
            assert_both_error(&bytes[..length], CircomAdapterError::UnexpectedEof);
        }
    }

    #[test]
    fn malformed_sections_keep_existing_error_variants() {
        let sections = fixture_sections();
        for id in [1u32, 2] {
            let missing: Vec<_> = sections
                .iter()
                .filter(|(kind, _)| *kind != id)
                .cloned()
                .collect();
            assert_both_error(&encode(&missing), CircomAdapterError::MissingSection(id));
        }
        for id in [1u32, 2, 3] {
            let mut duplicate = sections.clone();
            duplicate.push(sections[(id - 1) as usize].clone());
            assert_both_error(
                &encode(&duplicate),
                CircomAdapterError::DuplicateSection(id),
            );
        }
        for id in [4u32, 5, 9] {
            let mut unknown = sections.clone();
            unknown.push((id, Vec::new()));
            assert_both_error(
                &encode(&unknown),
                CircomAdapterError::UnsupportedSection { id },
            );
        }
        let mut bad_magic = encode(&sections);
        bad_magic[0] ^= 1;
        assert_both_error(&bad_magic, CircomAdapterError::InvalidMagic("r1cs"));
        let mut version = encode(&sections);
        version[4..8].copy_from_slice(&2u32.to_le_bytes());
        assert_both_error(
            &version,
            CircomAdapterError::UnsupportedVersion {
                format: "r1cs",
                version: 2,
            },
        );
        let mut too_many = b"r1cs".to_vec();
        too_many.extend(1u32.to_le_bytes());
        too_many.extend(((MAX_CIRCOM_SECTIONS + 1) as u32).to_le_bytes());
        assert_both_error(
            &too_many,
            CircomAdapterError::ResourceLimitExceeded {
                resource: "sections",
                limit: MAX_CIRCOM_SECTIONS as u64,
                actual: (MAX_CIRCOM_SECTIONS + 1) as u64,
            },
        );
        let mut oversized_section = b"r1cs".to_vec();
        oversized_section.extend(1u32.to_le_bytes());
        oversized_section.extend(1u32.to_le_bytes());
        oversized_section.extend(1u32.to_le_bytes());
        oversized_section.extend((MAX_CIRCOM_FILE_BYTES as u64 + 1).to_le_bytes());
        assert_both_error(
            &oversized_section,
            CircomAdapterError::ResourceLimitExceeded {
                resource: "section bytes",
                limit: MAX_CIRCOM_FILE_BYTES as u64,
                actual: MAX_CIRCOM_FILE_BYTES as u64 + 1,
            },
        );
    }

    #[test]
    fn header_bounds_and_layout_errors_match_for_paths_and_bytes() {
        for (offset, resource) in [
            (8, "wires"),
            (12, "public outputs"),
            (16, "public inputs"),
            (20, "private inputs"),
            (32, "constraints"),
        ] {
            let mut sections = fixture_sections();
            sections[0].1[offset..offset + 4]
                .copy_from_slice(&((MAX_CIRCOM_DIMENSION + 1) as u32).to_le_bytes());
            assert_both_error(
                &encode(&sections),
                CircomAdapterError::ResourceLimitExceeded {
                    resource,
                    limit: MAX_CIRCOM_DIMENSION as u64,
                    actual: (MAX_CIRCOM_DIMENSION + 1) as u64,
                },
            );
        }
        let mut labels = fixture_sections();
        labels[0].1[24..32].copy_from_slice(&(MAX_CIRCOM_DIMENSION as u64 + 1).to_le_bytes());
        assert_both_error(
            &encode(&labels),
            CircomAdapterError::ResourceLimitExceeded {
                resource: "labels",
                limit: MAX_CIRCOM_DIMENSION as u64,
                actual: MAX_CIRCOM_DIMENSION as u64 + 1,
            },
        );
        let mut width = fixture_sections();
        width[0].1[..4].copy_from_slice(&8u32.to_le_bytes());
        assert_both_error(
            &encode(&width),
            CircomAdapterError::InvalidFieldSize {
                expected: 4,
                actual: 8,
            },
        );
        let mut modulus = fixture_sections();
        modulus[0].1[4..8].copy_from_slice(&(KOALABEAR_MODULUS - 1).to_le_bytes());
        assert_both_error(
            &encode(&modulus),
            CircomAdapterError::InvalidModulus {
                expected: KOALABEAR_MODULUS,
                actual: (KOALABEAR_MODULUS - 1).to_le_bytes().to_vec(),
            },
        );
        for (offset, value) in [(8, 0u32), (12, 5), (20, 3)] {
            let mut sections = fixture_sections();
            sections[0].1[offset..offset + 4].copy_from_slice(&value.to_le_bytes());
            assert_both_error(&encode(&sections), CircomAdapterError::InvalidShape);
        }
    }

    #[test]
    fn rejects_oversized_lc_through_the_public_importer_before_allocation() {
        let mut sections = fixture_sections();
        sections[1].1[..4].copy_from_slice(&((MAX_CIRCOM_TERMS + 1) as u32).to_le_bytes());
        assert_both_error(
            &encode(&sections),
            CircomAdapterError::ResourceLimitExceeded {
                resource: "linear-combination terms",
                limit: MAX_CIRCOM_TERMS as u64,
                actual: (MAX_CIRCOM_TERMS + 1) as u64,
            },
        );
    }

    #[test]
    fn aggregate_term_limit_is_inclusive_and_rejects_arithmetic_overflow() {
        let third = MAX_CIRCOM_TERMS / 3;
        let mut total = 0;
        for count in [third, third, MAX_CIRCOM_TERMS - 2 * third, 0] {
            add_r1cs_terms(&mut total, count).unwrap();
        }
        assert_eq!(total, MAX_CIRCOM_TERMS);
        assert_eq!(
            add_r1cs_terms(&mut total, 1),
            Err(CircomAdapterError::ResourceLimitExceeded {
                resource: "linear-combination terms",
                limit: MAX_CIRCOM_TERMS as u64,
                actual: MAX_CIRCOM_TERMS as u64 + 1
            })
        );
        let mut overflow = usize::MAX;
        assert_eq!(
            add_r1cs_terms(&mut overflow, 1),
            Err(CircomAdapterError::ResourceLimitExceeded {
                resource: "linear-combination terms",
                limit: MAX_CIRCOM_TERMS as u64,
                actual: u64::MAX
            })
        );
    }

    // Only the header and three LC counts occupy memory. Missing payload bytes
    // read as zero. Reading beyond the final count fails before matrix reserve,
    // so a broken aggregate limit cannot allocate millions of sparse entries.
    struct SparseInput {
        position: u64,
        length: u64,
        fail_at: u64,
        blocks: Vec<(u64, Vec<u8>)>,
        read_past_count: bool,
    }

    impl Read for SparseInput {
        fn read(&mut self, out: &mut [u8]) -> io::Result<usize> {
            if out.is_empty() {
                return Ok(0);
            }
            if self.position >= self.fail_at {
                self.read_past_count = true;
                return Err(io::Error::other("read after final LC count"));
            }
            let count = (out.len() as u64)
                .min(self.length - self.position)
                .min(self.fail_at - self.position) as usize;
            out[..count].fill(0);
            for (offset, bytes) in &self.blocks {
                let begin = self.position.max(*offset);
                let end = (self.position + count as u64).min(*offset + bytes.len() as u64);
                if begin < end {
                    out[(begin - self.position) as usize..(end - self.position) as usize]
                        .copy_from_slice(
                            &bytes[(begin - offset) as usize..(end - offset) as usize],
                        );
                }
            }
            self.position += count as u64;
            Ok(count)
        }
    }

    impl Seek for SparseInput {
        fn seek(&mut self, from: SeekFrom) -> io::Result<u64> {
            let position = match from {
                SeekFrom::Start(position) => i128::from(position),
                SeekFrom::End(offset) => i128::from(self.length) + i128::from(offset),
                SeekFrom::Current(offset) => i128::from(self.position) + i128::from(offset),
            };
            self.position = u64::try_from(position)
                .map_err(|_| io::Error::new(io::ErrorKind::InvalidInput, "invalid seek"))?;
            Ok(self.position)
        }
    }

    #[test]
    fn production_reader_enforces_the_total_term_cap_across_all_three_matrices() {
        let mut header = fixture_sections()[0].1.clone();
        header[32..36].copy_from_slice(&1u32.to_le_bytes());
        let mut prefix = encode(&[(1, header)]);
        prefix[8..12].copy_from_slice(&2u32.to_le_bytes());
        prefix.extend(2u32.to_le_bytes());
        prefix.extend((12 + 8 * (MAX_CIRCOM_TERMS as u64 + 1)).to_le_bytes());
        let mut position = prefix.len() as u64;
        let mut blocks = vec![(0, prefix)];
        let counts = [MAX_CIRCOM_TERMS / 2, MAX_CIRCOM_TERMS / 2, 1];
        let mut fail_at = 0;
        for count in counts {
            blocks.push((position, (count as u32).to_le_bytes().to_vec()));
            fail_at = position + 4;
            position += 4 + 8 * count as u64;
        }
        let input = SparseInput {
            position: 0,
            length: position,
            fail_at,
            blocks,
            read_past_count: false,
        };
        let mut reader = BufReader::with_capacity(1 << 20, input);
        assert_eq!(
            import_r1cs_reader(&mut reader, position),
            Err(CircomAdapterError::ResourceLimitExceeded {
                resource: "linear-combination terms",
                limit: MAX_CIRCOM_TERMS as u64,
                actual: MAX_CIRCOM_TERMS as u64 + 1
            })
        );
        assert!(!reader.get_ref().read_past_count);
    }

    #[test]
    fn oversized_path_is_rejected_from_metadata_before_reading_its_payload() {
        let file = TemporaryFile::new(&[]);
        OpenOptions::new()
            .write(true)
            .open(file.path())
            .unwrap()
            .set_len(MAX_CIRCOM_FILE_BYTES as u64 + 1)
            .unwrap();
        assert_eq!(
            import_r1cs_path(file.path()),
            Err(CircomAdapterError::ResourceLimitExceeded {
                resource: "R1CS file",
                limit: MAX_CIRCOM_FILE_BYTES as u64,
                actual: MAX_CIRCOM_FILE_BYTES as u64 + 1
            })
        );
    }
}

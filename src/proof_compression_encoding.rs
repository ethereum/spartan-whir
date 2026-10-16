//! Experimental binary encoding for comparing proof representation sizes.
//!
//! Modes are supplied by the enclosing proof format. Integers are little endian;
//! packed `u32` values occupy 31 bits, and compact lengths and `u64` metadata use
//! canonical unsigned LEB128. Enum discriminants remain metadata. This codec
//! does not validate field membership or cryptographic proof semantics.

use alloc::{format, string::String, vec::Vec};
use core::fmt;
use serde::{de, ser, Deserialize, Serialize};

const MAX_BYTES: usize = 64 * 1024 * 1024;
const MAX_SEQUENCE_ELEMENTS: usize = 16 * 1024 * 1024;
const MAX_DEPTH: usize = 128;

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct CodecError(String);

impl CodecError {
    fn message(message: &str) -> Self {
        Self(String::from(message))
    }
}

impl fmt::Display for CodecError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(&self.0)
    }
}

impl core::error::Error for CodecError {}

impl ser::Error for CodecError {
    fn custom<T: fmt::Display>(message: T) -> Self {
        Self(format!("{message}"))
    }
}

impl de::Error for CodecError {
    fn custom<T: fmt::Display>(message: T) -> Self {
        Self(format!("{message}"))
    }
}

#[cfg(test)]
pub(crate) fn encode<T: Serialize>(
    value: &T,
    packed_fields: bool,
    compact_integers: bool,
) -> Result<Vec<u8>, CodecError> {
    encode_refined(value, packed_fields, compact_integers, false)
}

pub(crate) fn encode_refined<T: Serialize>(
    value: &T,
    packed_fields: bool,
    compact_integers: bool,
    refined_metadata: bool,
) -> Result<Vec<u8>, CodecError> {
    Ok(encode_state(value, packed_fields, compact_integers, refined_metadata)?.bytes)
}

fn encode_state<T: Serialize>(
    value: &T,
    packed_fields: bool,
    compact_integers: bool,
    refined_metadata: bool,
) -> Result<Encoder, CodecError> {
    let mut encoder = Encoder {
        bytes: Vec::new(),
        position: 0,
        packed_fields,
        compact_integers,
        refined_metadata,
        depth: 0,
    };
    value.serialize(&mut encoder)?;
    Ok(encoder)
}

/// Untimed statistics; multiple serializations keep counters out of hot loops.
#[derive(Debug, Clone, Serialize)]
pub(crate) struct EncodingStatistics {
    pub field_elements: usize,
    pub packed_field_bits: usize,
    pub other_bits: usize,
    pub refined_other_bits: usize,
}

pub(crate) fn encoding_statistics<T: Serialize>(
    value: &T,
) -> Result<EncodingStatistics, CodecError> {
    let original_bits = encode_state(value, false, true, false)?.position;
    let packed_bits = encode_state(value, true, true, false)?.position;
    let refined_bits = encode_state(value, true, true, true)?.position;
    let field_elements = original_bits - packed_bits;
    let packed_field_bits = field_elements * 31;
    Ok(EncodingStatistics {
        field_elements,
        packed_field_bits,
        other_bits: packed_bits - packed_field_bits,
        refined_other_bits: refined_bits - packed_field_bits,
    })
}

#[cfg(test)]
pub(crate) fn decode<'de, T: Deserialize<'de>>(
    bytes: &'de [u8],
    packed_fields: bool,
    compact_integers: bool,
) -> Result<T, CodecError> {
    decode_refined(bytes, packed_fields, compact_integers, false)
}

pub(crate) fn decode_refined<'de, T: Deserialize<'de>>(
    bytes: &'de [u8],
    packed_fields: bool,
    compact_integers: bool,
    refined_metadata: bool,
) -> Result<T, CodecError> {
    if bytes.len() > MAX_BYTES {
        return Err(CodecError::message("input exceeds codec limit"));
    }
    let mut decoder = Decoder {
        bytes,
        position: 0,
        packed_fields,
        compact_integers,
        refined_metadata,
        depth: 0,
        // This also bounds sequences whose elements consume no input bytes.
        element_budget: bytes
            .len()
            .saturating_mul(8)
            .saturating_add(1024)
            .min(MAX_SEQUENCE_ELEMENTS),
    };
    let value = T::deserialize(&mut decoder)?;
    let remaining = bytes.len() * 8 - decoder.position;
    if remaining >= 8 || decoder.read_bits(remaining)? != 0 {
        return Err(CodecError::message("trailing data or nonzero padding"));
    }
    Ok(value)
}

struct Encoder {
    bytes: Vec<u8>,
    position: usize,
    packed_fields: bool,
    compact_integers: bool,
    refined_metadata: bool,
    depth: usize,
}

impl Encoder {
    fn write_bits(&mut self, mut value: u128, mut width: usize) -> Result<(), CodecError> {
        if self
            .position
            .checked_add(width)
            .is_none_or(|end| end > MAX_BYTES * 8)
        {
            return Err(CodecError::message("encoded output exceeds codec limit"));
        }
        while width != 0 {
            let offset = self.position % 8;
            let take = width.min(8 - offset);
            if offset == 0 {
                self.bytes.push(0);
            }
            let mask = (1u128 << take) - 1;
            self.bytes[self.position / 8] |= ((value & mask) as u8) << offset;
            value >>= take;
            width -= take;
            self.position += take;
        }
        Ok(())
    }

    fn integer(&mut self, value: u64, fixed_width: usize) -> Result<(), CodecError> {
        if !self.compact_integers {
            return self.write_bits(u128::from(value), fixed_width);
        }
        self.varint(value)
    }

    fn varint(&mut self, mut value: u64) -> Result<(), CodecError> {
        loop {
            let mut byte = (value & 127) as u8;
            value >>= 7;
            if value != 0 {
                byte |= 128;
            }
            self.write_bits(u128::from(byte), 8)?;
            if value == 0 {
                return Ok(());
            }
        }
    }

    fn compound(&mut self, remaining: usize) -> Result<Compound<'_>, CodecError> {
        if remaining > MAX_SEQUENCE_ELEMENTS || self.depth == MAX_DEPTH {
            return Err(CodecError::message("container exceeds codec limit"));
        }
        self.depth += 1;
        Ok(Compound {
            encoder: self,
            remaining,
        })
    }

    fn nested<T: ?Sized + Serialize>(&mut self, value: &T) -> Result<(), CodecError> {
        if self.depth == MAX_DEPTH {
            return Err(CodecError::message("nesting exceeds codec limit"));
        }
        self.depth += 1;
        let result = value.serialize(&mut *self);
        self.depth -= 1;
        result
    }
}

struct Compound<'a> {
    encoder: &'a mut Encoder,
    remaining: usize,
}

impl Compound<'_> {
    fn element<T: ?Sized + Serialize>(&mut self, value: &T) -> Result<(), CodecError> {
        if self.remaining == 0 {
            return Err(CodecError::message("too many container elements"));
        }
        self.remaining -= 1;
        value.serialize(&mut *self.encoder)
    }

    fn finish(self) -> Result<(), CodecError> {
        self.encoder.depth -= 1;
        if self.remaining != 0 {
            return Err(CodecError::message("missing container elements"));
        }
        Ok(())
    }
}

macro_rules! sequence_serializer {
    ($trait_name:ident, $method:ident) => {
        impl ser::$trait_name for Compound<'_> {
            type Ok = ();
            type Error = CodecError;
            fn $method<T: ?Sized + Serialize>(&mut self, value: &T) -> Result<(), CodecError> {
                self.element(value)
            }
            fn end(self) -> Result<(), CodecError> {
                self.finish()
            }
        }
    };
}

sequence_serializer!(SerializeSeq, serialize_element);
sequence_serializer!(SerializeTuple, serialize_element);
sequence_serializer!(SerializeTupleStruct, serialize_field);
sequence_serializer!(SerializeTupleVariant, serialize_field);

macro_rules! struct_serializer {
    ($trait_name:ident) => {
        impl ser::$trait_name for Compound<'_> {
            type Ok = ();
            type Error = CodecError;
            fn serialize_field<T: ?Sized + Serialize>(
                &mut self,
                _key: &'static str,
                value: &T,
            ) -> Result<(), CodecError> {
                self.element(value)
            }
            fn end(self) -> Result<(), CodecError> {
                self.finish()
            }
        }
    };
}

struct_serializer!(SerializeStruct);
struct_serializer!(SerializeStructVariant);

macro_rules! fixed_serializer {
    ($method:ident, $ty:ty, $width:expr) => {
        fn $method(self, value: $ty) -> Result<(), CodecError> {
            self.write_bits(value as u128, $width)
        }
    };
}

impl<'a> ser::Serializer for &'a mut Encoder {
    type Ok = ();
    type Error = CodecError;
    type SerializeSeq = Compound<'a>;
    type SerializeTuple = Compound<'a>;
    type SerializeTupleStruct = Compound<'a>;
    type SerializeTupleVariant = Compound<'a>;
    type SerializeMap = ser::Impossible<(), CodecError>;
    type SerializeStruct = Compound<'a>;
    type SerializeStructVariant = Compound<'a>;

    fn serialize_bool(self, value: bool) -> Result<(), CodecError> {
        self.write_bits(u128::from(value), if self.refined_metadata { 1 } else { 8 })
    }
    fixed_serializer!(serialize_i8, i8, 8);
    fixed_serializer!(serialize_i16, i16, 16);
    fixed_serializer!(serialize_i32, i32, 32);
    fixed_serializer!(serialize_i64, i64, 64);
    fixed_serializer!(serialize_i128, i128, 128);
    fixed_serializer!(serialize_u8, u8, 8);
    fn serialize_u16(self, value: u16) -> Result<(), CodecError> {
        if self.refined_metadata {
            self.varint(u64::from(value))
        } else {
            self.write_bits(u128::from(value), 16)
        }
    }
    fixed_serializer!(serialize_u128, u128, 128);

    fn serialize_u32(self, value: u32) -> Result<(), CodecError> {
        if self.packed_fields && value >= (1 << 31) {
            return Err(CodecError::message("u32 does not fit in 31 bits"));
        }
        self.write_bits(u128::from(value), if self.packed_fields { 31 } else { 32 })
    }
    fn serialize_u64(self, value: u64) -> Result<(), CodecError> {
        self.integer(value, 64)
    }
    fn serialize_f32(self, _: f32) -> Result<(), CodecError> {
        Err(unsupported())
    }
    fn serialize_f64(self, _: f64) -> Result<(), CodecError> {
        Err(unsupported())
    }
    fn serialize_char(self, _: char) -> Result<(), CodecError> {
        Err(unsupported())
    }
    fn serialize_str(self, _: &str) -> Result<(), CodecError> {
        Err(unsupported())
    }
    fn serialize_bytes(self, _: &[u8]) -> Result<(), CodecError> {
        Err(unsupported())
    }
    fn serialize_none(self) -> Result<(), CodecError> {
        self.write_bits(0, if self.refined_metadata { 1 } else { 8 })
    }
    fn serialize_some<T: ?Sized + Serialize>(self, value: &T) -> Result<(), CodecError> {
        self.write_bits(1, if self.refined_metadata { 1 } else { 8 })?;
        self.nested(value)
    }
    fn serialize_unit(self) -> Result<(), CodecError> {
        Ok(())
    }
    fn serialize_unit_struct(self, _: &'static str) -> Result<(), CodecError> {
        Ok(())
    }
    fn serialize_unit_variant(
        self,
        _: &'static str,
        index: u32,
        _: &'static str,
    ) -> Result<(), CodecError> {
        self.integer(u64::from(index), 32)
    }
    fn serialize_newtype_struct<T: ?Sized + Serialize>(
        self,
        _: &'static str,
        value: &T,
    ) -> Result<(), CodecError> {
        self.nested(value)
    }
    fn serialize_newtype_variant<T: ?Sized + Serialize>(
        self,
        _: &'static str,
        index: u32,
        _: &'static str,
        value: &T,
    ) -> Result<(), CodecError> {
        self.integer(u64::from(index), 32)?;
        self.nested(value)
    }
    fn serialize_seq(self, length: Option<usize>) -> Result<Self::SerializeSeq, CodecError> {
        let length = length.ok_or_else(|| CodecError::message("sequence length is required"))?;
        self.integer(u64::try_from(length).map_err(|_| unsupported())?, 64)?;
        self.compound(length)
    }
    fn serialize_tuple(self, length: usize) -> Result<Self::SerializeTuple, CodecError> {
        self.compound(length)
    }
    fn serialize_tuple_struct(
        self,
        _: &'static str,
        length: usize,
    ) -> Result<Self::SerializeTupleStruct, CodecError> {
        self.compound(length)
    }
    fn serialize_tuple_variant(
        self,
        _: &'static str,
        index: u32,
        _: &'static str,
        length: usize,
    ) -> Result<Self::SerializeTupleVariant, CodecError> {
        self.integer(u64::from(index), 32)?;
        self.compound(length)
    }
    fn serialize_map(self, _: Option<usize>) -> Result<Self::SerializeMap, CodecError> {
        Err(unsupported())
    }
    fn serialize_struct(
        self,
        _: &'static str,
        length: usize,
    ) -> Result<Self::SerializeStruct, CodecError> {
        self.compound(length)
    }
    fn serialize_struct_variant(
        self,
        _: &'static str,
        index: u32,
        _: &'static str,
        length: usize,
    ) -> Result<Self::SerializeStructVariant, CodecError> {
        self.integer(u64::from(index), 32)?;
        self.compound(length)
    }
    fn is_human_readable(&self) -> bool {
        false
    }
}

fn unsupported() -> CodecError {
    CodecError::message("unsupported serde data type")
}

struct Decoder<'de> {
    bytes: &'de [u8],
    position: usize,
    packed_fields: bool,
    compact_integers: bool,
    refined_metadata: bool,
    depth: usize,
    element_budget: usize,
}

impl<'de> Decoder<'de> {
    fn read_bits(&mut self, mut width: usize) -> Result<u128, CodecError> {
        if self
            .position
            .checked_add(width)
            .is_none_or(|end| end > self.bytes.len() * 8)
        {
            return Err(CodecError::message("truncated input"));
        }
        let mut value = 0u128;
        let mut shift = 0;
        while width != 0 {
            let offset = self.position % 8;
            let take = width.min(8 - offset);
            let mask = (1u16 << take) - 1;
            value |=
                u128::from((u16::from(self.bytes[self.position / 8]) >> offset) & mask) << shift;
            shift += take;
            self.position += take;
            width -= take;
        }
        Ok(value)
    }

    fn integer(&mut self, fixed_width: usize) -> Result<u64, CodecError> {
        if !self.compact_integers {
            return Ok(self.read_bits(fixed_width)? as u64);
        }
        self.varint(fixed_width)
    }

    fn varint(&mut self, fixed_width: usize) -> Result<u64, CodecError> {
        let mut value = 0u64;
        for index in 0..10 {
            let byte = self.read_bits(8)? as u8;
            let payload = byte & 127;
            if index == 9 && (payload > 1 || byte & 128 != 0) {
                return Err(CodecError::message("overflowing varint"));
            }
            value |= u64::from(payload) << (7 * index);
            if byte & 128 == 0 {
                if index != 0 && payload == 0 {
                    return Err(CodecError::message("noncanonical varint"));
                }
                if fixed_width < 64 && value >= (1u64 << fixed_width) {
                    return Err(CodecError::message("varint overflows target integer"));
                }
                return Ok(value);
            }
        }
        Err(CodecError::message("overlong varint"))
    }

    fn nested<T>(
        &mut self,
        visit: impl FnOnce(&mut Self) -> Result<T, CodecError>,
    ) -> Result<T, CodecError> {
        if self.depth == MAX_DEPTH {
            return Err(CodecError::message("nesting exceeds codec limit"));
        }
        self.depth += 1;
        let result = visit(self);
        self.depth -= 1;
        result
    }

    fn sequence<V: de::Visitor<'de>>(
        &mut self,
        length: usize,
        visitor: V,
    ) -> Result<V::Value, CodecError> {
        if length > MAX_SEQUENCE_ELEMENTS || length > self.element_budget {
            return Err(CodecError::message("sequence exceeds codec limit"));
        }
        self.element_budget -= length;
        self.nested(|decoder| {
            let mut access = Sequence {
                decoder,
                remaining: length,
            };
            let result = visitor.visit_seq(&mut access)?;
            if access.remaining != 0 {
                return Err(CodecError::message("unconsumed sequence elements"));
            }
            Ok(result)
        })
    }
}

struct Sequence<'a, 'de> {
    decoder: &'a mut Decoder<'de>,
    remaining: usize,
}

impl<'de> de::SeqAccess<'de> for &mut Sequence<'_, 'de> {
    type Error = CodecError;
    fn next_element_seed<T: de::DeserializeSeed<'de>>(
        &mut self,
        seed: T,
    ) -> Result<Option<T::Value>, CodecError> {
        if self.remaining == 0 {
            return Ok(None);
        }
        self.remaining -= 1;
        seed.deserialize(&mut *self.decoder).map(Some)
    }
    // Avoid reserving attacker-selected capacities before decoding elements.
    fn size_hint(&self) -> Option<usize> {
        None
    }
}

struct Enum<'a, 'de> {
    decoder: &'a mut Decoder<'de>,
    index: u32,
}

impl<'a, 'de> de::EnumAccess<'de> for Enum<'a, 'de> {
    type Error = CodecError;
    type Variant = Self;
    fn variant_seed<V: de::DeserializeSeed<'de>>(
        self,
        seed: V,
    ) -> Result<(V::Value, Self::Variant), CodecError> {
        let index = seed.deserialize(de::value::U32Deserializer::<CodecError>::new(self.index))?;
        Ok((index, self))
    }
}

impl<'de> de::VariantAccess<'de> for Enum<'_, 'de> {
    type Error = CodecError;
    fn unit_variant(self) -> Result<(), CodecError> {
        Ok(())
    }
    fn newtype_variant_seed<T: de::DeserializeSeed<'de>>(
        self,
        seed: T,
    ) -> Result<T::Value, CodecError> {
        self.decoder.nested(|decoder| seed.deserialize(decoder))
    }
    fn tuple_variant<V: de::Visitor<'de>>(
        self,
        length: usize,
        visitor: V,
    ) -> Result<V::Value, CodecError> {
        self.decoder.sequence(length, visitor)
    }
    fn struct_variant<V: de::Visitor<'de>>(
        self,
        fields: &'static [&'static str],
        visitor: V,
    ) -> Result<V::Value, CodecError> {
        self.decoder.sequence(fields.len(), visitor)
    }
}

macro_rules! fixed_deserializer {
    ($method:ident, $visit:ident, $ty:ty, $width:expr) => {
        fn $method<V: de::Visitor<'de>>(self, visitor: V) -> Result<V::Value, CodecError> {
            visitor.$visit(self.read_bits($width)? as $ty)
        }
    };
}

impl<'de> de::Deserializer<'de> for &mut Decoder<'de> {
    type Error = CodecError;
    fn deserialize_any<V: de::Visitor<'de>>(self, _: V) -> Result<V::Value, CodecError> {
        Err(unsupported())
    }
    fn deserialize_bool<V: de::Visitor<'de>>(self, visitor: V) -> Result<V::Value, CodecError> {
        match self.read_bits(if self.refined_metadata { 1 } else { 8 })? {
            0 => visitor.visit_bool(false),
            1 => visitor.visit_bool(true),
            _ => Err(CodecError::message("noncanonical boolean")),
        }
    }
    fixed_deserializer!(deserialize_i8, visit_i8, i8, 8);
    fixed_deserializer!(deserialize_i16, visit_i16, i16, 16);
    fixed_deserializer!(deserialize_i32, visit_i32, i32, 32);
    fixed_deserializer!(deserialize_i64, visit_i64, i64, 64);
    fixed_deserializer!(deserialize_i128, visit_i128, i128, 128);
    fixed_deserializer!(deserialize_u8, visit_u8, u8, 8);
    fn deserialize_u16<V: de::Visitor<'de>>(self, visitor: V) -> Result<V::Value, CodecError> {
        let value = if self.refined_metadata {
            self.varint(16)?
        } else {
            self.read_bits(16)? as u64
        };
        visitor.visit_u16(value as u16)
    }
    fixed_deserializer!(deserialize_u128, visit_u128, u128, 128);
    fn deserialize_u32<V: de::Visitor<'de>>(self, visitor: V) -> Result<V::Value, CodecError> {
        visitor.visit_u32(self.read_bits(if self.packed_fields { 31 } else { 32 })? as u32)
    }
    fn deserialize_u64<V: de::Visitor<'de>>(self, visitor: V) -> Result<V::Value, CodecError> {
        visitor.visit_u64(self.integer(64)?)
    }
    fn deserialize_f32<V: de::Visitor<'de>>(self, _: V) -> Result<V::Value, CodecError> {
        Err(unsupported())
    }
    fn deserialize_f64<V: de::Visitor<'de>>(self, _: V) -> Result<V::Value, CodecError> {
        Err(unsupported())
    }
    fn deserialize_char<V: de::Visitor<'de>>(self, _: V) -> Result<V::Value, CodecError> {
        Err(unsupported())
    }
    fn deserialize_str<V: de::Visitor<'de>>(self, _: V) -> Result<V::Value, CodecError> {
        Err(unsupported())
    }
    fn deserialize_string<V: de::Visitor<'de>>(self, _: V) -> Result<V::Value, CodecError> {
        Err(unsupported())
    }
    fn deserialize_bytes<V: de::Visitor<'de>>(self, _: V) -> Result<V::Value, CodecError> {
        Err(unsupported())
    }
    fn deserialize_byte_buf<V: de::Visitor<'de>>(self, _: V) -> Result<V::Value, CodecError> {
        Err(unsupported())
    }
    fn deserialize_option<V: de::Visitor<'de>>(self, visitor: V) -> Result<V::Value, CodecError> {
        match self.read_bits(if self.refined_metadata { 1 } else { 8 })? {
            0 => visitor.visit_none(),
            1 => self.nested(|decoder| visitor.visit_some(decoder)),
            _ => Err(CodecError::message("noncanonical option tag")),
        }
    }
    fn deserialize_unit<V: de::Visitor<'de>>(self, visitor: V) -> Result<V::Value, CodecError> {
        visitor.visit_unit()
    }
    fn deserialize_unit_struct<V: de::Visitor<'de>>(
        self,
        _: &'static str,
        visitor: V,
    ) -> Result<V::Value, CodecError> {
        visitor.visit_unit()
    }
    fn deserialize_newtype_struct<V: de::Visitor<'de>>(
        self,
        _: &'static str,
        visitor: V,
    ) -> Result<V::Value, CodecError> {
        self.nested(|decoder| visitor.visit_newtype_struct(decoder))
    }
    fn deserialize_seq<V: de::Visitor<'de>>(self, visitor: V) -> Result<V::Value, CodecError> {
        let length = usize::try_from(self.integer(64)?)
            .map_err(|_| CodecError::message("sequence length overflows usize"))?;
        self.sequence(length, visitor)
    }
    fn deserialize_tuple<V: de::Visitor<'de>>(
        self,
        length: usize,
        visitor: V,
    ) -> Result<V::Value, CodecError> {
        self.sequence(length, visitor)
    }
    fn deserialize_tuple_struct<V: de::Visitor<'de>>(
        self,
        _: &'static str,
        length: usize,
        visitor: V,
    ) -> Result<V::Value, CodecError> {
        self.sequence(length, visitor)
    }
    fn deserialize_map<V: de::Visitor<'de>>(self, _: V) -> Result<V::Value, CodecError> {
        Err(unsupported())
    }
    fn deserialize_struct<V: de::Visitor<'de>>(
        self,
        _: &'static str,
        fields: &'static [&'static str],
        visitor: V,
    ) -> Result<V::Value, CodecError> {
        self.sequence(fields.len(), visitor)
    }
    fn deserialize_enum<V: de::Visitor<'de>>(
        self,
        _: &'static str,
        _: &'static [&'static str],
        visitor: V,
    ) -> Result<V::Value, CodecError> {
        let index = self.integer(32)? as u32;
        visitor.visit_enum(Enum {
            decoder: self,
            index,
        })
    }
    fn deserialize_identifier<V: de::Visitor<'de>>(self, _: V) -> Result<V::Value, CodecError> {
        Err(unsupported())
    }
    fn deserialize_ignored_any<V: de::Visitor<'de>>(self, _: V) -> Result<V::Value, CodecError> {
        Err(unsupported())
    }
    fn is_human_readable(&self) -> bool {
        false
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::vec;

    #[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
    enum Opening {
        Empty,
        Scalar(u32),
        Pair(u64, [u32; 5]),
        Batch {
            rows: Vec<[u32; 5]>,
            marker: Option<bool>,
        },
    }

    #[derive(Debug, PartialEq, Eq, Serialize, Deserialize)]
    struct Proof {
        roots: [[u32; 8]; 2],
        rounds: Vec<Opening>,
        count: usize,
        nonce: u64,
        auxiliary: Option<Vec<u8>>,
    }

    fn sample() -> Proof {
        Proof {
            roots: [[0, 1, 127, 128, 2_130_706_432, 42, 7, 15]; 2],
            rounds: vec![
                Opening::Empty,
                Opening::Scalar(5),
                Opening::Pair(256, [9; 5]),
                Opening::Batch {
                    rows: vec![[17; 5]; 3],
                    marker: Some(true),
                },
            ],
            count: 300,
            nonce: u64::MAX,
            auxiliary: Some(vec![1, 2, 255]),
        }
    }

    #[test]
    fn nested_proofs_round_trip_in_every_mode() {
        let proof = sample();
        for packed in [false, true] {
            for compact in [false, true] {
                let bytes = encode(&proof, packed, compact).unwrap();
                assert_eq!(decode::<Proof>(&bytes, packed, compact).unwrap(), proof);
                assert_eq!(
                    encode(
                        &decode::<Proof>(&bytes, packed, compact).unwrap(),
                        packed,
                        compact
                    )
                    .unwrap(),
                    bytes
                );
            }
        }
        assert_eq!(
            encode(&proof, false, false).unwrap(),
            bincode::serialize(&proof).unwrap()
        );
    }

    #[test]
    fn field_packing_and_metadata_have_independent_sizes() {
        let fields = [0u32; 8];
        assert_eq!(encode(&fields, false, false).unwrap().len(), 32);
        assert_eq!(encode(&fields, true, false).unwrap().len(), 31);
        assert_eq!(encode(&vec![0u32; 8], false, true).unwrap().len(), 33);
        assert_eq!(encode(&vec![0u32; 8], true, true).unwrap().len(), 32);
        assert!(encode(&(1u32 << 31), true, false).is_err());
    }

    #[test]
    fn truncation_padding_and_trailing_bytes_reject() {
        for packed in [false, true] {
            for compact in [false, true] {
                let bytes = encode(&sample(), packed, compact).unwrap();
                for length in 0..bytes.len() {
                    assert!(decode::<Proof>(&bytes[..length], packed, compact).is_err());
                }
                let mut extended = bytes;
                extended.push(0);
                assert!(decode::<Proof>(&extended, packed, compact).is_err());
            }
        }
        let mut bytes = encode(&7u32, true, false).unwrap();
        bytes[3] |= 128;
        assert!(decode::<u32>(&bytes, true, false).is_err());
        assert!(decode::<()>(&[0], false, false).is_err());
    }

    #[test]
    fn malformed_metadata_rejects_before_sequence_allocation() {
        for bytes in [vec![128, 0], vec![129, 0], vec![255; 10], vec![128; 11]] {
            assert!(decode::<u64>(&bytes, false, true).is_err());
        }
        assert!(decode::<bool>(&[2], false, false).is_err());
        assert!(decode::<Option<u32>>(&[2], false, false).is_err());
        assert!(decode::<Vec<u32>>(&u64::MAX.to_le_bytes(), false, false).is_err());
        assert!(decode::<Vec<()>>(&u64::MAX.to_le_bytes(), false, false).is_err());
        let oversized = encode(&((MAX_SEQUENCE_ELEMENTS as u64) + 1), false, true).unwrap();
        assert!(decode::<Vec<()>>(&oversized, false, true).is_err());
        let bad_variant = encode(&(u64::from(u32::MAX) + 1), false, true).unwrap();
        assert!(decode::<Opening>(&bad_variant, false, true).is_err());
    }

    #[test]
    fn unsupported_types_reject() {
        assert!(encode(&1.0f64, false, false).is_err());
        assert!(decode::<f64>(&[0; 8], false, false).is_err());
        assert!(encode(&"unsupported", false, false).is_err());
    }

    #[test]
    fn refined_metadata_round_trips_and_rejects_malformed_values() {
        let value = (vec![[7u32; 5]; 2], true, Some(17u16), 300u64);
        for packed in [false, true] {
            for compact in [false, true] {
                let bytes = encode_refined(&value, packed, compact, true).unwrap();
                let decoded: (Vec<[u32; 5]>, bool, Option<u16>, u64) =
                    decode_refined(&bytes, packed, compact, true).unwrap();
                assert_eq!(decoded, value);
                assert_eq!(
                    encode_refined(&decoded, packed, compact, true).unwrap(),
                    bytes
                );
                for length in 0..bytes.len() {
                    assert!(decode_refined::<(Vec<[u32; 5]>, bool, Option<u16>, u64)>(
                        &bytes[..length],
                        packed,
                        compact,
                        true,
                    )
                    .is_err());
                }
            }
        }
        for value in [0u16, 127, 128, 16_383, 16_384, u16::MAX] {
            let bytes = encode_refined(&value, true, false, true).unwrap();
            assert_eq!(
                decode_refined::<u16>(&bytes, true, false, true).unwrap(),
                value
            );
        }
        assert!(decode_refined::<u16>(&[128, 128, 4], true, true, true).is_err());
        assert!(decode_refined::<u16>(&[128, 0], true, true, true).is_err());
        assert!(decode_refined::<bool>(&[2], true, true, true).is_err());
        let stats = encoding_statistics(&value).unwrap();
        assert_eq!(stats.field_elements, 10);
        assert_eq!(stats.packed_field_bits, 310);
        assert_eq!(stats.other_bits, 56);
        assert_eq!(stats.refined_other_bits, 34);
    }

    #[test]
    fn recursive_variants_respect_nesting_limit() {
        #[derive(Debug, PartialEq, Eq, Serialize, Deserialize)]
        enum Nested {
            End,
            Next(alloc::boxed::Box<Nested>),
        }
        let mut value = Nested::End;
        for _ in 0..MAX_DEPTH {
            value = Nested::Next(alloc::boxed::Box::new(value));
        }
        let bytes = encode(&value, true, true).unwrap();
        assert_eq!(decode::<Nested>(&bytes, true, true).unwrap(), value);
        value = Nested::Next(alloc::boxed::Box::new(value));
        assert!(encode(&value, true, true).is_err());
        let mut too_deep = vec![1; MAX_DEPTH + 1];
        too_deep.push(0);
        assert!(decode::<Nested>(&too_deep, true, true).is_err());
    }
}

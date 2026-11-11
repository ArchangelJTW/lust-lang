use alloc::{string::String, vec::Vec};
use core::fmt;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum CanonicalTypeTag {
    Bool,
    Int64,
    UInt64,
    Float64,
    Decimal128,
    ObjectId,
    DateTimeUtc,
    StringSlice,
    BinarySlice,
    NestedStruct(u64),
}

impl CanonicalTypeTag {
    pub fn byte_size(&self) -> usize {
        match self {
            CanonicalTypeTag::Bool => 1,
            CanonicalTypeTag::Int64 | CanonicalTypeTag::UInt64 | CanonicalTypeTag::Float64 => 8,
            CanonicalTypeTag::DateTimeUtc => 8,
            CanonicalTypeTag::ObjectId => 12,
            CanonicalTypeTag::Decimal128 => 16,
            CanonicalTypeTag::StringSlice | CanonicalTypeTag::BinarySlice => 24, // (ptr, len, OwnerId)
            CanonicalTypeTag::NestedStruct(_) => 0, // Determined by inner layout
        }
    }

    pub fn alignment(&self) -> usize {
        match self {
            CanonicalTypeTag::Bool => 1,
            CanonicalTypeTag::ObjectId => 4,
            CanonicalTypeTag::Int64 | CanonicalTypeTag::UInt64 | CanonicalTypeTag::Float64 => 8,
            CanonicalTypeTag::DateTimeUtc => 8,
            CanonicalTypeTag::Decimal128 => 16,
            CanonicalTypeTag::StringSlice | CanonicalTypeTag::BinarySlice => 8,
            CanonicalTypeTag::NestedStruct(_) => 8,
        }
    }

    pub fn is_managed_slice(&self) -> bool {
        matches!(self, CanonicalTypeTag::StringSlice | CanonicalTypeTag::BinarySlice)
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LogicalFieldDesc {
    pub ordinal: u16,
    pub name: String,
    pub tag: CanonicalTypeTag,
    pub byte_offset: usize,
    pub byte_size: usize,
    pub is_nullable: bool,
    pub is_optional: bool,
}

impl LogicalFieldDesc {
    pub fn new(ordinal: u16, name: impl Into<String>, tag: CanonicalTypeTag) -> Self {
        let size = tag.byte_size();
        Self {
            ordinal,
            name: name.into(),
            tag,
            byte_offset: 0,
            byte_size: size,
            is_nullable: false,
            is_optional: false,
        }
    }

    pub fn with_nullability(mut self, is_nullable: bool, is_optional: bool) -> Self {
        self.is_nullable = is_nullable;
        self.is_optional = is_optional;
        self
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TargetAbi {
    pub pointer_width: usize,
    pub max_alignment: usize,
}

impl TargetAbi {
    pub const fn x86_64() -> Self {
        Self {
            pointer_width: 8,
            max_alignment: 16,
        }
    }

    pub const fn aarch64() -> Self {
        Self {
            pointer_width: 8,
            max_alignment: 16,
        }
    }

    pub const fn rv32() -> Self {
        Self {
            pointer_width: 4,
            max_alignment: 8,
        }
    }
}

/// Canonical Cross-Target Logical Layout Specification (HF-04)
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LogicalLayoutV1 {
    pub type_id: u64,
    pub type_name: String,
    pub fields: Vec<LogicalFieldDesc>,
    pub size_bytes: usize,
    pub align_bytes: usize,
    pub is_c_flat: bool,
    pub target_abi: TargetAbi,
}

impl LogicalLayoutV1 {
    pub fn new(type_id: u64, type_name: impl Into<String>, fields: Vec<LogicalFieldDesc>) -> Self {
        Self::new_with_abi(type_id, type_name, fields, TargetAbi::x86_64())
    }

    pub fn new_with_abi(
        type_id: u64,
        type_name: impl Into<String>,
        mut fields: Vec<LogicalFieldDesc>,
        target_abi: TargetAbi,
    ) -> Self {
        // Enforce declaration-order ordinal (0..N-1)
        fields.sort_by_key(|f| f.ordinal);
        let has_duplicates = fields.windows(2).any(|w| w[0].ordinal == w[1].ordinal);
        if has_duplicates {
            for (i, f) in fields.iter_mut().enumerate() {
                f.ordinal = i as u16;
            }
        }

        let mut current_offset = 0usize;
        let mut max_align = 1usize;
        let mut is_c_flat = true;

        for field in &mut fields {
            let mut align = field.tag.alignment();
            if matches!(field.tag, CanonicalTypeTag::StringSlice | CanonicalTypeTag::BinarySlice) {
                if target_abi.pointer_width == 4 {
                    field.byte_size = target_abi.pointer_width * 2 + 8;
                    align = 4;
                }
            }
            if target_abi.max_alignment > 0 && align > target_abi.max_alignment {
                align = target_abi.max_alignment;
            }
            if align > max_align {
                max_align = align;
            }

            // Align current offset
            let padding = (align - (current_offset % align)) % align;
            current_offset += padding;

            field.byte_offset = current_offset;
            current_offset += field.byte_size;

            if field.tag.is_managed_slice() || matches!(field.tag, CanonicalTypeTag::NestedStruct(_)) {
                is_c_flat = false;
            }
        }

        if target_abi.max_alignment > 0 && max_align > target_abi.max_alignment {
            max_align = target_abi.max_alignment;
        }

        // Tail padding to round up to max_align
        let tail_padding = (max_align - (current_offset % max_align)) % max_align;
        let total_size = current_offset + tail_padding;

        Self {
            type_id,
            type_name: type_name.into(),
            fields,
            size_bytes: total_size,
            align_bytes: max_align,
            is_c_flat,
            target_abi,
        }
    }

    pub fn field_by_ordinal(&self, ordinal: u16) -> Option<&LogicalFieldDesc> {
        self.fields.iter().find(|f| f.ordinal == ordinal)
    }

    pub fn field_by_name(&self, name: &str) -> Option<&LogicalFieldDesc> {
        self.fields.iter().find(|f| f.name == name)
    }

    pub fn field_count(&self) -> usize {
        self.fields.len()
    }

    /// Verifies if layout is compatible with pure C flat structs
    pub fn assert_c_flat(&self) -> Result<(), &'static str> {
        if self.is_c_flat {
            Ok(())
        } else {
            Err("Layout contains managed slices or nested dynamically-managed aggregates")
        }
    }
}

impl fmt::Display for LogicalLayoutV1 {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "LogicalLayoutV1(name: {}, size: {}, align: {}, fields: {})",
            self.type_name,
            self.size_bytes,
            self.align_bytes,
            self.fields.len()
        )
    }
}

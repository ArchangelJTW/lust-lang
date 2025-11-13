use crate::ast::{Type, TypeKind};
use crate::number::{LustFloat, LustInt};
use alloc::boxed::Box;
use alloc::vec::Vec;
use core::mem::{align_of, size_of};
use hashbrown::HashMap;

/// Describes how a type is specialized in JIT traces
#[derive(Debug, Clone, PartialEq)]
pub enum SpecializedLayout {
    /// Raw scalar value on stack
    Scalar { size: usize, align: usize },

    /// Specialized vector (ptr, len, cap on stack)
    Vec {
        element_layout: Box<SpecializedLayout>,
        element_size: usize,
    },

    /// Specialized hash map
    HashMap {
        key_layout: Box<SpecializedLayout>,
        value_layout: Box<SpecializedLayout>,
    },

    /// Unboxed struct with specialized fields
    Struct {
        field_layouts: Vec<SpecializedLayout>,
        field_offsets: Vec<usize>,
        total_size: usize,
    },

    /// Unboxed tuple
    Tuple {
        element_layouts: Vec<SpecializedLayout>,
        element_offsets: Vec<usize>,
        total_size: usize,
    },
}

impl SpecializedLayout {
    /// Get the stack space needed for this specialized value
    pub fn stack_size(&self) -> usize {
        match self {
            SpecializedLayout::Scalar { size, .. } => *size,
            SpecializedLayout::Vec { .. } => {
                // Vec<T> is represented as (ptr, len, cap)
                size_of::<usize>() * 3
            }
            SpecializedLayout::HashMap { .. } => {
                // HashMap metadata - we'll need to figure out exact size
                // For now, use a conservative estimate
                32
            }
            SpecializedLayout::Struct { total_size, .. } => *total_size,
            SpecializedLayout::Tuple { total_size, .. } => *total_size,
        }
    }

    /// Get the alignment requirement for this specialized value
    pub fn alignment(&self) -> usize {
        match self {
            SpecializedLayout::Scalar { align, .. } => *align,
            SpecializedLayout::Vec { .. } => align_of::<usize>(),
            SpecializedLayout::HashMap { .. } => align_of::<usize>(),
            SpecializedLayout::Struct { field_layouts, .. } => {
                // Use maximum alignment of all fields
                field_layouts
                    .iter()
                    .map(|l| l.alignment())
                    .max()
                    .unwrap_or(1)
            }
            SpecializedLayout::Tuple { element_layouts, .. } => {
                // Use maximum alignment of all elements
                element_layouts
                    .iter()
                    .map(|l| l.alignment())
                    .max()
                    .unwrap_or(1)
            }
        }
    }
}

/// Maps TypeKind to its specialized representation
pub struct SpecializationRegistry {
    cache: HashMap<TypeKind, Option<SpecializedLayout>>,
}

impl SpecializationRegistry {
    pub fn new() -> Self {
        Self {
            cache: HashMap::new(),
        }
    }

    /// Try to get a specialized layout for a type
    pub fn get_specialization(&mut self, type_kind: &TypeKind) -> Option<SpecializedLayout> {
        if let Some(cached) = self.cache.get(type_kind) {
            return cached.clone();
        }

        let layout = self.compute_specialization(type_kind);
        self.cache.insert(type_kind.clone(), layout.clone());
        layout
    }

    fn compute_specialization(&self, type_kind: &TypeKind) -> Option<SpecializedLayout> {
        match type_kind {
            // Primitives can be unboxed to raw values
            TypeKind::Int => Some(SpecializedLayout::Scalar {
                size: size_of::<LustInt>(),
                align: align_of::<LustInt>(),
            }),

            TypeKind::Float => Some(SpecializedLayout::Scalar {
                size: size_of::<LustFloat>(),
                align: align_of::<LustFloat>(),
            }),

            TypeKind::Bool => Some(SpecializedLayout::Scalar {
                size: 1,
                align: 1,
            }),

            // Array<T> where T is specializable
            TypeKind::Array(element_type) => {
                self.get_specialization_for_type(element_type)
                    .map(|elem_layout| SpecializedLayout::Vec {
                        element_size: elem_layout.stack_size(),
                        element_layout: Box::new(elem_layout),
                    })
            }

            // Map<K, V> where K and V are specializable
            TypeKind::Map(key_type, value_type) => {
                let key_layout = self.get_specialization_for_type(key_type)?;
                let value_layout = self.get_specialization_for_type(value_type)?;
                Some(SpecializedLayout::HashMap {
                    key_layout: Box::new(key_layout),
                    value_layout: Box::new(value_layout),
                })
            }

            // Tuple with specializable elements
            TypeKind::Tuple(elements) => {
                let mut element_layouts = Vec::new();
                let mut element_offsets = Vec::new();
                let mut offset = 0;

                for elem_type in elements {
                    let layout = self.get_specialization_for_type(elem_type)?;
                    let align = layout.alignment();
                    // Align offset to element's alignment requirement
                    offset = (offset + align - 1) & !(align - 1);
                    element_offsets.push(offset);
                    offset += layout.stack_size();
                    element_layouts.push(layout);
                }

                Some(SpecializedLayout::Tuple {
                    element_layouts,
                    element_offsets,
                    total_size: offset,
                })
            }

            // GenericInstance (e.g., Array<int>, Map<string, int>)
            TypeKind::GenericInstance { name, type_args } => {
                // Reconstruct the proper TypeKind and recurse
                match name.as_str() {
                    "Array" if type_args.len() == 1 => {
                        self.compute_specialization(&TypeKind::Array(Box::new(type_args[0].clone())))
                    }
                    "Map" if type_args.len() == 2 => self.compute_specialization(&TypeKind::Map(
                        Box::new(type_args[0].clone()),
                        Box::new(type_args[1].clone()),
                    )),
                    _ => None,
                }
            }

            // Not specializable
            _ => None,
        }
    }

    /// Helper to get specialization for a Type (not just TypeKind)
    fn get_specialization_for_type(&self, ty: &Type) -> Option<SpecializedLayout> {
        self.compute_specialization(&ty.kind)
    }
}

impl Default for SpecializationRegistry {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ast::Span;

    #[test]
    fn test_primitive_specialization() {
        let mut registry = SpecializationRegistry::new();

        let int_layout = registry.get_specialization(&TypeKind::Int);
        assert!(int_layout.is_some());
        assert_eq!(int_layout.unwrap().stack_size(), size_of::<LustInt>());

        let float_layout = registry.get_specialization(&TypeKind::Float);
        assert!(float_layout.is_some());
        assert_eq!(float_layout.unwrap().stack_size(), size_of::<LustFloat>());

        let bool_layout = registry.get_specialization(&TypeKind::Bool);
        assert!(bool_layout.is_some());
        assert_eq!(bool_layout.unwrap().stack_size(), 1);
    }

    #[test]
    fn test_array_int_specialization() {
        let mut registry = SpecializationRegistry::new();

        let array_int_type = TypeKind::Array(Box::new(Type::new(TypeKind::Int, Span::dummy())));
        let layout = registry.get_specialization(&array_int_type);

        assert!(layout.is_some());
        if let Some(SpecializedLayout::Vec {
            element_size,
            element_layout,
        }) = layout
        {
            assert_eq!(element_size, size_of::<LustInt>());
            assert!(matches!(
                *element_layout,
                SpecializedLayout::Scalar { .. }
            ));
        } else {
            panic!("Expected Vec layout");
        }
    }

    #[test]
    fn test_cache_works() {
        let mut registry = SpecializationRegistry::new();

        // First call computes
        let layout1 = registry.get_specialization(&TypeKind::Int);
        // Second call uses cache
        let layout2 = registry.get_specialization(&TypeKind::Int);

        assert_eq!(layout1, layout2);
    }
}

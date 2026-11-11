use crate::data::{DataValue, Document, ExactNumber};
use crate::schema::error::ValidationError;
use crate::schema::ir::SchemaIR;

pub fn validate_ir(
    schema: &SchemaIR,
    value: &DataValue,
    path: &mut Vec<String>,
    errors: &mut Vec<ValidationError>,
) -> Option<DataValue> {
    match schema {
        SchemaIR::Any => Some(value.clone()),

        SchemaIR::String { min_len, max_len, regex } => {
            if let DataValue::String(s) = value {
                let mut valid = true;
                if let Some(min) = min_len {
                    if s.chars().count() < *min {
                        errors.push(ValidationError::new(
                            path.clone(),
                            format!("String length must be at least {}", min),
                            "too_small",
                        ).with_received(s.chars().count().to_string()));
                        valid = false;
                    }
                }
                if let Some(max) = max_len {
                    if s.chars().count() > *max {
                        errors.push(ValidationError::new(
                            path.clone(),
                            format!("String length must be at most {}", max),
                            "too_big",
                        ).with_received(s.chars().count().to_string()));
                        valid = false;
                    }
                }
                if let Some(pattern) = regex {
                    match crate::regex::BoundedRegex::new(pattern) {
                        Ok(re) => match re.is_match(s) {
                            Ok(true) => {}
                            Ok(false) => {
                                errors.push(
                                    ValidationError::new(
                                        path.clone(),
                                        format!("String does not match pattern '{}'", pattern),
                                        "regex_mismatch",
                                    )
                                    .with_received(s.clone()),
                                );
                                valid = false;
                            }
                            Err(e) => {
                                errors.push(ValidationError::new(
                                    path.clone(),
                                    format!("Regex execution error: {}", e),
                                    "regex_error",
                                ));
                                valid = false;
                            }
                        },
                        Err(e) => {
                            errors.push(ValidationError::new(
                                path.clone(),
                                format!("Invalid regex pattern '{}': {}", pattern, e),
                                "invalid_regex",
                            ));
                            valid = false;
                        }
                    }
                }
                if valid {
                    Some(DataValue::String(s.clone()))
                } else {
                    None
                }
            } else {
                errors.push(ValidationError::new(
                    path.clone(),
                    "Expected string",
                    "invalid_type",
                ).with_expected("string").with_received(value_type_name(value)));
                None
            }
        }

        SchemaIR::Int { min, max } => {
            let int_val = match value {
                DataValue::Number(ExactNumber::Int(i)) => Some(*i),
                DataValue::Number(ExactNumber::UInt(u)) if *u <= i64::MAX as u64 => Some(*u as i64),
                _ => None,
            };

            if let Some(i) = int_val {
                let mut valid = true;
                if let Some(min_val) = min {
                    if i < *min_val {
                        errors.push(ValidationError::new(
                            path.clone(),
                            format!("Integer must be >= {}", min_val),
                            "too_small",
                        ).with_received(i.to_string()));
                        valid = false;
                    }
                }
                if let Some(max_val) = max {
                    if i > *max_val {
                        errors.push(ValidationError::new(
                            path.clone(),
                            format!("Integer must be <= {}", max_val),
                            "too_big",
                        ).with_received(i.to_string()));
                        valid = false;
                    }
                }
                if valid {
                    Some(DataValue::Number(ExactNumber::Int(i)))
                } else {
                    None
                }
            } else {
                errors.push(ValidationError::new(
                    path.clone(),
                    "Expected integer",
                    "invalid_type",
                ).with_expected("integer").with_received(value_type_name(value)));
                None
            }
        }

        SchemaIR::Float { min, max } => {
            let float_val = match value {
                DataValue::Number(ExactNumber::Float(f)) => Some(*f),
                DataValue::Number(ExactNumber::Int(i)) => Some(*i as f64),
                DataValue::Number(ExactNumber::UInt(u)) => Some(*u as f64),
                _ => None,
            };

            if let Some(f) = float_val {
                let mut valid = true;
                if f.is_nan() {
                    if min.is_some() || max.is_some() {
                        errors.push(ValidationError::new(
                            path.clone(),
                            "Float value must not be NaN when bounds are specified",
                            "invalid_range",
                        ).with_received("NaN"));
                        valid = false;
                    }
                } else {
                    if let Some(min_val) = min {
                        if f < *min_val {
                            errors.push(ValidationError::new(
                                path.clone(),
                                format!("Float must be >= {}", min_val),
                                "too_small",
                            ).with_received(f.to_string()));
                            valid = false;
                        }
                    }
                    if let Some(max_val) = max {
                        if f > *max_val {
                            errors.push(ValidationError::new(
                                path.clone(),
                                format!("Float must be <= {}", max_val),
                                "too_big",
                            ).with_received(f.to_string()));
                            valid = false;
                        }
                    }
                }
                if valid {
                    Some(DataValue::Number(ExactNumber::Float(f)))
                } else {
                    None
                }
            } else {
                errors.push(ValidationError::new(
                    path.clone(),
                    "Expected float",
                    "invalid_type",
                ).with_expected("float").with_received(value_type_name(value)));
                None
            }
        }

        SchemaIR::Bool => {
            if let DataValue::Bool(b) = value {
                Some(DataValue::Bool(*b))
            } else {
                errors.push(ValidationError::new(
                    path.clone(),
                    "Expected boolean",
                    "invalid_type",
                ).with_expected("boolean").with_received(value_type_name(value)));
                None
            }
        }

        SchemaIR::Binary { min_len, max_len } => {
            let bytes_len = match value {
                DataValue::Binary(b) => Some(b.len()),
                _ => None,
            };

            if let Some(len) = bytes_len {
                let mut valid = true;
                if let Some(min) = min_len {
                    if len < *min {
                        errors.push(ValidationError::new(
                            path.clone(),
                            format!("Binary length must be at least {}", min),
                            "too_small",
                        ).with_received(len.to_string()));
                        valid = false;
                    }
                }
                if let Some(max) = max_len {
                    if len > *max {
                        errors.push(ValidationError::new(
                            path.clone(),
                            format!("Binary length must be at most {}", max),
                            "too_big",
                        ).with_received(len.to_string()));
                        valid = false;
                    }
                }
                if valid {
                    Some(value.clone())
                } else {
                    None
                }
            } else {
                errors.push(ValidationError::new(
                    path.clone(),
                    "Expected binary",
                    "invalid_type",
                ).with_expected("binary").with_received(value_type_name(value)));
                None
            }
        }

        SchemaIR::ObjectId => {
            if let DataValue::ObjectId(oid) = value {
                Some(DataValue::ObjectId(*oid))
            } else {
                errors.push(ValidationError::new(
                    path.clone(),
                    "Expected ObjectId",
                    "invalid_type",
                ).with_expected("ObjectId").with_received(value_type_name(value)));
                None
            }
        }

        SchemaIR::DateTimeUtc => {
            if let DataValue::DateTimeUtc(dt) = value {
                Some(DataValue::DateTimeUtc(*dt))
            } else {
                errors.push(ValidationError::new(
                    path.clone(),
                    "Expected DateTimeUtc",
                    "invalid_type",
                ).with_expected("DateTimeUtc").with_received(value_type_name(value)));
                None
            }
        }

        SchemaIR::Decimal128 => {
            if let DataValue::Number(ExactNumber::Decimal128(d)) = value {
                Some(DataValue::Number(ExactNumber::Decimal128(*d)))
            } else {
                errors.push(ValidationError::new(
                    path.clone(),
                    "Expected Decimal128",
                    "invalid_type",
                ).with_expected("Decimal128").with_received(value_type_name(value)));
                None
            }
        }

        SchemaIR::Literal(expected_val) => {
            if value == expected_val {
                Some(value.clone())
            } else {
                errors.push(ValidationError::new(
                    path.clone(),
                    format!("Expected literal value {:?}", expected_val),
                    "invalid_literal",
                ));
                None
            }
        }

        SchemaIR::Array { item, min_items, max_items } => {
            if let DataValue::Array(items) = value {
                let mut valid = true;
                if let Some(min) = min_items {
                    if items.len() < *min {
                        errors.push(ValidationError::new(
                            path.clone(),
                            format!("Array must have at least {} items", min),
                            "too_few_items",
                        ).with_received(items.len().to_string()));
                        valid = false;
                    }
                }
                if let Some(max) = max_items {
                    if items.len() > *max {
                        errors.push(ValidationError::new(
                            path.clone(),
                            format!("Array must have at most {} items", max),
                            "too_many_items",
                        ).with_received(items.len().to_string()));
                        valid = false;
                    }
                }

                let mut validated_items = Vec::with_capacity(items.len());
                for (idx, elem) in items.iter().enumerate() {
                    path.push(idx.to_string());
                    if let Some(val) = validate_ir(item, elem, path, errors) {
                        validated_items.push(val);
                    } else {
                        valid = false;
                    }
                    path.pop();
                }

                if valid {
                    Some(DataValue::Array(validated_items))
                } else {
                    None
                }
            } else {
                errors.push(ValidationError::new(
                    path.clone(),
                    "Expected array",
                    "invalid_type",
                ).with_expected("array").with_received(value_type_name(value)));
                None
            }
        }

        SchemaIR::Object { fields, strict } => {
            if let DataValue::Document(doc) = value {
                let mut validated_doc = Document::new();
                let mut valid = true;

                // Validate defined fields
                for field in fields {
                    path.push(field.name.clone());
                    match doc.get(&field.name) {
                        Some(val) => {
                            if let Some(v) = validate_ir(&field.schema, val, path, errors) {
                                let _ = validated_doc.try_insert(field.name.clone(), v);
                            } else {
                                valid = false;
                            }
                        }
                        None => {
                            if let Some(default_val) = &field.default {
                                let _ = validated_doc.try_insert(field.name.clone(), default_val.clone());
                            } else if field.optional {
                                // Field is optional and omitted
                            } else {
                                errors.push(ValidationError::new(
                                    path.clone(),
                                    format!("Required field '{}' is missing", field.name),
                                    "missing_field",
                                ));
                                valid = false;
                            }
                        }
                    }
                    path.pop();
                }

                // If strict, reject unexpected keys
                if *strict {
                    for (k, _) in doc.iter() {
                        if !fields.iter().any(|f| &f.name == k) {
                            path.push(k.clone());
                            errors.push(ValidationError::new(
                                path.clone(),
                                format!("Unrecognized key '{}' in strict object", k),
                                "unrecognized_key",
                            ));
                            path.pop();
                            valid = false;
                        }
                    }
                } else {
                    // Copy other fields through
                    for (k, v) in doc.iter() {
                        if !fields.iter().any(|f| &f.name == k) {
                            let _ = validated_doc.try_insert(k.clone(), v.clone());
                        }
                    }
                }

                if valid {
                    Some(DataValue::Document(validated_doc))
                } else {
                    None
                }
            } else {
                errors.push(ValidationError::new(
                    path.clone(),
                    "Expected object document",
                    "invalid_type",
                ).with_expected("object").with_received(value_type_name(value)));
                None
            }
        }

        SchemaIR::Record { key, value: val_schema } => {
            if let DataValue::Document(doc) = value {
                let mut validated_doc = Document::new();
                let mut valid = true;

                for (k, v) in doc.iter() {
                    path.push(k.clone());
                    let key_as_val = DataValue::String(k.clone());
                    let mut key_errors = Vec::new();
                    if validate_ir(key, &key_as_val, path, &mut key_errors).is_none() {
                        errors.extend(key_errors);
                        valid = false;
                    }
                    if let Some(val) = validate_ir(val_schema, v, path, errors) {
                        let _ = validated_doc.try_insert(k.clone(), val);
                    } else {
                        valid = false;
                    }
                    path.pop();
                }

                if valid {
                    Some(DataValue::Document(validated_doc))
                } else {
                    None
                }
            } else {
                errors.push(ValidationError::new(
                    path.clone(),
                    "Expected object document",
                    "invalid_type",
                ).with_expected("object").with_received(value_type_name(value)));
                None
            }
        }

        SchemaIR::Union(variants) => {
            for variant in variants {
                let mut branch_errors = Vec::new();
                if let Some(val) = validate_ir(variant, value, path, &mut branch_errors) {
                    return Some(val);
                }
            }
            errors.push(ValidationError::new(
                path.clone(),
                "Value does not match any union variant",
                "invalid_union",
            ));
            None
        }

        SchemaIR::DiscriminatedUnion { discriminator, variants } => {
            if let DataValue::Document(doc) = value {
                if let Some(tag_val) = doc.get(discriminator) {
                    for variant in variants {
                        if let SchemaIR::Object { fields, .. } = variant {
                            if let Some(field) = fields.iter().find(|f| &f.name == discriminator) {
                                if let SchemaIR::Literal(lit) = &field.schema {
                                    if lit == tag_val {
                                        return validate_ir(variant, value, path, errors);
                                    }
                                }
                            }
                        }
                    }
                    errors.push(ValidationError::new(
                        path.clone(),
                        format!("Unknown discriminator tag value '{:?}'", tag_val),
                        "invalid_discriminator",
                    ));
                    None
                } else {
                    path.push(discriminator.clone());
                    errors.push(ValidationError::new(
                        path.clone(),
                        format!("Missing discriminator property '{}'", discriminator),
                        "missing_discriminator",
                    ));
                    path.pop();
                    None
                }
            } else {
                errors.push(ValidationError::new(
                    path.clone(),
                    "Expected object document for discriminated union",
                    "invalid_type",
                ));
                None
            }
        }

        SchemaIR::Optional(inner) => {
            // In standalone value validation, null or undefined can match optional
            if matches!(value, DataValue::Null) {
                Some(DataValue::Null)
            } else {
                validate_ir(inner, value, path, errors)
            }
        }

        SchemaIR::Nullable(inner) => {
            if matches!(value, DataValue::Null) {
                Some(DataValue::Null)
            } else {
                validate_ir(inner, value, path, errors)
            }
        }

        SchemaIR::Refine { inner, predicate, message } => {
            if let Some(inner_val) = validate_ir(inner, value, path, errors) {
                if (predicate.f)(&inner_val) {
                    Some(inner_val)
                } else {
                    let msg = message.as_deref().unwrap_or("Value failed refinement predicate");
                    errors.push(
                        ValidationError::new(path.clone(), msg, "custom_refinement")
                            .with_received(format!("{:?}", inner_val)),
                    );
                    None
                }
            } else {
                None
            }
        }

        SchemaIR::Transform { inner, transform } => {
            if let Some(inner_val) = validate_ir(inner, value, path, errors) {
                let transformed = (transform.f)(inner_val);
                Some(transformed)
            } else {
                None
            }
        }

        SchemaIR::InvalidRegex { pattern, error } => {
            errors.push(ValidationError::new(
                path.clone(),
                format!("Invalid regex pattern '{}': {}", pattern, error),
                "invalid_regex",
            ));
            None
        }
    }
}

fn value_type_name(value: &DataValue) -> String {
    match value {
        DataValue::Null => "null".to_string(),
        DataValue::Bool(_) => "boolean".to_string(),
        DataValue::Number(ExactNumber::Int(_)) => "integer".to_string(),
        DataValue::Number(ExactNumber::UInt(_)) => "unsigned integer".to_string(),
        DataValue::Number(ExactNumber::Float(_)) => "float".to_string(),
        DataValue::Number(ExactNumber::Decimal128(_)) => "decimal128".to_string(),
        DataValue::String(_) => "string".to_string(),
        DataValue::Array(_) => "array".to_string(),
        DataValue::Document(_) => "object".to_string(),
        DataValue::Binary(_) => "binary".to_string(),
        DataValue::ObjectId(_) => "object_id".to_string(),
        DataValue::DateTimeUtc(_) => "datetime_utc".to_string(),
    }
}

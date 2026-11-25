//! Multi-format byte validation pipeline and facades (DS-93).
//!
//! Provides `validate_json`, `validate_jsonc`, `validate_bson`, `validate_yaml`, and
//! `validate_toon` methods on `Schema`, strictly differentiating syntax parse errors
//! from schema constraint failures without masking errors.

use crate::data::DataValue;
use crate::schema::builder::Schema;
use crate::schema::error::{ValidationError, ValidationErrors};
use crate::schema::Effect;

impl Schema {
    /// Validate raw JSON bytes against this schema.
    ///
    /// Preserves tagged syntax Parse errors vs schema Constraint errors.
    pub fn validate_json(&self, bytes: &[u8]) -> Effect<DataValue, ValidationErrors, ()> {
        let val = match crate::data::json::from_slice_with_limits(bytes, 256, 67_108_864) {
            Ok(v) => v,
            Err(e) => {
                return Effect::failure(ValidationErrors::single(ValidationError::parse(
                    e.to_string(),
                    "json_parse_error",
                )));
            }
        };
        self.validate(&val)
    }

    /// Validate raw JSONC bytes (with comments and trailing commas) against this schema.
    pub fn validate_jsonc(&self, bytes: &[u8]) -> Effect<DataValue, ValidationErrors, ()> {
        let val = match crate::data::jsonc::from_slice_with_limits(bytes, 256, 67_108_864) {
            Ok(v) => v,
            Err(e) => {
                return Effect::failure(ValidationErrors::single(ValidationError::parse(
                    e.to_string(),
                    "jsonc_parse_error",
                )));
            }
        };
        self.validate(&val)
    }

    /// Validate raw BSON bytes against this schema.
    pub fn validate_bson(&self, bytes: &[u8]) -> Effect<DataValue, ValidationErrors, ()> {
        let mut decoder = crate::data::bson::BsonDecoder::new(bytes);
        let doc = match decoder.decode_document() {
            Ok(d) => d,
            Err(e) => {
                return Effect::failure(ValidationErrors::single(ValidationError::parse(
                    e.to_string(),
                    "bson_parse_error",
                )));
            }
        };
        self.validate(&DataValue::Document(doc))
    }

    /// Validate raw YAML bytes against this schema using LustSafeYaml12 profile.
    pub fn validate_yaml(&self, bytes: &[u8]) -> Effect<DataValue, ValidationErrors, ()> {
        let s = match std::str::from_utf8(bytes) {
            Ok(s) => s,
            Err(e) => {
                return Effect::failure(ValidationErrors::single(ValidationError::parse(
                    format!("Invalid UTF-8 in YAML buffer: {}", e),
                    "yaml_utf8_error",
                )));
            }
        };
        let mut parser = crate::data::yaml::LustSafeYaml12::new();
        let val = match parser.parse(s) {
            Ok(v) => v,
            Err(e) => {
                return Effect::failure(ValidationErrors::single(ValidationError::parse(
                    e.to_string(),
                    "yaml_parse_error",
                )));
            }
        };
        self.validate(&val)
    }

    /// Validate raw TOON bytes against this schema.
    pub fn validate_toon(&self, bytes: &[u8]) -> Effect<DataValue, ValidationErrors, ()> {
        let s = match std::str::from_utf8(bytes) {
            Ok(s) => s,
            Err(e) => {
                return Effect::failure(ValidationErrors::single(ValidationError::parse(
                    format!("Invalid UTF-8 in TOON buffer: {}", e),
                    "toon_utf8_error",
                )));
            }
        };
        let val = match crate::data::toon::ToonCodec::decode(s) {
            Ok(v) => v,
            Err(e) => {
                return Effect::failure(ValidationErrors::single(ValidationError::parse(
                    e.to_string(),
                    "toon_parse_error",
                )));
            }
        };
        self.validate(&val)
    }
}

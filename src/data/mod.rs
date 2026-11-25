//! Data model, formats, zero-copy views, and budget tracking for Lust.
//!
//! # Examples
//!
//! ```rust
//! use lust::data::{Document, DataValue, ExactNumber};
//!
//! let mut doc = Document::new();
//! doc.try_insert("id", DataValue::Number(ExactNumber::Int(1))).unwrap();
//! doc.try_insert("name", DataValue::String("Lust".to_string())).unwrap();
//!
//! // Gate-SCHEMA-01: Duplicate keys are strictly rejected
//! assert!(doc.try_insert("id", DataValue::Number(ExactNumber::Int(2))).is_err());
//! ```

pub mod bson;
pub mod budget;
pub mod chunked_stream;
pub mod csv;
pub mod document;
pub mod json;
pub mod jsonc;
pub mod jsonl;
pub mod layout;
pub mod materialize;
pub mod msgpack;
pub mod number;
pub mod owner;
pub mod toon;
pub mod value;
pub mod view;
pub mod yaml;

pub use bson::{BsonArrayView, BsonDocumentView, BsonElementIter, BsonError, BsonValueView};
pub use budget::{BudgetError, BudgetTracker};
pub use chunked_stream::{
    BsonStreamDecoder, CsvStreamDecoder, JsonStreamDecoder, StreamDecoderOptions, StreamError,
    ToonStreamDecoder, YamlStreamDecoder,
};
pub use csv::{
    csv_to_documents, documents_to_csv, documents_to_tsv, parse_csv, parse_tsv, to_csv_string,
    to_tsv_string, tsv_to_documents, CsvError, CsvOptions, CsvParser, CsvTable,
};
pub use document::{Document, DuplicateKeyError};
pub use jsonl::{
    line_results, lines, lines_with_options, validate_line_results, validate_lines, write_lines,
    BlankLines, CancellationToken, ControlFlow, JsonlReader, JsonlReaderOptions, JsonlResults,
    JsonlValidated, JsonlValidatedResults, JsonlWriter, JsonlWriterOptions, LineError,
    LineErrorKind, StreamValidationError, StreamWriteError, WriteErrorKind,
};
pub use layout::{CanonicalTypeTag, LogicalFieldDesc, LogicalLayoutV1, TargetAbi};
pub use materialize::{
    materialize_from_document, materialize_from_item, materialize_into_struct, struct_def_to_layout,
    MaterializeError, MaterializedStruct,
};
pub use msgpack::{
    from_reader as from_msgpack_reader, from_reader_document as from_msgpack_reader_document,
    from_slice as from_msgpack_slice, from_slice_document as from_msgpack_slice_document,
    to_vec as to_msgpack_vec, to_vec_document as to_msgpack_vec_document, MsgPackError,
};
pub use number::{DataError, DateTimeUtc, Decimal128, ExactNumber, ObjectId};
pub use owner::{LeaseError, LeaseLifecycleState, OwnerId, OwnerRegistry, SlotEntry};
pub use toon::{ToonCodec, ToonError, ToonTable};
pub use value::DataValue;
pub use view::DataView;
pub use yaml::{LustSafeYaml12, YamlError};



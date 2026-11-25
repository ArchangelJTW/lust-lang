use std::fmt;

/// Category of validation error, differentiating codec syntax parse errors from schema constraint failures (DS-93).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum ValidationErrorKind {
    #[default]
    Constraint,
    Parse,
}

impl fmt::Display for ValidationErrorKind {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ValidationErrorKind::Constraint => write!(f, "Constraint"),
            ValidationErrorKind::Parse => write!(f, "Parse"),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ValidationError {
    pub path: Vec<String>,
    pub message: String,
    pub code: String,
    pub expected: Option<String>,
    pub received: Option<String>,
    pub kind: ValidationErrorKind,
}

impl ValidationError {
    pub fn new(path: Vec<String>, message: impl Into<String>, code: impl Into<String>) -> Self {
        Self {
            path,
            message: message.into(),
            code: code.into(),
            expected: None,
            received: None,
            kind: ValidationErrorKind::Constraint,
        }
    }

    /// Construct a syntax or codec parse error.
    pub fn parse(message: impl Into<String>, code: impl Into<String>) -> Self {
        Self {
            path: Vec::new(),
            message: message.into(),
            code: code.into(),
            expected: None,
            received: None,
            kind: ValidationErrorKind::Parse,
        }
    }

    pub fn with_kind(mut self, kind: ValidationErrorKind) -> Self {
        self.kind = kind;
        self
    }

    pub fn with_expected(mut self, expected: impl Into<String>) -> Self {
        self.expected = Some(expected.into());
        self
    }

    pub fn with_received(mut self, received: impl Into<String>) -> Self {
        self.received = Some(received.into());
        self
    }

    pub fn is_parse(&self) -> bool {
        matches!(self.kind, ValidationErrorKind::Parse)
    }

    pub fn is_constraint(&self) -> bool {
        matches!(self.kind, ValidationErrorKind::Constraint)
    }

    pub fn path_string(&self) -> String {
        if self.path.is_empty() {
            "$".to_string()
        } else {
            self.path.join(".")
        }
    }
}

impl fmt::Display for ValidationError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "[{}] {}: {}", self.code, self.path_string(), self.message)?;
        if let (Some(exp), Some(rec)) = (&self.expected, &self.received) {
            write!(f, " (expected {}, received {})", exp, rec)?;
        }
        Ok(())
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ValidationErrors {
    pub errors: Vec<ValidationError>,
}

impl ValidationErrors {
    pub fn new(errors: Vec<ValidationError>) -> Self {
        Self { errors }
    }

    pub fn single(err: ValidationError) -> Self {
        Self { errors: vec![err] }
    }

    pub fn is_empty(&self) -> bool {
        self.errors.is_empty()
    }

    pub fn len(&self) -> usize {
        self.errors.len()
    }

    pub fn has_parse_error(&self) -> bool {
        self.errors.iter().any(|e| e.is_parse())
    }

    pub fn has_constraint_error(&self) -> bool {
        self.errors.iter().any(|e| e.is_constraint())
    }
}

impl fmt::Display for ValidationErrors {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        for (i, err) in self.errors.iter().enumerate() {
            if i > 0 {
                writeln!(f)?;
            }
            write!(f, "{}", err)?;
        }
        Ok(())
    }
}

impl std::error::Error for ValidationErrors {}

impl From<ValidationError> for ValidationErrors {
    fn from(err: ValidationError) -> Self {
        ValidationErrors::single(err)
    }
}

impl From<ValidationErrors> for ValidationError {
    fn from(mut errs: ValidationErrors) -> Self {
        if !errs.errors.is_empty() {
            errs.errors.remove(0)
        } else {
            ValidationError::new(vec![], "Validation failed", "validation_failed")
        }
    }
}

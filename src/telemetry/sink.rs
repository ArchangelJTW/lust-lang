use alloc::sync::Arc;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SpanStatus {
    Success,
    Fail,
    Die,
    Interrupt,
    Ok,
    Error,
}

impl SpanStatus {
    #[inline]
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::Success | Self::Ok => "success",
            Self::Fail | Self::Error => "fail",
            Self::Die => "die",
            Self::Interrupt => "interrupt",
        }
    }
}

pub trait TelemetrySink: Send + Sync + core::fmt::Debug {
    fn counter_add(&self, name: &str, delta: u64, labels: &[(&str, &str)]);
    fn gauge_set(&self, name: &str, value: f64, labels: &[(&str, &str)]);
    fn histogram_record(&self, name: &str, value: f64, labels: &[(&str, &str)]);
    fn span_enter(&self, name: &str, labels: &[(&str, &str)]) -> u64;
    fn span_exit(&self, span_id: u64, status: SpanStatus);
    fn flush(&self) {}
}

#[derive(Debug, Default, Clone, Copy)]
pub struct NoopSink;

impl TelemetrySink for NoopSink {
    #[inline(always)]
    fn counter_add(&self, _name: &str, _delta: u64, _labels: &[(&str, &str)]) {}

    #[inline(always)]
    fn gauge_set(&self, _name: &str, _value: f64, _labels: &[(&str, &str)]) {}

    #[inline(always)]
    fn histogram_record(&self, _name: &str, _value: f64, _labels: &[(&str, &str)]) {}

    #[inline(always)]
    fn span_enter(&self, _name: &str, _labels: &[(&str, &str)]) -> u64 {
        0
    }

    #[inline(always)]
    fn span_exit(&self, _span_id: u64, _status: SpanStatus) {}

    #[inline(always)]
    fn flush(&self) {}
}

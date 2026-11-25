// When telemetry features are disabled: complete zero-cost no-op, arguments are NEVER evaluated.
#[cfg(not(any(
    feature = "telemetry",
    feature = "telemetry-all",
    feature = "telemetry-vm",
    feature = "telemetry-jit",
    feature = "telemetry-effects",
    feature = "telemetry-user"
)))]
#[macro_export]
macro_rules! tel_emit {
    ($($arg:tt)*) => { () };
}

#[cfg(not(any(
    feature = "telemetry",
    feature = "telemetry-all",
    feature = "telemetry-vm",
    feature = "telemetry-jit",
    feature = "telemetry-effects",
    feature = "telemetry-user"
)))]
#[macro_export]
macro_rules! tel_counter {
    ($($arg:tt)*) => { () };
}

#[cfg(not(any(
    feature = "telemetry",
    feature = "telemetry-all",
    feature = "telemetry-vm",
    feature = "telemetry-jit",
    feature = "telemetry-effects",
    feature = "telemetry-user"
)))]
#[macro_export]
macro_rules! tel_gauge {
    ($($arg:tt)*) => { () };
}

#[cfg(not(any(
    feature = "telemetry",
    feature = "telemetry-all",
    feature = "telemetry-vm",
    feature = "telemetry-jit",
    feature = "telemetry-effects",
    feature = "telemetry-user"
)))]
#[macro_export]
macro_rules! tel_histogram {
    ($($arg:tt)*) => { () };
}

#[cfg(not(any(
    feature = "telemetry",
    feature = "telemetry-all",
    feature = "telemetry-vm",
    feature = "telemetry-jit",
    feature = "telemetry-effects",
    feature = "telemetry-user"
)))]
#[macro_export]
macro_rules! tel_span_enter {
    ($($arg:tt)*) => { 0u64 };
}

#[cfg(not(any(
    feature = "telemetry",
    feature = "telemetry-all",
    feature = "telemetry-vm",
    feature = "telemetry-jit",
    feature = "telemetry-effects",
    feature = "telemetry-user"
)))]
#[macro_export]
macro_rules! tel_span_exit {
    ($($arg:tt)*) => { () };
}

// When telemetry is enabled:
#[cfg(any(
    feature = "telemetry",
    feature = "telemetry-all",
    feature = "telemetry-vm",
    feature = "telemetry-jit",
    feature = "telemetry-effects",
    feature = "telemetry-user"
))]
#[macro_export]
macro_rules! tel_emit {
    ($sink:expr, $method:ident ($($arg:expr),* $(,)?)) => {
        if let Some(ref sink) = $sink {
            sink.$method($($arg),*);
        }
    };
}

#[cfg(any(
    feature = "telemetry",
    feature = "telemetry-all",
    feature = "telemetry-vm",
    feature = "telemetry-jit",
    feature = "telemetry-effects",
    feature = "telemetry-user"
))]
#[macro_export]
macro_rules! tel_counter {
    ($vm:expr, $name:expr, $delta:expr) => {
        $vm.record_telemetry_counter($name, $delta, &[]);
    };
    ($vm:expr, $name:expr, $delta:expr, $labels:expr) => {
        $vm.record_telemetry_counter($name, $delta, $labels);
    };
}

#[cfg(any(
    feature = "telemetry",
    feature = "telemetry-all",
    feature = "telemetry-vm",
    feature = "telemetry-jit",
    feature = "telemetry-effects",
    feature = "telemetry-user"
))]
#[macro_export]
macro_rules! tel_gauge {
    ($vm:expr, $name:expr, $val:expr) => {
        $vm.record_telemetry_gauge($name, $val, &[]);
    };
    ($vm:expr, $name:expr, $val:expr, $labels:expr) => {
        $vm.record_telemetry_gauge($name, $val, $labels);
    };
}

#[cfg(any(
    feature = "telemetry",
    feature = "telemetry-all",
    feature = "telemetry-vm",
    feature = "telemetry-jit",
    feature = "telemetry-effects",
    feature = "telemetry-user"
))]
#[macro_export]
macro_rules! tel_histogram {
    ($vm:expr, $name:expr, $val:expr) => {
        $vm.record_telemetry_histogram($name, $val, &[]);
    };
    ($vm:expr, $name:expr, $val:expr, $labels:expr) => {
        $vm.record_telemetry_histogram($name, $val, $labels);
    };
}

#[cfg(any(
    feature = "telemetry",
    feature = "telemetry-all",
    feature = "telemetry-vm",
    feature = "telemetry-jit",
    feature = "telemetry-effects",
    feature = "telemetry-user"
))]
#[macro_export]
macro_rules! tel_span_enter {
    ($vm:expr, $name:expr, $labels:expr) => {
        $vm.telemetry_span_enter($name, $labels)
    };
}

#[cfg(any(
    feature = "telemetry",
    feature = "telemetry-all",
    feature = "telemetry-vm",
    feature = "telemetry-jit",
    feature = "telemetry-effects",
    feature = "telemetry-user"
))]
#[macro_export]
macro_rules! tel_span_exit {
    ($vm:expr, $span_id:expr, $status:expr) => {
        $vm.telemetry_span_exit($span_id, $status);
    };
}

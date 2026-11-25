use alloc::{
    string::{String, ToString},
    vec::Vec,
};
use core::sync::atomic::{AtomicU64, Ordering};
#[cfg(feature = "std")]
use std::sync::Mutex;

use crate::config::TelemetryLimitsConfig;

#[derive(Debug)]
pub struct TelemetryCardinalityGuard {
    limits: TelemetryLimitsConfig,
    #[cfg(feature = "std")]
    metric_names: Mutex<hashbrown::HashSet<String>>,
    #[cfg(feature = "std")]
    series: Mutex<hashbrown::HashSet<String>>,
    dropped_series_total: AtomicU64,
}

impl TelemetryCardinalityGuard {
    pub fn new(limits: TelemetryLimitsConfig) -> Self {
        Self {
            limits,
            #[cfg(feature = "std")]
            metric_names: Mutex::new(hashbrown::HashSet::new()),
            #[cfg(feature = "std")]
            series: Mutex::new(hashbrown::HashSet::new()),
            dropped_series_total: AtomicU64::new(0),
        }
    }

    pub fn dropped_series_total(&self) -> u64 {
        self.dropped_series_total.load(Ordering::Relaxed)
    }

    #[cfg(feature = "std")]
    pub fn check_and_fold<'a>(
        &self,
        metric_name: &'a str,
        labels: &[(&'a str, &'a str)],
    ) -> Option<Vec<(String, String)>> {
        // 1. Check metric names bound
        let mut names = self.metric_names.lock().unwrap();
        if !names.contains(metric_name) {
            if names.len() >= self.limits.max_unique_metric_names {
                self.dropped_series_total.fetch_add(1, Ordering::Relaxed);
                return None;
            }
            names.insert(metric_name.to_string());
        }
        drop(names);

        // 2. Bound label pairs
        let effective_labels = if labels.len() > self.limits.max_label_pairs_per_metric {
            &labels[..self.limits.max_label_pairs_per_metric]
        } else {
            labels
        };

        // 3. Truncate label values
        let sanitized: Vec<(String, String)> = effective_labels
            .iter()
            .map(|(k, v)| {
                let val = if v.len() > self.limits.max_label_value_len {
                    v[..self.limits.max_label_value_len].to_string()
                } else {
                    v.to_string()
                };
                (k.to_string(), val)
            })
            .collect();

        // 4. Series check
        let mut series_key = String::with_capacity(64);
        series_key.push_str(metric_name);
        for (k, v) in &sanitized {
            series_key.push(';');
            series_key.push_str(k);
            series_key.push('=');
            series_key.push_str(v);
        }

        let mut series = self.series.lock().unwrap();
        if !series.contains(&series_key) {
            if series.len() >= self.limits.max_unique_series {
                self.dropped_series_total.fetch_add(1, Ordering::Relaxed);
                // Fold into overflow = "true"
                return Some(alloc::vec![("overflow".to_string(), "true".to_string())]);
            }
            series.insert(series_key);
        }

        Some(sanitized)
    }

    #[cfg(not(feature = "std"))]
    pub fn check_and_fold<'a>(
        &self,
        _metric_name: &'a str,
        _labels: &[(&'a str, &'a str)],
    ) -> Option<Vec<(String, String)>> {
        None
    }
}

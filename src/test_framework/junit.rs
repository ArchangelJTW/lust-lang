//! JUnit XML 1.0 reporter formatter.

use super::runner::{TestOutcome, TestSuiteResult, TestSummary};

/// Escapes XML attributes and element bodies according to the XML 1.0 specification.
pub fn escape_xml(s: &str) -> String {
    s.replace('&', "&amp;")
        .replace('<', "&lt;")
        .replace('>', "&gt;")
        .replace('"', "&quot;")
        .replace('\'', "&apos;")
}

/// Formats test suites and summary into standard JUnit XML.
pub fn format(suites: &[TestSuiteResult], summary: &TestSummary) -> String {
    let mut out = String::new();
    out.push_str("<?xml version=\"1.0\" encoding=\"UTF-8\"?>\n");
    out.push_str(&format!(
        "<testsuites tests=\"{}\" failures=\"{}\" skipped=\"{}\" time=\"{:.3}\">\n",
        summary.total,
        summary.failed,
        summary.skipped,
        summary.duration.as_secs_f64()
    ));

    for suite in suites {
        out.push_str(&format!(
            "  <testsuite name=\"{}\" tests=\"{}\" time=\"{:.3}\">\n",
            escape_xml(&suite.name),
            suite.tests.len(),
            suite.duration.as_secs_f64()
        ));

        for test in &suite.tests {
            out.push_str(&format!(
                "    <testcase name=\"{}\" time=\"{:.3}\">",
                escape_xml(&test.name),
                test.duration.as_secs_f64()
            ));

            match &test.outcome {
                TestOutcome::Passed => {}
                TestOutcome::Failed(err) => {
                    out.push_str(&format!(
                        "\n      <failure message=\"{}\">{}</failure>\n    ",
                        escape_xml(&err.message),
                        escape_xml(&err.diff.clone().unwrap_or_else(|| err.message.clone()))
                    ));
                }
                TestOutcome::Skipped(reason) => {
                    out.push_str(&format!(
                        "\n      <skipped message=\"{}\"/>\n    ",
                        escape_xml(reason)
                    ));
                }
            }

            out.push_str("</testcase>\n");
        }

        out.push_str("  </testsuite>\n");
    }

    out.push_str("</testsuites>\n");
    out
}

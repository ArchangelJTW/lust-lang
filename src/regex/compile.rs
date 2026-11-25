use crate::regex::plan::RegexPlan;
use crate::regex::policy::{
    MAX_CAPTURE_GROUPS, MAX_COMPILED_PLAN_BYTES, MAX_NESTING_DEPTH, MAX_PATTERN_BYTES, RegexError,
};
use regex_automata::dfa::onepass::DFA;
use regex_automata::util::syntax::Config as SyntaxConfig;

fn classify_compile_error(pattern: &str, err: &(dyn std::error::Error + 'static)) -> RegexError {
    let mut msg = err.to_string();
    let mut curr = err.source();
    while let Some(src) = curr {
        msg.push_str(" -> ");
        msg.push_str(&src.to_string());
        curr = src.source();
    }
    if msg.contains("nested") || msg.contains("nesting") {
        RegexError::NestingTooDeep(MAX_NESTING_DEPTH + 1)
    } else if msg.contains("too many explicit capturing groups") || msg.contains("capturing groups") {
        RegexError::TooManyCaptureGroups(MAX_CAPTURE_GROUPS + 1)
    } else {
        RegexError::UnsupportedPattern(format!("{}: {}", pattern, msg))
    }
}

pub fn compile_bounded(pattern: &str) -> Result<RegexPlan, RegexError> {
    if pattern.len() > MAX_PATTERN_BYTES {
        return Err(RegexError::PatternTooLarge(pattern.len()));
    }

    let syntax_config = SyntaxConfig::new().nest_limit(MAX_NESTING_DEPTH as u32);

    // Strict one-pass DFA compiler: any pattern requiring backtracking or exceeding limits fails closed
    let dfa = DFA::builder()
        .syntax(syntax_config.clone())
        .build(pattern)
        .map_err(|e| classify_compile_error(pattern, &e))?;

    let fwd_dfa = regex_automata::hybrid::dfa::DFA::builder()
        .syntax(syntax_config.clone())
        .build(pattern)
        .map_err(|e| classify_compile_error(pattern, &e))?;

    let rev_dfa = regex_automata::hybrid::dfa::DFA::builder()
        .syntax(syntax_config)
        .configure(
            regex_automata::hybrid::dfa::DFA::config()
                .prefilter(None)
                .specialize_start_states(false)
                .match_kind(regex_automata::MatchKind::All),
        )
        .thompson(regex_automata::nfa::thompson::Config::new().reverse(true))
        .build(pattern)
        .map_err(|e| classify_compile_error(pattern, &e))?;

    let caps = dfa.create_captures();
    let group_count = caps.group_len();
    if group_count > MAX_CAPTURE_GROUPS {
        return Err(RegexError::TooManyCaptureGroups(group_count));
    }

    let estimated_bytes = dfa.memory_usage()
        .saturating_add(fwd_dfa.memory_usage())
        .saturating_add(rev_dfa.memory_usage());
    if estimated_bytes > MAX_COMPILED_PLAN_BYTES {
        return Err(RegexError::UnsupportedPattern(format!(
            "Compiled automaton memory {} exceeds limit {}",
            estimated_bytes, MAX_COMPILED_PLAN_BYTES
        )));
    }

    Ok(RegexPlan::new(pattern.to_string(), dfa, fwd_dfa, rev_dfa, group_count))
}

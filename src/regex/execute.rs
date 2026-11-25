use crate::regex::plan::RegexPlan;
use crate::regex::policy::{POLL_INTERVAL_WORK_UNITS, RegexError, RegexWorkTracker};
use regex_automata::Input;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RegexMatch {
    pub start: usize,
    pub end: usize,
    pub text: String,
}

pub fn is_match(plan: &RegexPlan, input_text: &str) -> Result<bool, RegexError> {
    let mut tracker = RegexWorkTracker::new(input_text.len(), plan.group_count, 0)?;
    is_match_with_tracker(plan, input_text, &mut tracker)
}

pub fn is_match_with_tracker(
    plan: &RegexPlan,
    input_text: &str,
    tracker: &mut RegexWorkTracker,
) -> Result<bool, RegexError> {
    tracker.poll()?;
    if tracker.cancellation.is_some() && input_text.len() > POLL_INTERVAL_WORK_UNITS as usize {
        return Err(RegexError::UnsupportedBackendFeature(
            "Pinned regex-automata API cannot expose true work quanta mid-search (>4096 bytes) without instrumented bounded adapter".to_string(),
        ));
    }

    let mut caps = plan.dfa.create_captures();
    let mut cache = plan.dfa.create_cache();

    let chunk_size = POLL_INTERVAL_WORK_UNITS as usize;
    let mut offset = 0;
    while offset < input_text.len() {
        let chunk = (input_text.len() - offset).min(chunk_size);
        tracker.charge_scanned(chunk as u64)?;
        tracker.charge_transitions(chunk as u64)?;
        offset += chunk;
    }

    // Onepass DFA requires Anchored::Yes
    let input = Input::new(input_text).anchored(regex_automata::Anchored::Yes);
    plan.dfa.try_search(&mut cache, &input, &mut caps).map_err(|e| {
        RegexError::SyntaxError(e.to_string())
    })?;

    tracker.poll()?;

    if caps.is_match() {
        tracker.charge_capture_updates(plan.group_count as u64)?;
        Ok(true)
    } else {
        Ok(false)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RegexFindStats {
    /// Forward plus reverse DFA searches. A linear find is 1 when there is no match, or 2 when there is one.
    pub dfa_searches: u32,
    /// Total poll points encountered during search.
    pub poll_count: u64,
}

pub fn find(plan: &RegexPlan, input_text: &str) -> Result<Option<RegexMatch>, RegexError> {
    Ok(find_with_stats(plan, input_text)?.0)
}

pub fn find_with_stats(
    plan: &RegexPlan,
    input_text: &str,
) -> Result<(Option<RegexMatch>, RegexFindStats), RegexError> {
    let mut tracker = RegexWorkTracker::new(input_text.len(), plan.group_count, 0)?;
    find_with_tracker(plan, input_text, &mut tracker)
}

pub fn find_with_tracker(
    plan: &RegexPlan,
    input_text: &str,
    tracker: &mut RegexWorkTracker,
) -> Result<(Option<RegexMatch>, RegexFindStats), RegexError> {
    tracker.poll()?;
    if tracker.cancellation.is_some() && input_text.len() > POLL_INTERVAL_WORK_UNITS as usize {
        return Err(RegexError::UnsupportedBackendFeature(
            "Pinned regex-automata API cannot expose true work quanta mid-search (>4096 bytes) without instrumented bounded adapter".to_string(),
        ));
    }

    let mut dfa_searches = 0u32;
    let chunk_size = POLL_INTERVAL_WORK_UNITS as usize;

    // Linear-time forward DFA search
    let mut fwd_cache = plan.fwd_dfa.create_cache();
    let fwd_input = Input::new(input_text);

    let mut offset = 0;
    while offset < input_text.len() {
        let chunk = (input_text.len() - offset).min(chunk_size);
        tracker.charge_scanned(chunk as u64)?;
        tracker.charge_transitions(chunk as u64)?;
        offset += chunk;
    }

    dfa_searches = dfa_searches.saturating_add(1);
    let half_match = plan
        .fwd_dfa
        .try_search_fwd(&mut fwd_cache, &fwd_input)
        .map_err(|e| RegexError::SyntaxError(e.to_string()))?;

    tracker.poll()?;

    let hm = match half_match {
        Some(m) => m,
        None => return Ok((None, RegexFindStats { dfa_searches, poll_count: tracker.poll_count })),
    };

    let end = hm.offset();
    if end == 0 {
        tracker.charge_capture_updates(1)?;
        return Ok((
            Some(RegexMatch {
                start: 0,
                end: 0,
                text: String::new(),
            }),
            RegexFindStats { dfa_searches, poll_count: tracker.poll_count },
        ));
    }

    // Linear-time reverse DFA search: reverse search work is charged BEFORE running reverse search
    let matched_len = end as u64;
    let mut rev_offset = 0u64;
    while rev_offset < matched_len {
        let chunk = (matched_len - rev_offset).min(POLL_INTERVAL_WORK_UNITS);
        tracker.charge_scanned(chunk)?;
        tracker.charge_transitions(chunk)?;
        rev_offset += chunk;
    }

    if tracker.cancellation.is_some() && end > POLL_INTERVAL_WORK_UNITS as usize {
        return Err(RegexError::UnsupportedBackendFeature(
            "Pinned regex-automata API cannot expose true work quanta mid-search (>4096 bytes) without instrumented bounded adapter".to_string(),
        ));
    }

    let mut rev_cache = plan.rev_dfa.create_cache();
    let rev_input = Input::new(input_text)
        .range(..end)
        .anchored(regex_automata::Anchored::Yes);

    dfa_searches = dfa_searches.saturating_add(1);
    let rev_match = plan
        .rev_dfa
        .try_search_rev(&mut rev_cache, &rev_input)
        .map_err(|e| RegexError::SyntaxError(e.to_string()))?;

    tracker.poll()?;
    tracker.charge_capture_updates(1)?;

    let start = match rev_match {
        Some(m) => m.offset(),
        None => 0,
    };

    let matched_text = &input_text[start..end];
    Ok((
        Some(RegexMatch {
            start,
            end,
            text: matched_text.to_string(),
        }),
        RegexFindStats { dfa_searches, poll_count: tracker.poll_count },
    ))
}

pub fn replace_all(
    plan: &RegexPlan,
    input_text: &str,
    replacement: &str,
) -> Result<String, RegexError> {
    let max_matches = input_text.len().saturating_add(1);
    let max_output_estimate = input_text.len().saturating_add(max_matches.saturating_mul(replacement.len()));
    let mut tracker = RegexWorkTracker::new(input_text.len(), plan.group_count, max_output_estimate)?;
    replace_all_with_tracker(plan, input_text, replacement, &mut tracker)
}

pub fn replace_all_with_tracker(
    plan: &RegexPlan,
    input_text: &str,
    replacement: &str,
    tracker: &mut RegexWorkTracker,
) -> Result<String, RegexError> {
    tracker.poll()?;
    if tracker.cancellation.is_some() && input_text.len() > POLL_INTERVAL_WORK_UNITS as usize {
        return Err(RegexError::UnsupportedBackendFeature(
            "Pinned regex-automata API cannot expose true work quanta mid-search (>4096 bytes) without instrumented bounded adapter".to_string(),
        ));
    }

    let mut fwd_cache = plan.fwd_dfa.create_cache();
    let mut rev_cache = plan.rev_dfa.create_cache();
    let mut out = String::new();
    let mut last_idx = 0;
    let mut pos = 0;

    while pos <= input_text.len() {
        if tracker.cancellation.is_some() {
            tracker.poll()?;
        }

        let remaining_len = input_text.len() - pos;
        let remaining_budget = tracker.max_budget.saturating_sub(tracker.total_work());
        let max_affordable_bytes = (remaining_budget / 2) as usize;

        if remaining_len > 0 && max_affordable_bytes == 0 {
            tracker.charge_scanned(1)?;
            tracker.charge_transitions(1)?;
        }

        let window_end = if remaining_len > max_affordable_bytes {
            let mut boundary = pos + max_affordable_bytes;
            while boundary > pos && !input_text.is_char_boundary(boundary) {
                boundary -= 1;
            }
            if boundary == pos {
                let needed = remaining_len as u64;
                tracker.charge_scanned(needed)?;
                tracker.charge_transitions(needed)?;
            }
            boundary
        } else {
            input_text.len()
        };

        let fwd_input = Input::new(input_text).range(pos..window_end);
        let half_match = plan
            .fwd_dfa
            .try_search_fwd(&mut fwd_cache, &fwd_input)
            .map_err(|e| RegexError::SyntaxError(e.to_string()))?;

        tracker.poll()?;

        let hm = match half_match {
            Some(m) => m,
            None => {
                if window_end < input_text.len() {
                    let needed = (window_end - pos + 1) as u64;
                    tracker.charge_scanned(needed)?;
                    tracker.charge_transitions(needed)?;
                }
                let fwd_scanned = (input_text.len() - pos) as u64;
                tracker.charge_scanned(fwd_scanned)?;
                tracker.charge_transitions(fwd_scanned)?;
                break;
            }
        };

        let end = hm.offset();
        if window_end < input_text.len() && end == window_end {
            let needed = (window_end - pos + 1) as u64;
            tracker.charge_scanned(needed)?;
            tracker.charge_transitions(needed)?;
        }
        let fwd_scanned = (end - pos) as u64;
        tracker.charge_scanned(fwd_scanned)?;
        tracker.charge_transitions(fwd_scanned)?;

        let start = if end == pos {
            pos
        } else {
            let rev_scanned = (end - pos) as u64;
            tracker.charge_scanned(rev_scanned)?;
            tracker.charge_transitions(rev_scanned)?;

            let rev_input = Input::new(input_text)
                .range(pos..end)
                .anchored(regex_automata::Anchored::Yes);

            let rev_match = plan
                .rev_dfa
                .try_search_rev(&mut rev_cache, &rev_input)
                .map_err(|e| RegexError::SyntaxError(e.to_string()))?;

            tracker.poll()?;
            match rev_match {
                Some(m) => m.offset(),
                None => pos,
            }
        };

        tracker.charge_capture_updates(1)?;

        let prefix = &input_text[last_idx..start];
        out.push_str(prefix);
        tracker.charge_copied(prefix.len() as u64)?;

        out.push_str(replacement);
        tracker.charge_copied(replacement.len() as u64)?;

        last_idx = end;
        if end > pos {
            pos = end;
        } else {
            // Empty match: advance by 1 Unicode scalar character
            if pos < input_text.len() {
                let ch_len = input_text[pos..].chars().next().map(|c| c.len_utf8()).unwrap_or(1);
                let next_pos = pos + ch_len;
                let ch_slice = &input_text[pos..next_pos];
                out.push_str(ch_slice);
                tracker.charge_copied(ch_len as u64)?;
                last_idx = next_pos;
                pos = next_pos;
            } else {
                break;
            }
        }
    }

    if last_idx < input_text.len() {
        let suffix = &input_text[last_idx..];
        out.push_str(suffix);
        tracker.charge_copied(suffix.len() as u64)?;
    }

    Ok(out)
}

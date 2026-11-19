//! Wave 4 Track 4G (`D-LIFE`) static lifetime, region, and borrow-provenance analysis.
//!
//! Enforces `docs/RFC_WAVE4_4G_LIFETIME_CONTRACT.md` over the existing AST cleanly inside
//! `src/typechecker/*` while preserving `ADR-003` owned-value semantics for non-borrow code.

use alloc::{
    collections::{BTreeMap, BTreeSet},
    format,
    string::{String, ToString},
    vec::Vec,
};
use core::cmp::Ordering;
use hashbrown::{HashMap, HashSet};

use crate::{
    ast::*,
    error::{LustError, Result},
    modules::LoadedModule,
};

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum LifetimeDiagCode {
    E4G003,
    E4G002,
    E4G001,
    E4G004,
    E4G005,
    E4G006,
}

impl LifetimeDiagCode {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::E4G001 => "E4G001",
            Self::E4G002 => "E4G002",
            Self::E4G003 => "E4G003",
            Self::E4G004 => "E4G004",
            Self::E4G005 => "E4G005",
            Self::E4G006 => "E4G006",
        }
    }

    pub fn primary_message(self) -> &'static str {
        match self {
            Self::E4G001 => "borrowed value escapes its owner region",
            Self::E4G002 => "borrowed value captured by an escaping closure",
            Self::E4G003 => "borrowed value crosses a suspension or task boundary",
            Self::E4G004 => "owner mutation or destruction conflicts with a live borrow",
            Self::E4G005 => "borrow passed to a call without a proven safe lifetime summary",
            Self::E4G006 => "borrow provenance or lifetime cannot be established",
        }
    }

    pub fn precedence_rank(self) -> u8 {
        match self {
            Self::E4G003 => 0,
            Self::E4G002 => 1,
            Self::E4G001 => 2,
            Self::E4G004 => 3,
            Self::E4G005 => 4,
            Self::E4G006 => 5,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LifetimeDiagnostic {
    pub code: LifetimeDiagCode,
    pub detail: String,
    pub span: Span,
}

impl LifetimeDiagnostic {
    pub fn into_lust_error(self, module: &Option<String>) -> LustError {
        let message = format!(
            "{}: {} ({})",
            self.code.as_str(),
            self.code.primary_message(),
            self.detail
        );
        if self.span.start_line > 0 {
            LustError::TypeErrorWithSpan {
                message,
                line: self.span.start_line,
                column: self.span.start_col,
                module: module.clone(),
            }
        } else {
            LustError::TypeError { message }
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct BindingId(pub usize);

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct OwnerId(pub usize);

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct RegionId(pub usize);

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct LoanId(pub usize);

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BorrowProducerSpec {
    pub owner_arg_idx: usize,
    pub mutable: bool,
}

/// Registry describing explicitly audited borrow-producing, deep-materializing,
/// synchronous-safe, and suspending APIs.
#[derive(Debug, Clone)]
pub struct LifetimeRegistry {
    pub borrow_producers: HashMap<String, BorrowProducerSpec>,
    pub deep_materializers: HashSet<String>,
    pub sync_read_safe_callees: HashSet<String>,
    pub suspending_callees: HashSet<String>,
    pub drop_callees: HashSet<String>,
}

impl Default for LifetimeRegistry {
    fn default() -> Self {
        let mut sync_read_safe_callees = HashSet::new();
        for name in [
            "print",
            "println",
            "tostring",
            "assert",
            "assert_eq",
            "typeof",
        ] {
            sync_read_safe_callees.insert(name.to_string());
        }

        let mut suspending_callees = HashSet::new();
        for name in [
            "spawn",
            "await",
            "yield",
            "sleep",
            "task.spawn",
            "task.await",
            "task.sleep",
            "coord.send",
            "coord.recv",
            "coord.wait",
            "perform",
            "__lust_perform",
        ] {
            suspending_callees.insert(name.to_string());
        }

        let mut drop_callees = HashSet::new();
        for name in ["drop", "release", "close", "vm_drop", "destroy"] {
            drop_callees.insert(name.to_string());
        }

        Self {
            borrow_producers: HashMap::new(),
            deep_materializers: HashSet::new(),
            sync_read_safe_callees,
            suspending_callees,
            drop_callees,
        }
    }
}

impl LifetimeRegistry {
    pub fn register_borrow_producer(
        &mut self,
        callee: impl Into<String>,
        owner_arg_idx: usize,
        mutable: bool,
    ) {
        self.borrow_producers.insert(
            callee.into(),
            BorrowProducerSpec {
                owner_arg_idx,
                mutable,
            },
        );
    }

    pub fn register_deep_materializer(&mut self, callee: impl Into<String>) {
        self.deep_materializers.insert(callee.into());
    }

    pub fn register_suspending(&mut self, callee: impl Into<String>) {
        self.suspending_callees.insert(callee.into());
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct Loan {
    id: LoanId,
    owner: OwnerId,
    origin_region: RegionId,
    mutable: bool,
    span: Span,
}

#[derive(Debug, Clone, PartialEq, Eq)]
enum Provenance {
    Owned(OwnerId),
    Borrowed {
        loans: BTreeSet<LoanId>,
        owners: BTreeSet<OwnerId>,
    },
    UnresolvedBorrow,
}

impl Provenance {
    fn is_borrowed(&self) -> bool {
        !matches!(self, Self::Owned(_))
    }

    fn union(&self, other: &Self, fresh_owner: OwnerId) -> Self {
        match (self, other) {
            (Self::UnresolvedBorrow, _) | (_, Self::UnresolvedBorrow) => Self::UnresolvedBorrow,
            (Self::Owned(a), Self::Owned(b)) if a == b => Self::Owned(*a),
            (Self::Owned(_), Self::Owned(_)) => Self::Owned(fresh_owner),
            _ => {
                let mut loans = BTreeSet::new();
                let mut owners = BTreeSet::new();
                for src in [self, other] {
                    if let Self::Borrowed { loans: l, owners: o } = src {
                        loans.extend(l.iter().copied());
                        owners.extend(o.iter().copied());
                    }
                }
                Self::Borrowed { loans, owners }
            }
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct BindingInfo {
    id: BindingId,
    region: RegionId,
    owner: OwnerId,
    provenance: Provenance,
    aliased_owners: BTreeSet<OwnerId>,
    fn_targets: BTreeSet<String>,
    declared_ref: bool,
}

#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct FnSummary {
    pub param_count: usize,
    pub escapes_params_to_return: BTreeSet<usize>,
    pub escapes_params_to_global: BTreeSet<usize>,
    pub mutates_params: BTreeSet<usize>,
    pub captures_params: BTreeSet<usize>,
    pub unproven_params: BTreeSet<usize>,
    pub may_suspend: bool,
    pub returns_unresolved_borrow: bool,
}

impl FnSummary {
    fn is_safe_sync_readonly_for_arg(&self, arg_idx: usize) -> bool {
        !self.may_suspend
            && !self.returns_unresolved_borrow
            && !self.escapes_params_to_return.contains(&arg_idx)
            && !self.escapes_params_to_global.contains(&arg_idx)
            && !self.mutates_params.contains(&arg_idx)
            && !self.captures_params.contains(&arg_idx)
            && !self.unproven_params.contains(&arg_idx)
    }

    fn is_safe_indirect_readonly(&self) -> bool {
        !self.may_suspend
            && !self.returns_unresolved_borrow
            && self.escapes_params_to_return.is_empty()
            && self.escapes_params_to_global.is_empty()
            && self.mutates_params.is_empty()
            && self.captures_params.is_empty()
            && self.unproven_params.is_empty()
    }
}

#[derive(Debug, Clone, PartialEq)]
struct RegionFrame {
    id: RegionId,
    bindings: BTreeMap<String, BindingId>,
    owned_bindings: Vec<BindingId>,
    defers: Vec<Stmt>,
}

#[derive(Debug, Clone, PartialEq)]
struct FlowState {
    bindings: BTreeMap<BindingId, BindingInfo>,
    loans: BTreeMap<LoanId, Loan>,
    regions: Vec<RegionFrame>,
}

impl FlowState {
    fn current_region(&self) -> RegionId {
        self.regions.last().map(|r| r.id).unwrap_or(RegionId(0))
    }

    fn resolve(&self, name: &str) -> Option<&BindingInfo> {
        for frame in self.regions.iter().rev() {
            if let Some(bid) = frame.bindings.get(name) {
                return self.bindings.get(bid);
            }
        }
        None
    }

    fn resolve_id(&self, name: &str) -> Option<BindingId> {
        for frame in self.regions.iter().rev() {
            if let Some(&bid) = frame.bindings.get(name) {
                return Some(bid);
            }
        }
        None
    }

    fn join(&mut self, other: &Self, id_gen: &mut IdGen) {
        for (bid, o_info) in &other.bindings {
            if let Some(s_info) = self.bindings.get_mut(bid) {
                let fresh = id_gen.next_owner();
                s_info.provenance = s_info.provenance.union(&o_info.provenance, fresh);
                s_info
                    .aliased_owners
                    .extend(o_info.aliased_owners.iter().copied());
                s_info.fn_targets.extend(o_info.fn_targets.iter().cloned());
            } else {
                self.bindings.insert(*bid, o_info.clone());
            }
        }
        for (lid, loan) in &other.loans {
            self.loans.entry(*lid).or_insert_with(|| loan.clone());
        }
    }
}

struct IdGen {
    next_binding: usize,
    next_owner: usize,
    next_region: usize,
    next_loan: usize,
}

impl IdGen {
    fn new() -> Self {
        Self {
            next_binding: 1,
            next_owner: 1,
            next_region: 1,
            next_loan: 1,
        }
    }

    fn next_binding(&mut self) -> BindingId {
        let id = BindingId(self.next_binding);
        self.next_binding += 1;
        id
    }

    fn next_owner(&mut self) -> OwnerId {
        let id = OwnerId(self.next_owner);
        self.next_owner += 1;
        id
    }

    fn next_region(&mut self) -> RegionId {
        let id = RegionId(self.next_region);
        self.next_region += 1;
        id
    }

    fn next_loan(&mut self) -> LoanId {
        let id = LoanId(self.next_loan);
        self.next_loan += 1;
        id
    }
}

fn type_contains_ref(ty: &Type) -> Option<(bool, Span)> {
    match &ty.kind {
        TypeKind::Ref(_) => Some((false, ty.span)),
        TypeKind::MutRef(_) => Some((true, ty.span)),
        TypeKind::Array(inner) | TypeKind::Option(inner) => type_contains_ref(inner),
        TypeKind::Result(ok, err) => type_contains_ref(ok).or_else(|| type_contains_ref(err)),
        TypeKind::Tuple(elems) => elems.iter().find_map(type_contains_ref),
        TypeKind::GenericInstance { type_args, .. } => {
            type_args.iter().find_map(type_contains_ref)
        }
        _ => None,
    }
}

fn expr_root_name(expr: &Expr) -> Option<&str> {
    match &expr.kind {
        ExprKind::Identifier(name) => Some(name.as_str()),
        ExprKind::FieldAccess { object, .. } | ExprKind::Index { object, .. } => {
            expr_root_name(object)
        }
        ExprKind::Paren(inner) => expr_root_name(inner),
        _ => None,
    }
}

fn callee_name(expr: &Expr) -> Option<String> {
    match &expr.kind {
        ExprKind::Identifier(name) => Some(name.clone()),
        ExprKind::FieldAccess { object, field } => {
            callee_name(object).map(|prefix| format!("{prefix}.{field}"))
        }
        ExprKind::Paren(inner) => callee_name(inner),
        _ => None,
    }
}

fn is_immediate_lambda_callee(expr: &Expr) -> Option<(&Vec<(String, Option<Type>)>, &Expr)> {
    match &expr.kind {
        ExprKind::Lambda { params, body, .. } => Some((params, body)),
        ExprKind::Paren(inner) => is_immediate_lambda_callee(inner),
        _ => None,
    }
}

#[derive(Debug, Clone)]
struct EvalValue {
    provenance: Provenance,
    fn_targets: BTreeSet<String>,
    origin_owner: Option<OwnerId>,
    touched_owners: BTreeSet<OwnerId>,
}

impl EvalValue {
    fn owned(owner: OwnerId) -> Self {
        let mut touched = BTreeSet::new();
        touched.insert(owner);
        Self {
            provenance: Provenance::Owned(owner),
            fn_targets: BTreeSet::new(),
            origin_owner: Some(owner),
            touched_owners: touched,
        }
    }

    fn union(&self, other: &Self, fresh_owner: OwnerId) -> Self {
        let mut fn_targets = self.fn_targets.clone();
        fn_targets.extend(other.fn_targets.iter().cloned());
        let mut touched_owners = self.touched_owners.clone();
        touched_owners.extend(other.touched_owners.iter().copied());
        Self {
            provenance: self.provenance.union(&other.provenance, fresh_owner),
            fn_targets,
            origin_owner: None,
            touched_owners,
        }
    }
}

struct Analyzer<'a> {
    registry: &'a LifetimeRegistry,
    fn_defs: &'a BTreeMap<String, FunctionDef>,
    summaries: &'a BTreeMap<String, FnSummary>,
    module_name: Option<String>,
    id_gen: IdGen,
    diagnostics: Vec<LifetimeDiagnostic>,
    active_param_bindings: HashMap<BindingId, usize>,
    in_function_name: Option<String>,
}

impl<'a> Analyzer<'a> {
    fn new(
        registry: &'a LifetimeRegistry,
        fn_defs: &'a BTreeMap<String, FunctionDef>,
        summaries: &'a BTreeMap<String, FnSummary>,
        module_name: Option<String>,
    ) -> Self {
        Self {
            registry,
            fn_defs,
            summaries,
            module_name,
            id_gen: IdGen::new(),
            diagnostics: Vec::new(),
            active_param_bindings: HashMap::new(),
            in_function_name: None,
        }
    }

    fn emit(&mut self, code: LifetimeDiagCode, detail: impl Into<String>, span: Span) {
        self.diagnostics.push(LifetimeDiagnostic {
            code,
            detail: detail.into(),
            span,
        });
    }

    fn record_highest_diagnostic(self) -> Result<()> {
        if let Some(best) = self.diagnostics.into_iter().min_by(|a, b| {
            match a.code.precedence_rank().cmp(&b.code.precedence_rank()) {
                Ordering::Equal => a.span.start_line.cmp(&b.span.start_line),
                ord => ord,
            }
        }) {
            return Err(best.into_lust_error(&self.module_name));
        }
        Ok(())
    }

    fn push_region(&mut self, state: &mut FlowState) -> RegionId {
        let rid = self.id_gen.next_region();
        state.regions.push(RegionFrame {
            id: rid,
            bindings: BTreeMap::new(),
            owned_bindings: Vec::new(),
            defers: Vec::new(),
        });
        rid
    }

    fn run_scope_defers(&mut self, state: &mut FlowState, up_to_idx: usize) {
        let mut defer_batches = Vec::new();
        for idx in (up_to_idx..state.regions.len()).rev() {
            if !state.regions[idx].defers.is_empty() {
                defer_batches.push(state.regions[idx].defers.clone());
            }
        }
        for batch in defer_batches {
            for stmt in batch.iter().rev() {
                self.check_stmt(state, stmt);
            }
        }
    }

    fn pop_region(&mut self, state: &mut FlowState, span: Span) {
        let top_idx = state.regions.len().saturating_sub(1);
        self.run_scope_defers(state, top_idx);
        let Some(frame) = state.regions.pop() else {
            return;
        };
        let mut dying_owners = BTreeSet::new();
        for bid in &frame.owned_bindings {
            if let Some(info) = state.bindings.get(bid) {
                dying_owners.insert(info.owner);
                dying_owners.extend(info.aliased_owners.iter().copied());
            }
        }
        let popped_region = frame.id;
        // Retire loans created strictly inside this region.
        state.loans.retain(|_, loan| loan.origin_region < popped_region);

        // Any still-live loan targeting a dying owner is an escape/conflict.
        let escaped_owners: Vec<(OwnerId, Span)> = state
            .loans
            .values()
            .filter(|loan| dying_owners.contains(&loan.owner))
            .map(|loan| (loan.owner, loan.span))
            .collect();
        for (_, loan_span) in escaped_owners {
            self.emit(
                LifetimeDiagCode::E4G004,
                "owner destroyed at scope exit while borrowed by outer region",
                if loan_span.start_line > 0 {
                    loan_span
                } else {
                    span
                },
            );
        }
        for bid in frame.owned_bindings {
            state.bindings.remove(&bid);
        }
    }

    fn declare_binding(
        &mut self,
        state: &mut FlowState,
        name: &str,
        provenance: Provenance,
        fn_targets: BTreeSet<String>,
        aliased_owners: BTreeSet<OwnerId>,
        declared_ref: bool,
    ) -> BindingId {
        let bid = self.id_gen.next_binding();
        let owner = self.id_gen.next_owner();
        let region = state.current_region();
        state.bindings.insert(
            bid,
            BindingInfo {
                id: bid,
                region,
                owner,
                provenance,
                aliased_owners,
                fn_targets,
                declared_ref,
            },
        );
        if let Some(frame) = state.regions.last_mut() {
            frame.bindings.insert(name.to_string(), bid);
            frame.owned_bindings.push(bid);
        }
        bid
    }

    fn active_loans_for_owner(&self, state: &FlowState, owner: OwnerId) -> Vec<Loan> {
        state
            .loans
            .values()
            .filter(|l| l.owner == owner)
            .cloned()
            .collect()
    }

    fn check_owner_mutation(&mut self, state: &FlowState, target_name: &str, span: Span) {
        let Some(info) = state.resolve(target_name).cloned() else {
            return;
        };
        let mut target_owners = info.aliased_owners.clone();
        target_owners.insert(info.owner);
        if let Provenance::Owned(o) = info.provenance {
            target_owners.insert(o);
        }
        for owner in target_owners {
            if !self.active_loans_for_owner(state, owner).is_empty() {
                self.emit(
                    LifetimeDiagCode::E4G004,
                    format!("cannot mutate or drop `{target_name}` while borrowed"),
                    span,
                );
                return;
            }
        }
        if let Provenance::Borrowed { loans, .. } = &info.provenance {
            // Re-assigning a mutably borrowed local or mutating through shared borrow conflicts cleanly
            if !loans.is_empty() {
                let any_shared = loans
                    .iter()
                    .filter_map(|lid| state.loans.get(lid))
                    .any(|l| !l.mutable);
                if any_shared {
                    self.emit(
                        LifetimeDiagCode::E4G004,
                        format!("cannot mutate `{target_name}` through a shared borrow"),
                        span,
                    );
                }
            }
        }
    }

    fn check_borrow_suspension(&mut self, state: &FlowState, callee: &str, args: &[EvalValue], span: Span) {
        let any_arg_borrowed = args.iter().any(|a| a.provenance.is_borrowed());
        let any_live_loan = !state.loans.is_empty();
        if any_arg_borrowed || any_live_loan {
            self.emit(
                LifetimeDiagCode::E4G003,
                format!("live borrow cannot cross suspension/task call `{callee}`"),
                span,
            );
        }
    }

    fn check_items(&mut self, state: &mut FlowState, items: &[Item]) {
        for item in items {
            match &item.kind {
                ItemKind::Script(stmts) => {
                    for stmt in stmts {
                        self.check_stmt(state, stmt);
                    }
                }
                ItemKind::Function(func) => {
                    self.check_function(state, func);
                }
                ItemKind::Impl(impl_block) => {
                    for method in &impl_block.methods {
                        self.check_function(state, method);
                    }
                }
                ItemKind::Module { items: nested, .. } => {
                    self.check_items(state, nested);
                }
                ItemKind::Test(t) => {
                    self.push_region(state);
                    for stmt in &t.body {
                        self.check_stmt(state, stmt);
                    }
                    self.pop_region(state, t.span);
                }
                ItemKind::Suite(s) => {
                    self.check_suite(state, s);
                }
                _ => {}
            }
        }
    }

    fn check_suite(&mut self, state: &mut FlowState, suite: &SuiteDecl) {
        for hook in [
            suite.hooks.before_all.as_ref(),
            suite.hooks.before_each.as_ref(),
            suite.hooks.after_each.as_ref(),
            suite.hooks.after_all.as_ref(),
        ]
        .into_iter()
        .flatten()
        {
            self.push_region(state);
            for stmt in hook {
                self.check_stmt(state, stmt);
            }
            self.pop_region(state, suite.span);
        }
        for t in &suite.tests {
            self.push_region(state);
            for stmt in &t.body {
                self.check_stmt(state, stmt);
            }
            self.pop_region(state, t.span);
        }
        for nested in &suite.nested_suites {
            self.check_suite(state, nested);
        }
    }

    fn check_function(&mut self, outer_state: &FlowState, func: &FunctionDef) {
        let mut fn_state = FlowState {
            bindings: outer_state.bindings.clone(),
            loans: BTreeMap::new(),
            regions: vec![outer_state.regions[0].clone()],
        };
        self.push_region(&mut fn_state);
        self.active_param_bindings.clear();
        let prev_fn = self.in_function_name.replace(func.name.clone());

        for (idx, param) in func.params.iter().enumerate() {
            let is_ref = type_contains_ref(&param.ty).is_some();
            let owner = self.id_gen.next_owner();
            let provenance = if is_ref {
                let lid = self.id_gen.next_loan();
                let mutable = matches!(param.ty.kind, TypeKind::MutRef(_));
                let region = fn_state.current_region();
                fn_state.loans.insert(
                    lid,
                    Loan {
                        id: lid,
                        owner,
                        origin_region: region,
                        mutable,
                        span: param.ty.span,
                    },
                );
                let mut loans = BTreeSet::new();
                loans.insert(lid);
                let mut owners = BTreeSet::new();
                owners.insert(owner);
                Provenance::Borrowed { loans, owners }
            } else {
                Provenance::Owned(owner)
            };
            let bid = self.declare_binding(
                &mut fn_state,
                &param.name,
                provenance,
                BTreeSet::new(),
                BTreeSet::new(),
                is_ref,
            );
            self.active_param_bindings.insert(bid, idx);
        }

        let body_len = func.body.len();
        for (idx, stmt) in func.body.iter().enumerate() {
            let is_last = idx + 1 == body_len;
            if is_last && matches!(stmt.kind, StmtKind::Expr(_)) && func.return_type.is_some() {
                if let StmtKind::Expr(expr) = &stmt.kind {
                    let val = self.check_expr(&mut fn_state, expr);
                    if val.provenance.is_borrowed() {
                        self.emit(
                            LifetimeDiagCode::E4G001,
                            "implicit tail return cannot escape borrowed value",
                            expr.span,
                        );
                    }
                    continue;
                }
            }
            self.check_stmt(&mut fn_state, stmt);
        }

        self.pop_region(&mut fn_state, Span::dummy());
        self.active_param_bindings.clear();
        self.in_function_name = prev_fn;
    }

    fn check_stmt(&mut self, state: &mut FlowState, stmt: &Stmt) {
        match &stmt.kind {
            StmtKind::Local {
                bindings,
                initializer,
            } => {
                let mut evals = Vec::new();
                if let Some(inits) = initializer {
                    for expr in inits {
                        evals.push(self.check_expr(state, expr));
                    }
                }
                for (i, binding) in bindings.iter().enumerate() {
                    let ref_ann = binding
                        .type_annotation
                        .as_ref()
                        .and_then(type_contains_ref);
                    let init_val = evals.get(i).cloned();
                    let (provenance, fn_targets, aliased_owners) = match (ref_ann, init_val) {
                        (Some(_), Some(ev)) => match ev.provenance {
                            Provenance::Borrowed { .. } => (
                                ev.provenance,
                                ev.fn_targets,
                                ev.touched_owners,
                            ),
                            Provenance::Owned(_) | Provenance::UnresolvedBorrow => {
                                self.emit(
                                    LifetimeDiagCode::E4G006,
                                    format!(
                                        "local `{}` annotated as reference without audited borrow producer",
                                        binding.name
                                    ),
                                    binding.span,
                                );
                                (Provenance::UnresolvedBorrow, BTreeSet::new(), BTreeSet::new())
                            }
                        },
                        (Some(_), None) => {
                            self.emit(
                                LifetimeDiagCode::E4G006,
                                format!("uninitialized reference local `{}`", binding.name),
                                binding.span,
                            );
                            (Provenance::UnresolvedBorrow, BTreeSet::new(), BTreeSet::new())
                        }
                        (None, Some(ev)) => (ev.provenance, ev.fn_targets, ev.touched_owners),
                        (None, None) => {
                            let fresh = self.id_gen.next_owner();
                            (Provenance::Owned(fresh), BTreeSet::new(), BTreeSet::new())
                        }
                    };
                    self.declare_binding(
                        state,
                        &binding.name,
                        provenance,
                        fn_targets,
                        aliased_owners,
                        ref_ann.is_some(),
                    );
                }
            }
            StmtKind::Assign { targets, values } => {
                let vals: Vec<EvalValue> = values.iter().map(|v| self.check_expr(state, v)).collect();
                for (i, target) in targets.iter().enumerate() {
                    let val = vals.get(i).cloned().unwrap_or_else(|| {
                        let o = self.id_gen.next_owner();
                        EvalValue::owned(o)
                    });
                    if let Some(name) = expr_root_name(target) {
                        self.check_owner_mutation(state, name, target.span);
                        if let Some(bid) = state.resolve_id(name) {
                            if let Some(dest_info) = state.bindings.get(&bid).cloned() {
                                if val.provenance.is_borrowed() {
                                    let mut borrowed_lids = Vec::new();
                                    let escapes = match &val.provenance {
                                        Provenance::Borrowed { loans, .. } => loans.iter().any(|lid| {
                                            borrowed_lids.push(*lid);
                                            if let Some(l) = state.loans.get(lid) {
                                                let owner_region = state
                                                    .bindings
                                                    .values()
                                                    .find(|b| b.owner == l.owner)
                                                    .map(|b| b.region)
                                                    .unwrap_or(l.origin_region);
                                                dest_info.region < owner_region
                                                    || dest_info.region == RegionId(0)
                                            } else {
                                                true
                                            }
                                        }),
                                        Provenance::UnresolvedBorrow => true,
                                        Provenance::Owned(_) => false,
                                    };
                                    if escapes {
                                        self.emit(
                                            LifetimeDiagCode::E4G001,
                                            format!("borrowed value assigned to longer-lived target `{name}`"),
                                            target.span,
                                        );
                                    } else {
                                        for lid in borrowed_lids {
                                            if let Some(loan) = state.loans.get_mut(&lid) {
                                                if dest_info.region < loan.origin_region {
                                                    loan.origin_region = dest_info.region;
                                                }
                                            }
                                        }
                                    }
                                }
                                if let Some(entry) = state.bindings.get_mut(&bid) {
                                    entry.provenance = val.provenance.clone();
                                    entry.fn_targets = val.fn_targets.clone();
                                    entry.aliased_owners = val.touched_owners.clone();
                                }
                            }
                        } else if val.provenance.is_borrowed() {
                            self.emit(
                                LifetimeDiagCode::E4G001,
                                format!("borrowed value stored into global `{name}`"),
                                target.span,
                            );
                        }
                    } else if val.provenance.is_borrowed() {
                        self.emit(
                            LifetimeDiagCode::E4G001,
                            "borrowed value assigned to non-local place",
                            target.span,
                        );
                    }
                }
            }
            StmtKind::CompoundAssign { target, value, .. } => {
                let _ = self.check_expr(state, value);
                if let Some(name) = expr_root_name(target) {
                    self.check_owner_mutation(state, name, target.span);
                }
            }
            StmtKind::Expr(expr) => {
                let _ = self.check_expr(state, expr);
            }
            StmtKind::If {
                condition,
                then_block,
                elseif_branches,
                else_block,
            } => {
                let _ = self.check_expr(state, condition);
                let base = state.clone();

                let mut then_state = base.clone();
                self.push_region(&mut then_state);
                for s in then_block {
                    self.check_stmt(&mut then_state, s);
                }
                self.pop_region(&mut then_state, stmt.span);
                state.join(&then_state, &mut self.id_gen);

                for (cond, branch) in elseif_branches {
                    let _ = self.check_expr(state, cond);
                    let mut b_state = base.clone();
                    self.push_region(&mut b_state);
                    for s in branch {
                        self.check_stmt(&mut b_state, s);
                    }
                    self.pop_region(&mut b_state, stmt.span);
                    state.join(&b_state, &mut self.id_gen);
                }

                if let Some(eb) = else_block {
                    let mut e_state = base;
                    self.push_region(&mut e_state);
                    for s in eb {
                        self.check_stmt(&mut e_state, s);
                    }
                    self.pop_region(&mut e_state, stmt.span);
                    state.join(&e_state, &mut self.id_gen);
                }
            }
            StmtKind::While { condition, body } => {
                self.check_loop_fixed_point(state, Some(condition), body, stmt.span);
            }
            StmtKind::ForNumeric {
                variable,
                start,
                end,
                step,
                body,
            } => {
                let _ = self.check_expr(state, start);
                let _ = self.check_expr(state, end);
                if let Some(st) = step {
                    let _ = self.check_expr(state, st);
                }
                let o = self.id_gen.next_owner();
                self.push_region(state);
                self.declare_binding(
                    state,
                    variable,
                    Provenance::Owned(o),
                    BTreeSet::new(),
                    BTreeSet::new(),
                    false,
                );
                self.check_loop_fixed_point(state, None, body, stmt.span);
                self.pop_region(state, stmt.span);
            }
            StmtKind::ForIn {
                variables,
                iterator,
                body,
            } => {
                let iter_val = self.check_expr(state, iterator);
                self.push_region(state);
                for var in variables {
                    self.declare_binding(
                        state,
                        var,
                        iter_val.provenance.clone(),
                        BTreeSet::new(),
                        iter_val.touched_owners.clone(),
                        false,
                    );
                }
                self.check_loop_fixed_point(state, None, body, stmt.span);
                self.pop_region(state, stmt.span);
            }
            StmtKind::Return(values) => {
                let vals: Vec<EvalValue> = values.iter().map(|v| self.check_expr(state, v)).collect();
                self.run_scope_defers(state, 1);
                for (expr, val) in values.iter().zip(vals.iter()) {
                    if val.provenance.is_borrowed() {
                        self.emit(
                            LifetimeDiagCode::E4G001,
                            "cannot return borrowed value from owner region",
                            expr.span,
                        );
                    }
                }
            }
            StmtKind::Break | StmtKind::Continue => {
                let top = state.regions.len().saturating_sub(1);
                self.run_scope_defers(state, top);
            }
            StmtKind::Block(stmts) | StmtKind::Scoped(stmts) => {
                self.push_region(state);
                for s in stmts {
                    self.check_stmt(state, s);
                }
                self.pop_region(state, stmt.span);
            }
            StmtKind::Defer(inner) | StmtKind::ErrDefer(inner) => {
                if let Some(frame) = state.regions.last_mut() {
                    frame.defers.push((**inner).clone());
                }
            }
        }
    }

    fn check_loop_fixed_point(
        &mut self,
        state: &mut FlowState,
        condition: Option<&Expr>,
        body: &[Stmt],
        span: Span,
    ) {
        const MAX_LOOP_ITERS: usize = 8;
        for _ in 0..MAX_LOOP_ITERS {
            let before = state.clone();
            if let Some(cond) = condition {
                let _ = self.check_expr(state, cond);
            }
            let mut iter_state = state.clone();
            self.push_region(&mut iter_state);
            for s in body {
                self.check_stmt(&mut iter_state, s);
            }
            self.pop_region(&mut iter_state, span);
            state.join(&iter_state, &mut self.id_gen);
            if *state == before {
                return;
            }
        }
    }

    fn check_expr(&mut self, state: &mut FlowState, expr: &Expr) -> EvalValue {
        match &expr.kind {
            ExprKind::Literal(_) | ExprKind::RegexLiteral { .. } => {
                let o = self.id_gen.next_owner();
                EvalValue::owned(o)
            }
            ExprKind::Identifier(name) => {
                if let Some(info) = state.resolve(name).cloned() {
                    let mut touched = info.aliased_owners.clone();
                    touched.insert(info.owner);
                    let mut fn_targets = info.fn_targets.clone();
                    if self.fn_defs.contains_key(name) {
                        fn_targets.insert(name.clone());
                    }
                    EvalValue {
                        provenance: info.provenance,
                        fn_targets,
                        origin_owner: Some(info.owner),
                        touched_owners: touched,
                    }
                } else {
                    let mut fn_targets = BTreeSet::new();
                    if self.fn_defs.contains_key(name) {
                        fn_targets.insert(name.clone());
                    }
                    let o = self.id_gen.next_owner();
                    EvalValue {
                        provenance: Provenance::Owned(o),
                        fn_targets,
                        origin_owner: Some(o),
                        touched_owners: BTreeSet::new(),
                    }
                }
            }
            ExprKind::Binary { left, right, .. } => {
                let _ = self.check_expr(state, left);
                let _ = self.check_expr(state, right);
                let o = self.id_gen.next_owner();
                EvalValue::owned(o)
            }
            ExprKind::Unary { operand, .. } => {
                let v = self.check_expr(state, operand);
                let o = self.id_gen.next_owner();
                if v.provenance.is_borrowed() {
                    v
                } else {
                    EvalValue::owned(o)
                }
            }
            ExprKind::Paren(inner) | ExprKind::Cast { expr: inner, .. } => {
                self.check_expr(state, inner)
            }
            ExprKind::TypeCheck { expr: inner, .. } | ExprKind::IsPattern { expr: inner, .. } => {
                let _ = self.check_expr(state, inner);
                let o = self.id_gen.next_owner();
                EvalValue::owned(o)
            }
            ExprKind::FieldAccess { object, .. } => {
                let obj = self.check_expr(state, object);
                if obj.provenance.is_borrowed() {
                    obj
                } else {
                    let o = self.id_gen.next_owner();
                    let mut out = EvalValue::owned(o);
                    out.touched_owners = obj.touched_owners;
                    out
                }
            }
            ExprKind::Index { object, index } => {
                let obj = self.check_expr(state, object);
                let _ = self.check_expr(state, index);
                if obj.provenance.is_borrowed() {
                    obj
                } else {
                    let o = self.id_gen.next_owner();
                    let mut out = EvalValue::owned(o);
                    out.touched_owners = obj.touched_owners;
                    out
                }
            }
            ExprKind::Array(elems) | ExprKind::Tuple(elems) => {
                let mut acc: Option<EvalValue> = None;
                for elem in elems {
                    let val = self.check_expr(state, elem);
                    acc = Some(match acc {
                        None => val,
                        Some(prev) => {
                            let f = self.id_gen.next_owner();
                            prev.union(&val, f)
                        }
                    });
                }
                acc.unwrap_or_else(|| EvalValue::owned(self.id_gen.next_owner()))
            }
            ExprKind::Map(entries) => {
                let mut acc: Option<EvalValue> = None;
                for (k, v) in entries {
                    let kv = self.check_expr(state, k);
                    let vv = self.check_expr(state, v);
                    let f1 = self.id_gen.next_owner();
                    let combined = kv.union(&vv, f1);
                    acc = Some(match acc {
                        None => combined,
                        Some(prev) => {
                            let f2 = self.id_gen.next_owner();
                            prev.union(&combined, f2)
                        }
                    });
                }
                acc.unwrap_or_else(|| EvalValue::owned(self.id_gen.next_owner()))
            }
            ExprKind::StructLiteral { fields, .. } => {
                let mut acc: Option<EvalValue> = None;
                for field in fields {
                    let val = self.check_expr(state, &field.value);
                    acc = Some(match acc {
                        None => val,
                        Some(prev) => {
                            let f = self.id_gen.next_owner();
                            prev.union(&val, f)
                        }
                    });
                }
                acc.unwrap_or_else(|| EvalValue::owned(self.id_gen.next_owner()))
            }
            ExprKind::EnumConstructor { args, .. } => {
                let mut acc: Option<EvalValue> = None;
                for arg in args {
                    let val = self.check_expr(state, arg);
                    acc = Some(match acc {
                        None => val,
                        Some(prev) => {
                            let f = self.id_gen.next_owner();
                            prev.union(&val, f)
                        }
                    });
                }
                acc.unwrap_or_else(|| EvalValue::owned(self.id_gen.next_owner()))
            }
            ExprKind::Lambda { params, body, .. } => {
                // Non-immediate lambda: cannot capture any live borrowed value from outer scope.
                let mut captured = BTreeSet::new();
                let mut bound: HashSet<String> = params.iter().map(|(n, _)| n.clone()).collect();
                collect_free_vars(body, &mut bound, &mut captured);
                for var in captured {
                    if let Some(info) = state.resolve(&var) {
                        if info.provenance.is_borrowed() {
                            self.emit(
                                LifetimeDiagCode::E4G002,
                                format!("closure captures borrowed variable `{var}` across escape boundary"),
                                expr.span,
                            );
                        }
                    }
                }
                let o = self.id_gen.next_owner();
                EvalValue::owned(o)
            }
            ExprKind::Call { callee, args, .. } => {
                if let Some((params, body)) = is_immediate_lambda_callee(callee) {
                    let arg_vals: Vec<EvalValue> =
                        args.iter().map(|a| self.check_expr(state, a)).collect();
                    self.push_region(state);
                    for (idx, (p_name, p_ty)) in params.iter().enumerate() {
                        let ref_ann = p_ty.as_ref().and_then(type_contains_ref).is_some();
                        let av = arg_vals.get(idx).cloned().unwrap_or_else(|| {
                            let o = self.id_gen.next_owner();
                            EvalValue::owned(o)
                        });
                        self.declare_binding(
                            state,
                            p_name,
                            av.provenance,
                            av.fn_targets,
                            av.touched_owners,
                            ref_ann,
                        );
                    }
                    let body_val = self.check_expr(state, body);
                    self.pop_region(state, expr.span);
                    return body_val;
                }

                let arg_vals: Vec<EvalValue> =
                    args.iter().map(|a| self.check_expr(state, a)).collect();
                let cname = callee_name(callee);
                if let Some(name) = cname.as_deref() {
                    return self.check_named_call(state, name, args, &arg_vals, expr.span);
                }

                // Dynamic callee expression (no direct name)
                let callee_val = self.check_expr(state, callee);
                self.check_indirect_call(state, &callee_val.fn_targets, &arg_vals, expr.span)
            }
            ExprKind::MethodCall {
                receiver,
                method,
                args,
                ..
            } => {
                let recv_val = self.check_expr(state, receiver);
                let arg_vals: Vec<EvalValue> =
                    args.iter().map(|a| self.check_expr(state, a)).collect();
                if self.registry.suspending_callees.contains(method.as_str()) {
                    let mut all_args = Vec::with_capacity(1 + arg_vals.len());
                    all_args.push(recv_val);
                    all_args.extend(arg_vals);
                    self.check_borrow_suspension(state, method, &all_args, expr.span);
                    return EvalValue::owned(self.id_gen.next_owner());
                }
                if recv_val.provenance.is_borrowed()
                    || arg_vals.iter().any(|a| a.provenance.is_borrowed())
                {
                    // Shallow .clone() or unknown method NEVER materializes owned copy of a borrow!
                    self.emit(
                        LifetimeDiagCode::E4G005,
                        format!("method call `{method}` lacks proven safe lifetime summary for borrow"),
                        expr.span,
                    );
                }
                EvalValue::owned(self.id_gen.next_owner())
            }
            ExprKind::If {
                condition,
                then_branch,
                else_branch,
            } => {
                let _ = self.check_expr(state, condition);
                let base = state.clone();
                let mut ts = base.clone();
                let tv = self.check_expr(&mut ts, then_branch);
                state.join(&ts, &mut self.id_gen);
                if let Some(eb) = else_branch {
                    let mut es = base;
                    let ev = self.check_expr(&mut es, eb);
                    state.join(&es, &mut self.id_gen);
                    let f = self.id_gen.next_owner();
                    tv.union(&ev, f)
                } else {
                    tv
                }
            }
            ExprKind::Block(stmts) | ExprKind::EffectBlock(stmts) => {
                self.push_region(state);
                let mut tail = EvalValue::owned(self.id_gen.next_owner());
                let len = stmts.len();
                for (i, s) in stmts.iter().enumerate() {
                    if i + 1 == len {
                        if let StmtKind::Expr(e) = &s.kind {
                            tail = self.check_expr(state, e);
                            continue;
                        }
                    }
                    self.check_stmt(state, s);
                }
                self.pop_region(state, expr.span);
                tail
            }
            ExprKind::Return(values) => {
                for v in values {
                    let ev = self.check_expr(state, v);
                    if ev.provenance.is_borrowed() {
                        self.emit(
                            LifetimeDiagCode::E4G001,
                            "cannot return borrowed value from expression return",
                            v.span,
                        );
                    }
                }
                EvalValue::owned(self.id_gen.next_owner())
            }
            ExprKind::Range { start, end, .. } => {
                let _ = self.check_expr(state, start);
                let _ = self.check_expr(state, end);
                EvalValue::owned(self.id_gen.next_owner())
            }
        }
    }

    fn check_named_call(
        &mut self,
        state: &mut FlowState,
        name: &str,
        arg_exprs: &[Expr],
        arg_vals: &[EvalValue],
        span: Span,
    ) -> EvalValue {
        if let Some(spec) = self.registry.borrow_producers.get(name) {
            if let Some(owner_arg) = arg_vals.get(spec.owner_arg_idx) {
                let owner_id = owner_arg
                    .origin_owner
                    .unwrap_or_else(|| self.id_gen.next_owner());
                // Check conflict with existing loans on this owner
                let existing = self.active_loans_for_owner(state, owner_id);
                let conflict = if spec.mutable {
                    !existing.is_empty()
                } else {
                    existing.iter().any(|l| l.mutable)
                };
                if conflict {
                    self.emit(
                        LifetimeDiagCode::E4G004,
                        format!("conflicting borrow on `{name}` while owner already borrowed"),
                        span,
                    );
                }
                let lid = self.id_gen.next_loan();
                let region = state.current_region();
                state.loans.insert(
                    lid,
                    Loan {
                        id: lid,
                        owner: owner_id,
                        origin_region: region,
                        mutable: spec.mutable,
                        span,
                    },
                );
                let mut loans = BTreeSet::new();
                loans.insert(lid);
                let mut owners = BTreeSet::new();
                owners.insert(owner_id);
                let mut touched = owner_arg.touched_owners.clone();
                touched.insert(owner_id);
                return EvalValue {
                    provenance: Provenance::Borrowed { loans, owners },
                    fn_targets: BTreeSet::new(),
                    origin_owner: None,
                    touched_owners: touched,
                };
            }
        }

        if self.registry.deep_materializers.contains(name) {
            let fresh = self.id_gen.next_owner();
            return EvalValue::owned(fresh);
        }

        if self.registry.drop_callees.contains(name) {
            if let Some(first_expr) = arg_exprs.first() {
                if let Some(root) = expr_root_name(first_expr) {
                    self.check_owner_mutation(state, root, span);
                }
            }
            return EvalValue::owned(self.id_gen.next_owner());
        }

        if self.registry.suspending_callees.contains(name) {
            self.check_borrow_suspension(state, name, arg_vals, span);
            return EvalValue::owned(self.id_gen.next_owner());
        }

        if let Some(info) = state.resolve(name).cloned() {
            if !info.fn_targets.is_empty() {
                return self.check_indirect_call(state, &info.fn_targets, arg_vals, span);
            }
        }

        if self.registry.sync_read_safe_callees.contains(name) {
            return EvalValue::owned(self.id_gen.next_owner());
        }

        if let Some(summary) = self.summaries.get(name) {
            if summary.may_suspend {
                self.check_borrow_suspension(state, name, arg_vals, span);
            }
            for (idx, av) in arg_vals.iter().enumerate() {
                if av.provenance.is_borrowed() {
                    if summary.escapes_params_to_return.contains(&idx)
                        || summary.escapes_params_to_global.contains(&idx)
                    {
                        self.emit(
                            LifetimeDiagCode::E4G001,
                            format!("call to `{name}` may escape borrowed argument #{idx}"),
                            span,
                        );
                    } else if summary.captures_params.contains(&idx) {
                        self.emit(
                            LifetimeDiagCode::E4G002,
                            format!("call to `{name}` may capture borrowed argument #{idx}"),
                            span,
                        );
                    } else if !summary.is_safe_sync_readonly_for_arg(idx) {
                        self.emit(
                            LifetimeDiagCode::E4G005,
                            format!("call to `{name}` lacks proven safe summary for argument #{idx}"),
                            span,
                        );
                    }
                }
            }
            return EvalValue::owned(self.id_gen.next_owner());
        }

        // Unknown / native / unaudited callee
        if arg_vals.iter().any(|a| a.provenance.is_borrowed()) {
            self.emit(
                LifetimeDiagCode::E4G005,
                format!("unknown or unaudited call `{name}` cannot accept borrowed value"),
                span,
            );
        }
        EvalValue::owned(self.id_gen.next_owner())
    }

    fn check_indirect_call(
        &mut self,
        state: &FlowState,
        targets: &BTreeSet<String>,
        arg_vals: &[EvalValue],
        span: Span,
    ) -> EvalValue {
        let any_borrowed = arg_vals.iter().any(|a| a.provenance.is_borrowed());
        if targets.is_empty() {
            if any_borrowed {
                self.emit(
                    LifetimeDiagCode::E4G005,
                    "indirect call with unknown target set cannot accept borrowed value",
                    span,
                );
            }
            return EvalValue::owned(self.id_gen.next_owner());
        }
        let all_suspending = targets
            .iter()
            .any(|t| self.registry.suspending_callees.contains(t) || self.summaries.get(t).map(|s| s.may_suspend).unwrap_or(false));
        if all_suspending && (any_borrowed || !state.loans.is_empty()) {
            self.emit(
                LifetimeDiagCode::E4G003,
                "indirect call target set includes a suspending function",
                span,
            );
            return EvalValue::owned(self.id_gen.next_owner());
        }
        if any_borrowed {
            for target in targets {
                let safe = self.registry.sync_read_safe_callees.contains(target)
                    || self
                        .summaries
                        .get(target)
                        .map(|s| s.is_safe_indirect_readonly())
                        .unwrap_or(false);
                if !safe {
                    self.emit(
                        LifetimeDiagCode::E4G005,
                        format!("indirect target `{target}` is not proven synchronous read-only non-escaping"),
                        span,
                    );
                    break;
                }
            }
        }
        EvalValue::owned(self.id_gen.next_owner())
    }
}

fn collect_free_vars(expr: &Expr, bound: &mut HashSet<String>, out: &mut BTreeSet<String>) {
    match &expr.kind {
        ExprKind::Identifier(name) => {
            if !bound.contains(name) {
                out.insert(name.clone());
            }
        }
        ExprKind::Binary { left, right, .. } | ExprKind::Range { start: left, end: right, .. } => {
            collect_free_vars(left, bound, out);
            collect_free_vars(right, bound, out);
        }
        ExprKind::Unary { operand, .. }
        | ExprKind::Paren(operand)
        | ExprKind::Cast { expr: operand, .. }
        | ExprKind::TypeCheck { expr: operand, .. }
        | ExprKind::IsPattern { expr: operand, .. }
        | ExprKind::FieldAccess { object: operand, .. } => {
            collect_free_vars(operand, bound, out);
        }
        ExprKind::Index { object, index } => {
            collect_free_vars(object, bound, out);
            collect_free_vars(index, bound, out);
        }
        ExprKind::Call { callee, args, .. } => {
            collect_free_vars(callee, bound, out);
            for a in args {
                collect_free_vars(a, bound, out);
            }
        }
        ExprKind::MethodCall { receiver, args, .. } => {
            collect_free_vars(receiver, bound, out);
            for a in args {
                collect_free_vars(a, bound, out);
            }
        }
        ExprKind::Array(items) | ExprKind::Tuple(items) | ExprKind::Return(items) => {
            for item in items {
                collect_free_vars(item, bound, out);
            }
        }
        ExprKind::Map(pairs) => {
            for (k, v) in pairs {
                collect_free_vars(k, bound, out);
                collect_free_vars(v, bound, out);
            }
        }
        ExprKind::StructLiteral { fields, .. } => {
            for f in fields {
                collect_free_vars(&f.value, bound, out);
            }
        }
        ExprKind::EnumConstructor { args, .. } => {
            for a in args {
                collect_free_vars(a, bound, out);
            }
        }
        ExprKind::Lambda { params, body, .. } => {
            let mut inner_bound = bound.clone();
            for (p, _) in params {
                inner_bound.insert(p.clone());
            }
            collect_free_vars(body, &mut inner_bound, out);
        }
        ExprKind::If {
            condition,
            then_branch,
            else_branch,
        } => {
            collect_free_vars(condition, bound, out);
            collect_free_vars(then_branch, bound, out);
            if let Some(eb) = else_branch {
                collect_free_vars(eb, bound, out);
            }
        }
        ExprKind::Block(stmts) | ExprKind::EffectBlock(stmts) => {
            let mut inner_bound = bound.clone();
            for s in stmts {
                collect_free_vars_stmt(s, &mut inner_bound, out);
            }
        }
        ExprKind::Literal(_) | ExprKind::RegexLiteral { .. } => {}
    }
}

fn collect_free_vars_stmt(stmt: &Stmt, bound: &mut HashSet<String>, out: &mut BTreeSet<String>) {
    match &stmt.kind {
        StmtKind::Local {
            bindings,
            initializer,
        } => {
            if let Some(inits) = initializer {
                for e in inits {
                    collect_free_vars(e, bound, out);
                }
            }
            for b in bindings {
                bound.insert(b.name.clone());
            }
        }
        StmtKind::Assign { targets, values } => {
            for t in targets {
                collect_free_vars(t, bound, out);
            }
            for v in values {
                collect_free_vars(v, bound, out);
            }
        }
        StmtKind::CompoundAssign { target, value, .. } => {
            collect_free_vars(target, bound, out);
            collect_free_vars(value, bound, out);
        }
        StmtKind::Expr(e) => collect_free_vars(e, bound, out),
        StmtKind::If {
            condition,
            then_block,
            elseif_branches,
            else_block,
        } => {
            collect_free_vars(condition, bound, out);
            let mut tb = bound.clone();
            for s in then_block {
                collect_free_vars_stmt(s, &mut tb, out);
            }
            for (c, b) in elseif_branches {
                collect_free_vars(c, bound, out);
                let mut eb = bound.clone();
                for s in b {
                    collect_free_vars_stmt(s, &mut eb, out);
                }
            }
            if let Some(eblock) = else_block {
                let mut eb = bound.clone();
                for s in eblock {
                    collect_free_vars_stmt(s, &mut eb, out);
                }
            }
        }
        StmtKind::While { condition, body } => {
            collect_free_vars(condition, bound, out);
            let mut bb = bound.clone();
            for s in body {
                collect_free_vars_stmt(s, &mut bb, out);
            }
        }
        StmtKind::ForNumeric {
            variable,
            start,
            end,
            step,
            body,
        } => {
            collect_free_vars(start, bound, out);
            collect_free_vars(end, bound, out);
            if let Some(st) = step {
                collect_free_vars(st, bound, out);
            }
            let mut bb = bound.clone();
            bb.insert(variable.clone());
            for s in body {
                collect_free_vars_stmt(s, &mut bb, out);
            }
        }
        StmtKind::ForIn {
            variables,
            iterator,
            body,
        } => {
            collect_free_vars(iterator, bound, out);
            let mut bb = bound.clone();
            for v in variables {
                bb.insert(v.clone());
            }
            for s in body {
                collect_free_vars_stmt(s, &mut bb, out);
            }
        }
        StmtKind::Return(vals) => {
            for v in vals {
                collect_free_vars(v, bound, out);
            }
        }
        StmtKind::Block(stmts) | StmtKind::Scoped(stmts) => {
            let mut bb = bound.clone();
            for s in stmts {
                collect_free_vars_stmt(s, &mut bb, out);
            }
        }
        StmtKind::Defer(inner) | StmtKind::ErrDefer(inner) => {
            collect_free_vars_stmt(inner, bound, out);
        }
        StmtKind::Break | StmtKind::Continue => {}
    }
}

fn collect_functions(items: &[Item], out: &mut BTreeMap<String, FunctionDef>) {
    for item in items {
        match &item.kind {
            ItemKind::Function(f) => {
                out.insert(f.name.clone(), f.clone());
            }
            ItemKind::Impl(imp) => {
                for m in &imp.methods {
                    out.insert(m.name.clone(), m.clone());
                }
            }
            ItemKind::Module { items: nested, .. } => {
                collect_functions(nested, out);
            }
            _ => {}
        }
    }
}

fn compute_fn_summaries(
    fn_defs: &BTreeMap<String, FunctionDef>,
    registry: &LifetimeRegistry,
) -> BTreeMap<String, FnSummary> {
    let mut summaries = BTreeMap::new();
    for (name, def) in fn_defs {
        let mut s = FnSummary {
            param_count: def.params.len(),
            ..FnSummary::default()
        };
        if def
            .return_type
            .as_ref()
            .and_then(type_contains_ref)
            .is_some()
        {
            s.returns_unresolved_borrow = true;
        }
        summaries.insert(name.clone(), s);
    }

    for _ in 0..8 {
        let mut changed = false;
        for (name, def) in fn_defs {
            let param_names: HashMap<String, usize> = def
                .params
                .iter()
                .enumerate()
                .map(|(i, p)| (p.name.clone(), i))
                .collect();
            let mut next = summaries.get(name).cloned().unwrap_or_default();
            let mut calls = Vec::new();
            let mut returns_param = BTreeSet::new();
            let mut mutates_param = BTreeSet::new();
            let mut captures_param = BTreeSet::new();
            for stmt in &def.body {
                scan_fn_summary_stmt(
                    stmt,
                    &param_names,
                    &mut calls,
                    &mut returns_param,
                    &mut mutates_param,
                    &mut captures_param,
                );
            }
            for idx in returns_param {
                if next.escapes_params_to_return.insert(idx) {
                    changed = true;
                }
            }
            for idx in mutates_param {
                if next.mutates_params.insert(idx) {
                    changed = true;
                }
            }
            for idx in captures_param {
                if next.captures_params.insert(idx) {
                    changed = true;
                }
            }
            for (callee, arg_param_idxs) in calls {
                if registry.suspending_callees.contains(&callee) {
                    if !next.may_suspend {
                        next.may_suspend = true;
                        changed = true;
                    }
                    for opt_pidx in arg_param_idxs {
                        if let Some(pidx) = opt_pidx {
                            if next.unproven_params.insert(pidx) {
                                changed = true;
                            }
                        }
                    }
                } else if registry.sync_read_safe_callees.contains(&callee)
                    || registry.deep_materializers.contains(&callee)
                    || registry.borrow_producers.contains_key(&callee)
                {
                    // Audited callees handled directly
                } else if registry.drop_callees.contains(&callee) {
                    for opt_pidx in arg_param_idxs {
                        if let Some(pidx) = opt_pidx {
                            if next.mutates_params.insert(pidx) {
                                changed = true;
                            }
                        }
                    }
                } else if let Some(csum) = summaries.get(&callee) {
                    let is_recursive = callee == *name || calls_reach(&fn_defs, &callee, name);
                    if csum.may_suspend && !next.may_suspend {
                        next.may_suspend = true;
                        changed = true;
                    }
                    for (pos, opt_pidx) in arg_param_idxs.into_iter().enumerate() {
                        if let Some(pidx) = opt_pidx {
                            if is_recursive && next.unproven_params.insert(pidx) {
                                changed = true;
                            }
                            if csum.escapes_params_to_global.contains(&pos)
                                && next.escapes_params_to_global.insert(pidx)
                            {
                                changed = true;
                            }
                            if csum.mutates_params.contains(&pos)
                                && next.mutates_params.insert(pidx)
                            {
                                changed = true;
                            }
                            if csum.captures_params.contains(&pos)
                                && next.captures_params.insert(pidx)
                            {
                                changed = true;
                            }
                            if csum.unproven_params.contains(&pos)
                                && next.unproven_params.insert(pidx)
                            {
                                changed = true;
                            }
                        }
                    }
                } else {
                    // Unknown / native / unaudited callee fails closed for forwarded borrows
                    for opt_pidx in arg_param_idxs {
                        if let Some(pidx) = opt_pidx {
                            if next.unproven_params.insert(pidx) {
                                changed = true;
                            }
                        }
                    }
                }
            }
            summaries.insert(name.clone(), next);
        }
        if !changed {
            break;
        }
    }
    summaries
}

fn calls_reach(
    fn_defs: &BTreeMap<String, FunctionDef>,
    start: &str,
    target: &str,
) -> bool {
    let mut visited = HashSet::new();
    let mut queue = vec![start.to_string()];
    while let Some(curr) = queue.pop() {
        if !visited.insert(curr.clone()) {
            continue;
        }
        if let Some(def) = fn_defs.get(&curr) {
            let dummy_params = HashMap::new();
            let mut calls = Vec::new();
            let mut d1 = BTreeSet::new();
            let mut d2 = BTreeSet::new();
            let mut d3 = BTreeSet::new();
            for s in &def.body {
                scan_fn_summary_stmt(s, &dummy_params, &mut calls, &mut d1, &mut d2, &mut d3);
            }
            for (next, _) in calls {
                if next == target {
                    return true;
                }
                if !visited.contains(&next) && fn_defs.contains_key(&next) {
                    queue.push(next);
                }
            }
        }
    }
    false
}

fn scan_fn_summary_stmt(
    stmt: &Stmt,
    params: &HashMap<String, usize>,
    calls: &mut Vec<(String, Vec<Option<usize>>)>,
    returns_param: &mut BTreeSet<usize>,
    mutates_param: &mut BTreeSet<usize>,
    captures_param: &mut BTreeSet<usize>,
) {
    match &stmt.kind {
        StmtKind::Local { initializer, .. } => {
            if let Some(inits) = initializer {
                for e in inits {
                    scan_fn_summary_expr(
                        e,
                        params,
                        calls,
                        returns_param,
                        mutates_param,
                        captures_param,
                    );
                }
            }
        }
        StmtKind::Assign { targets, values } => {
            for t in targets {
                if let Some(root) = expr_root_name(t) {
                    if let Some(&idx) = params.get(root) {
                        mutates_param.insert(idx);
                    }
                }
            }
            for v in values {
                scan_fn_summary_expr(v, params, calls, returns_param, mutates_param, captures_param);
            }
        }
        StmtKind::CompoundAssign { target, value, .. } => {
            if let Some(root) = expr_root_name(target) {
                if let Some(&idx) = params.get(root) {
                    mutates_param.insert(idx);
                }
            }
            scan_fn_summary_expr(
                value,
                params,
                calls,
                returns_param,
                mutates_param,
                captures_param,
            );
        }
        StmtKind::Expr(e) => scan_fn_summary_expr(
            e,
            params,
            calls,
            returns_param,
            mutates_param,
            captures_param,
        ),
        StmtKind::Return(vals) => {
            for v in vals {
                if let Some(root) = expr_root_name(v) {
                    if let Some(&idx) = params.get(root) {
                        returns_param.insert(idx);
                    }
                }
                scan_fn_summary_expr(v, params, calls, returns_param, mutates_param, captures_param);
            }
        }
        StmtKind::If {
            condition,
            then_block,
            elseif_branches,
            else_block,
        } => {
            scan_fn_summary_expr(
                condition,
                params,
                calls,
                returns_param,
                mutates_param,
                captures_param,
            );
            for s in then_block {
                scan_fn_summary_stmt(s, params, calls, returns_param, mutates_param, captures_param);
            }
            for (c, b) in elseif_branches {
                scan_fn_summary_expr(
                    c,
                    params,
                    calls,
                    returns_param,
                    mutates_param,
                    captures_param,
                );
                for s in b {
                    scan_fn_summary_stmt(
                        s,
                        params,
                        calls,
                        returns_param,
                        mutates_param,
                        captures_param,
                    );
                }
            }
            if let Some(eb) = else_block {
                for s in eb {
                    scan_fn_summary_stmt(
                        s,
                        params,
                        calls,
                        returns_param,
                        mutates_param,
                        captures_param,
                    );
                }
            }
        }
        StmtKind::While { condition, body } => {
            scan_fn_summary_expr(
                condition,
                params,
                calls,
                returns_param,
                mutates_param,
                captures_param,
            );
            for s in body {
                scan_fn_summary_stmt(s, params, calls, returns_param, mutates_param, captures_param);
            }
        }
        StmtKind::ForNumeric {
            start,
            end,
            step,
            body,
            ..
        } => {
            scan_fn_summary_expr(start, params, calls, returns_param, mutates_param, captures_param);
            scan_fn_summary_expr(end, params, calls, returns_param, mutates_param, captures_param);
            if let Some(st) = step {
                scan_fn_summary_expr(st, params, calls, returns_param, mutates_param, captures_param);
            }
            for s in body {
                scan_fn_summary_stmt(s, params, calls, returns_param, mutates_param, captures_param);
            }
        }
        StmtKind::ForIn { iterator, body, .. } => {
            scan_fn_summary_expr(
                iterator,
                params,
                calls,
                returns_param,
                mutates_param,
                captures_param,
            );
            for s in body {
                scan_fn_summary_stmt(s, params, calls, returns_param, mutates_param, captures_param);
            }
        }
        StmtKind::Block(stmts) | StmtKind::Scoped(stmts) => {
            for s in stmts {
                scan_fn_summary_stmt(s, params, calls, returns_param, mutates_param, captures_param);
            }
        }
        StmtKind::Defer(inner) | StmtKind::ErrDefer(inner) => {
            scan_fn_summary_stmt(inner, params, calls, returns_param, mutates_param, captures_param);
        }
        StmtKind::Break | StmtKind::Continue => {}
    }
}

fn scan_fn_summary_expr(
    expr: &Expr,
    params: &HashMap<String, usize>,
    calls: &mut Vec<(String, Vec<Option<usize>>)>,
    returns_param: &mut BTreeSet<usize>,
    mutates_param: &mut BTreeSet<usize>,
    captures_param: &mut BTreeSet<usize>,
) {
    match &expr.kind {
        ExprKind::Call { callee, args, .. } => {
            if let Some(name) = callee_name(callee) {
                let idxs = args
                    .iter()
                    .map(|a| expr_root_name(a).and_then(|n| params.get(n).copied()))
                    .collect();
                calls.push((name, idxs));
            }
            for a in args {
                scan_fn_summary_expr(a, params, calls, returns_param, mutates_param, captures_param);
            }
        }
        ExprKind::Lambda {
            params: lparams,
            body,
            ..
        } => {
            let mut bound: HashSet<String> = lparams.iter().map(|(n, _)| n.clone()).collect();
            let mut captured = BTreeSet::new();
            collect_free_vars(body, &mut bound, &mut captured);
            for c in captured {
                if let Some(&idx) = params.get(&c) {
                    captures_param.insert(idx);
                }
            }
        }
        ExprKind::Binary { left, right, .. } | ExprKind::Range { start: left, end: right, .. } => {
            scan_fn_summary_expr(left, params, calls, returns_param, mutates_param, captures_param);
            scan_fn_summary_expr(right, params, calls, returns_param, mutates_param, captures_param);
        }
        ExprKind::Unary { operand, .. }
        | ExprKind::Paren(operand)
        | ExprKind::Cast { expr: operand, .. }
        | ExprKind::TypeCheck { expr: operand, .. }
        | ExprKind::IsPattern { expr: operand, .. }
        | ExprKind::FieldAccess { object: operand, .. } => {
            scan_fn_summary_expr(
                operand,
                params,
                calls,
                returns_param,
                mutates_param,
                captures_param,
            );
        }
        ExprKind::Index { object, index } => {
            scan_fn_summary_expr(object, params, calls, returns_param, mutates_param, captures_param);
            scan_fn_summary_expr(index, params, calls, returns_param, mutates_param, captures_param);
        }
        ExprKind::MethodCall { receiver, args, .. } => {
            scan_fn_summary_expr(
                receiver,
                params,
                calls,
                returns_param,
                mutates_param,
                captures_param,
            );
            for a in args {
                scan_fn_summary_expr(a, params, calls, returns_param, mutates_param, captures_param);
            }
        }
        ExprKind::Array(elems) | ExprKind::Tuple(elems) | ExprKind::Return(elems) => {
            for e in elems {
                scan_fn_summary_expr(e, params, calls, returns_param, mutates_param, captures_param);
            }
        }
        ExprKind::Map(pairs) => {
            for (k, v) in pairs {
                scan_fn_summary_expr(k, params, calls, returns_param, mutates_param, captures_param);
                scan_fn_summary_expr(v, params, calls, returns_param, mutates_param, captures_param);
            }
        }
        ExprKind::StructLiteral { fields, .. } => {
            for f in fields {
                scan_fn_summary_expr(
                    &f.value,
                    params,
                    calls,
                    returns_param,
                    mutates_param,
                    captures_param,
                );
            }
        }
        ExprKind::EnumConstructor { args, .. } => {
            for a in args {
                scan_fn_summary_expr(a, params, calls, returns_param, mutates_param, captures_param);
            }
        }
        ExprKind::If {
            condition,
            then_branch,
            else_branch,
        } => {
            scan_fn_summary_expr(
                condition,
                params,
                calls,
                returns_param,
                mutates_param,
                captures_param,
            );
            scan_fn_summary_expr(
                then_branch,
                params,
                calls,
                returns_param,
                mutates_param,
                captures_param,
            );
            if let Some(eb) = else_branch {
                scan_fn_summary_expr(
                    eb,
                    params,
                    calls,
                    returns_param,
                    mutates_param,
                    captures_param,
                );
            }
        }
        ExprKind::Block(stmts) | ExprKind::EffectBlock(stmts) => {
            for s in stmts {
                scan_fn_summary_stmt(s, params, calls, returns_param, mutates_param, captures_param);
            }
        }
        ExprKind::Literal(_) | ExprKind::Identifier(_) | ExprKind::RegexLiteral { .. } => {}
    }
}

pub fn check_module_lifetimes_with_registry(
    items: &[Item],
    module_name: &Option<String>,
    registry: &LifetimeRegistry,
) -> Result<()> {
    let mut fn_defs = BTreeMap::new();
    collect_functions(items, &mut fn_defs);
    let summaries = compute_fn_summaries(&fn_defs, registry);

    let mut analyzer = Analyzer::new(registry, &fn_defs, &summaries, module_name.clone());
    let mut state = FlowState {
        bindings: BTreeMap::new(),
        loans: BTreeMap::new(),
        regions: vec![RegionFrame {
            id: RegionId(0),
            bindings: BTreeMap::new(),
            owned_bindings: Vec::new(),
            defers: Vec::new(),
        }],
    };
    analyzer.check_items(&mut state, items);
    analyzer.record_highest_diagnostic()
}

pub fn check_module_lifetimes(items: &[Item], module_name: &Option<String>) -> Result<()> {
    check_module_lifetimes_with_registry(items, module_name, &LifetimeRegistry::default())
}

pub fn check_program_lifetimes(modules: &[LoadedModule]) -> Result<()> {
    let registry = LifetimeRegistry::default();
    let mut fn_defs = BTreeMap::new();
    for m in modules {
        collect_functions(&m.items, &mut fn_defs);
    }
    let summaries = compute_fn_summaries(&fn_defs, &registry);

    for m in modules {
        let mut analyzer = Analyzer::new(&registry, &fn_defs, &summaries, Some(m.path.clone()));
        let mut state = FlowState {
            bindings: BTreeMap::new(),
            loans: BTreeMap::new(),
            regions: vec![RegionFrame {
                id: RegionId(0),
                bindings: BTreeMap::new(),
                owned_bindings: Vec::new(),
                defers: Vec::new(),
            }],
        };
        analyzer.check_items(&mut state, &m.items);
        analyzer.record_highest_diagnostic()?;
    }
    Ok(())
}

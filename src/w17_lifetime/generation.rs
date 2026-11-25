//! Weak-VM and monotonic generation-tagged runtime handle invalidation for `D-LIFE`.
//!
//! Enforces:
//! - Monotonic non-wrapping `u64` VM generation counters (no wraparound-based revival).
//! - Weak VM liveness references (`Weak<VmLiveness>`) so retained handles never keep a
//!   dropped VM alive and deterministically return `Err(LifecycleError::VmDropped)` after `drop(vm)`.
//! - Recreating a VM allocates a strictly higher generation; old handles remain permanently invalid
//!   and cross-VM validation returns `Err(LifecycleError::GenerationMismatch)`.
//! - Backing owner slot tracking via `OwnerRegistry` (`cancel -> acknowledge_join -> release`),
//!   invalidating views on owner destruction and bumping slot generations on slot reuse.
//! - Zero-allocation fast path (`PreparedScalarHandle::read_i64` and `GenStructHandle::read_view`)
//!   that verifies weak-VM liveness, VM generation, and owner generation before dereference.

use std::cell::{Cell, RefCell};
use std::fmt;
use std::marker::PhantomData;
use std::rc::{Rc, Weak};
use std::sync::atomic::{AtomicU32, AtomicU64, Ordering};

use crate::bytecode::{Value, ValueKey};
use crate::data::owner::{LeaseError, LeaseLifecycleState, OwnerId, OwnerRegistry};
use crate::embed::{
    ArrayHandle, CompiledBytecode, EmbeddedProgram, FromLustValue, FunctionArgs, FunctionHandle,
    IntoTypedValue, LustStructView, MapHandle, StructField, StructHandle, ValueRef,
};
use crate::error::LustError;

use super::script_bridge::{declare_view_externs, install_view_runtime};

static NEXT_VM_GENERATION: AtomicU64 = AtomicU64::new(1);
static NEXT_RUNTIME_ID: AtomicU32 = AtomicU32::new(1);

fn allocate_vm_generation() -> u64 {
    let generation = NEXT_VM_GENERATION.fetch_add(1, Ordering::Relaxed);
    assert!(
        generation != 0 && generation != u64::MAX,
        "Fatal: LifetimeVm monotonic generation counter exhausted (wraparound revival forbidden)"
    );
    generation
}

fn allocate_runtime_id() -> u32 {
    let rt = NEXT_RUNTIME_ID.fetch_add(1, Ordering::Relaxed);
    assert!(
        rt != 0 && rt != u32::MAX,
        "Fatal: LifetimeVm runtime_id counter exhausted (wraparound revival forbidden)"
    );
    rt
}

/// Deterministic lifecycle and generation validation error.
#[derive(Debug)]
pub enum LifecycleError {
    VmDropped {
        handle_generation: u64,
    },
    GenerationMismatch {
        expected_vm_generation: u64,
        handle_generation: u64,
    },
    OwnerInvalidated(LeaseError),
    BorrowConflict {
        target: &'static str,
    },
    FieldNotFound {
        type_name: String,
        field: String,
    },
    TypeMismatch {
        expected: &'static str,
        actual: String,
    },
    UnsupportedTransfer(&'static str),
    Lust(LustError),
}

impl fmt::Display for LifecycleError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::VmDropped { handle_generation } => write!(
                f,
                "stale handle: owning VM (generation {}) has been dropped",
                handle_generation
            ),
            Self::GenerationMismatch {
                expected_vm_generation,
                handle_generation,
            } => write!(
                f,
                "stale handle generation mismatch: active VM generation is {}, handle generation is {}",
                expected_vm_generation, handle_generation
            ),
            Self::OwnerInvalidated(err) => write!(f, "owner invalidated: {}", err),
            Self::BorrowConflict { target } => {
                write!(f, "conflicting mutation while '{}' is actively borrowed", target)
            }
            Self::FieldNotFound { type_name, field } => {
                write!(f, "struct '{}' has no field '{}'", type_name, field)
            }
            Self::TypeMismatch { expected, actual } => {
                write!(f, "type mismatch: expected {}, got {}", expected, actual)
            }
            Self::UnsupportedTransfer(reason) => {
                write!(f, "illegal transfer: {}", reason)
            }
            Self::Lust(err) => write!(f, "{}", err),
        }
    }
}

impl std::error::Error for LifecycleError {}

impl From<LustError> for LifecycleError {
    fn from(err: LustError) -> Self {
        Self::Lust(err)
    }
}

/// Shared single-threaded liveness and owner-generation state owned exclusively by `LifetimeVm`.
///
/// Handles retain only a `Weak<VmLiveness>`, so dropping `LifetimeVm` immediately flips `alive`
/// to `false`, revokes all active owners, and drops the sole strong `Rc<VmLiveness>`.
pub(crate) struct VmLiveness {
    pub(crate) generation: u64,
    pub(crate) runtime_id: u32,
    pub(crate) alive: Cell<bool>,
    pub(crate) active_owners: RefCell<Vec<OwnerId>>,
    pub(crate) owner_registry: RefCell<OwnerRegistry>,
}

impl VmLiveness {
    fn new(generation: u64, runtime_id: u32) -> Rc<Self> {
        Rc::new(Self {
            generation,
            runtime_id,
            alive: Cell::new(true),
            active_owners: RefCell::new(Vec::new()),
            owner_registry: RefCell::new(OwnerRegistry::new(runtime_id)),
        })
    }

    fn allocate_owner(&self, tenant_id: u32) -> OwnerId {
        let owner = self.owner_registry.borrow_mut().allocate(tenant_id);
        self.active_owners.borrow_mut().push(owner);
        owner
    }

    #[inline]
    fn validate_handle(
        weak: &Weak<Self>,
        handle_generation: u64,
        owner_id: OwnerId,
        tenant_id: u32,
    ) -> Result<(), LifecycleError> {
        let state = weak.upgrade().ok_or(LifecycleError::VmDropped {
            handle_generation,
        })?;
        if !state.alive.get() {
            return Err(LifecycleError::VmDropped { handle_generation });
        }
        if state.generation != handle_generation {
            return Err(LifecycleError::GenerationMismatch {
                expected_vm_generation: state.generation,
                handle_generation,
            });
        }
        state
            .owner_registry
            .borrow()
            .validate_access(owner_id, tenant_id)
            .map_err(LifecycleError::OwnerInvalidated)?;
        Ok(())
    }

    #[inline]
    fn validate_vm_only(weak: &Weak<Self>, handle_generation: u64) -> Result<(), LifecycleError> {
        let state = weak.upgrade().ok_or(LifecycleError::VmDropped {
            handle_generation,
        })?;
        if !state.alive.get() {
            return Err(LifecycleError::VmDropped { handle_generation });
        }
        if state.generation != handle_generation {
            return Err(LifecycleError::GenerationMismatch {
                expected_vm_generation: state.generation,
                handle_generation,
            });
        }
        Ok(())
    }

    fn invalidate_all_on_drop(&self) {
        self.alive.set(false);
        let owners: Vec<OwnerId> = self.active_owners.borrow_mut().drain(..).collect();
        let mut reg = self.owner_registry.borrow_mut();
        for owner in owners {
            match reg.slot_state(owner.slot_id) {
                Some(LeaseLifecycleState::Active) => {
                    let _ = reg.cancel(owner);
                    let _ = reg.acknowledge_join(owner);
                    let _ = reg.release(owner);
                }
                Some(LeaseLifecycleState::Cancelled) => {
                    let _ = reg.acknowledge_join(owner);
                    let _ = reg.release(owner);
                }
                Some(LeaseLifecycleState::Joined) => {
                    let _ = reg.release(owner);
                }
                _ => {}
            }
        }
    }
}

/// Owning VM wrapper enforcing weak-VM generation invalidation and owner lease lifecycle.
///
/// Explicitly `!Send + !Sync` (contains `EmbeddedProgram` and `Rc<VmLiveness>`).
pub struct LifetimeVm {
    program: EmbeddedProgram,
    liveness: Rc<VmLiveness>,
    default_tenant: u32,
    _not_send_sync: PhantomData<*const ()>,
}

impl Drop for LifetimeVm {
    fn drop(&mut self) {
        self.liveness.invalidate_all_on_drop();
    }
}

impl LifetimeVm {
    /// Compiles a Lust module through the real lexer, parser, typechecker, and static
    /// lifetime checker, installs the script view runtime bridge, and stamps a fresh
    /// monotonic VM generation.
    pub fn compile_script(source: &str) -> Result<Self, LifecycleError> {
        let builder = declare_view_externs(
            EmbeddedProgram::builder()
                .module("main", source)
                .entry_module("main"),
        );
        let mut program = builder.compile()?;
        install_view_runtime(&mut program);
        Ok(Self::from_program(program, 1))
    }

    /// Instantiates a fresh `LifetimeVm` from precompiled bytecode with a strictly new
    /// monotonic generation counter.
    pub fn from_compiled(bytecode: &CompiledBytecode) -> Result<Self, LifecycleError> {
        let mut program = EmbeddedProgram::from_compiled(bytecode)?;
        install_view_runtime(&mut program);
        Ok(Self::from_program(program, 1))
    }

    /// Wraps an already-compiled `EmbeddedProgram` with a fresh monotonic generation and
    /// owner registry.
    pub fn from_program(program: EmbeddedProgram, default_tenant: u32) -> Self {
        let generation = allocate_vm_generation();
        let runtime_id = allocate_runtime_id();
        Self {
            program,
            liveness: VmLiveness::new(generation, runtime_id),
            default_tenant,
            _not_send_sync: PhantomData,
        }
    }

    #[inline]
    pub fn generation(&self) -> u64 {
        self.liveness.generation
    }

    #[inline]
    pub fn runtime_id(&self) -> u32 {
        self.liveness.runtime_id
    }

    #[inline]
    pub fn default_tenant(&self) -> u32 {
        self.default_tenant
    }

    pub fn program(&self) -> &EmbeddedProgram {
        &self.program
    }

    pub fn program_mut(&mut self) -> &mut EmbeddedProgram {
        &mut self.program
    }

    /// Calls a compiled Lust function on this live VM.
    pub fn call(&mut self, func_name: &str, args: Vec<Value>) -> Result<Value, LifecycleError> {
        Ok(self.program.vm_mut().call(func_name, args)?)
    }

    /// Creates a real `StructInstance` inside the VM and returns a weak-VM,
    /// generation-tagged handle bound to a newly allocated `OwnerId`.
    pub fn create_struct<I>(
        &self,
        type_name: impl Into<String>,
        fields: I,
    ) -> Result<GenStructHandle, LifecycleError>
    where
        I: IntoIterator,
        I::Item: Into<StructField>,
    {
        self.create_struct_for_tenant(type_name, fields, self.default_tenant)
    }

    pub fn create_struct_for_tenant<I>(
        &self,
        type_name: impl Into<String>,
        fields: I,
        tenant_id: u32,
    ) -> Result<GenStructHandle, LifecycleError>
    where
        I: IntoIterator,
        I::Item: Into<StructField>,
    {
        let inst = self.program.struct_instance(type_name, fields)?;
        let owner_id = self.liveness.allocate_owner(tenant_id);
        Ok(GenStructHandle {
            handle: inst.into_handle(),
            vm_liveness: Rc::downgrade(&self.liveness),
            vm_generation: self.liveness.generation,
            owner_id,
            tenant_id,
            _not_send_sync: PhantomData,
        })
    }

    /// Wraps a real VM `ArrayHandle` in a weak-VM, generation-tagged handle.
    pub fn wrap_array(&self, handle: ArrayHandle) -> GenArrayHandle {
        self.wrap_array_for_tenant(handle, self.default_tenant)
    }

    pub fn wrap_array_for_tenant(&self, handle: ArrayHandle, tenant_id: u32) -> GenArrayHandle {
        let owner_id = self.liveness.allocate_owner(tenant_id);
        GenArrayHandle {
            handle,
            vm_liveness: Rc::downgrade(&self.liveness),
            vm_generation: self.liveness.generation,
            owner_id,
            tenant_id,
            _not_send_sync: PhantomData,
        }
    }

    /// Wraps a real VM `MapHandle` in a weak-VM, generation-tagged handle.
    pub fn wrap_map(&self, handle: MapHandle) -> GenMapHandle {
        self.wrap_map_for_tenant(handle, self.default_tenant)
    }

    pub fn wrap_map_for_tenant(&self, handle: MapHandle, tenant_id: u32) -> GenMapHandle {
        let owner_id = self.liveness.allocate_owner(tenant_id);
        GenMapHandle {
            handle,
            vm_liveness: Rc::downgrade(&self.liveness),
            vm_generation: self.liveness.generation,
            owner_id,
            tenant_id,
            _not_send_sync: PhantomData,
        }
    }

    /// Obtains a generation-tagged handle to a compiled Lust function on this VM.
    pub fn function_handle(&self, name: &str) -> Result<GenFunctionHandle, LifecycleError> {
        let handle = self.program.function_handle(name)?;
        Ok(GenFunctionHandle {
            handle,
            vm_liveness: Rc::downgrade(&self.liveness),
            vm_generation: self.liveness.generation,
            _not_send_sync: PhantomData,
        })
    }

    /// Validates that `handle` belongs to THIS exact live VM generation and active owner slot.
    pub fn validate_struct_handle(&self, handle: &GenStructHandle) -> Result<(), LifecycleError> {
        if handle.vm_generation != self.liveness.generation {
            return Err(LifecycleError::GenerationMismatch {
                expected_vm_generation: self.liveness.generation,
                handle_generation: handle.vm_generation,
            });
        }
        handle.validate_live()?;
        self.liveness
            .owner_registry
            .borrow()
            .validate_access(handle.owner_id, handle.tenant_id)
            .map_err(LifecycleError::OwnerInvalidated)?;
        Ok(())
    }

    /// Step 1 of explicit owner destruction: signal cancellation.
    pub fn cancel_owner(&self, owner: OwnerId) -> Result<(), LifecycleError> {
        self.liveness
            .owner_registry
            .borrow_mut()
            .cancel(owner)
            .map_err(LifecycleError::OwnerInvalidated)
    }

    /// Step 2 of explicit owner destruction: acknowledge child/worker join.
    pub fn acknowledge_owner_join(&self, owner: OwnerId) -> Result<(), LifecycleError> {
        self.liveness
            .owner_registry
            .borrow_mut()
            .acknowledge_join(owner)
            .map_err(LifecycleError::OwnerInvalidated)
    }

    /// Step 3 of explicit owner destruction: release backing slot (requires prior join).
    pub fn release_owner(&self, owner: OwnerId) -> Result<(), LifecycleError> {
        self.liveness
            .owner_registry
            .borrow_mut()
            .release(owner)
            .map_err(LifecycleError::OwnerInvalidated)
    }

    /// Full owner destruction (`cancel -> acknowledge_join -> release`), immediately
    /// invalidating every generation-tagged handle targeting `owner`.
    pub fn destroy_owner(&self, owner: OwnerId) -> Result<(), LifecycleError> {
        self.cancel_owner(owner)?;
        self.acknowledge_owner_join(owner)?;
        self.release_owner(owner)?;
        Ok(())
    }

    pub fn owner_slot_state(&self, owner: OwnerId) -> Option<LeaseLifecycleState> {
        self.liveness.owner_registry.borrow().slot_state(owner.slot_id)
    }
}

/// Scoped, non-escaping, `!Send + !Sync` borrowed view guard tied to a live handle's borrow region.
pub struct ScopedViewGuard<'a> {
    value_ref: ValueRef<'a>,
    vm_generation: u64,
    owner_id: OwnerId,
    _not_send_sync: PhantomData<*const ()>,
}

impl<'a> ScopedViewGuard<'a> {
    #[inline]
    pub fn vm_generation(&self) -> u64 {
        self.vm_generation
    }

    #[inline]
    pub fn owner_id(&self) -> OwnerId {
        self.owner_id
    }

    #[inline]
    pub fn as_value(&self) -> &Value {
        self.value_ref.as_value()
    }

    #[inline]
    pub fn as_int(&self) -> Option<i64> {
        self.value_ref.as_int()
    }

    #[inline]
    pub fn as_float(&self) -> Option<f64> {
        self.value_ref.as_float()
    }

    #[inline]
    pub fn as_bool(&self) -> Option<bool> {
        self.value_ref.as_bool()
    }

    #[inline]
    pub fn as_string(&self) -> Option<&str> {
        self.value_ref.as_string()
    }
}

/// Weak-VM, generation-tagged handle to a real Lust struct instance.
///
/// Explicitly `!Send + !Sync`.
#[derive(Clone)]
pub struct GenStructHandle {
    handle: StructHandle,
    vm_liveness: Weak<VmLiveness>,
    vm_generation: u64,
    owner_id: OwnerId,
    tenant_id: u32,
    _not_send_sync: PhantomData<*const ()>,
}

impl GenStructHandle {
    #[inline]
    pub fn vm_generation(&self) -> u64 {
        self.vm_generation
    }

    #[inline]
    pub fn owner_id(&self) -> OwnerId {
        self.owner_id
    }

    #[inline]
    pub fn tenant_id(&self) -> u32 {
        self.tenant_id
    }

    #[inline]
    pub fn validate_live(&self) -> Result<(), LifecycleError> {
        VmLiveness::validate_handle(
            &self.vm_liveness,
            self.vm_generation,
            self.owner_id,
            self.tenant_id,
        )
    }

    #[inline]
    pub fn validate_for_tenant(&self, current_tenant: u32) -> Result<(), LifecycleError> {
        VmLiveness::validate_handle(
            &self.vm_liveness,
            self.vm_generation,
            self.owner_id,
            current_tenant,
        )
    }

    #[inline]
    pub fn checked_raw_handle(&self) -> Result<&StructHandle, LifecycleError> {
        self.validate_live()?;
        Ok(&self.handle)
    }

    /// Zero-allocation checked construction of a derived `LustStructView<'a>`.
    #[inline]
    pub fn read_view<'a, V: LustStructView<'a>>(&'a self) -> Result<V, LifecycleError> {
        self.validate_live()?;
        Ok(V::from_handle(&self.handle)?)
    }

    /// Borrows a field as a scoped `!Send + !Sync` view guard after validating VM & owner liveness.
    pub fn borrow_field(&self, field: &str) -> Result<ScopedViewGuard<'_>, LifecycleError> {
        self.validate_live()?;
        let value_ref = self.handle.borrow_field(field)?;
        Ok(ScopedViewGuard {
            value_ref,
            vm_generation: self.vm_generation,
            owner_id: self.owner_id,
            _not_send_sync: PhantomData,
        })
    }

    /// Zero-allocation checked read of a scalar `int` field.
    #[inline]
    pub fn read_int_field(&self, field: &str) -> Result<i64, LifecycleError> {
        self.validate_live()?;
        let guard = self.handle.borrow_field(field)?;
        guard.as_int().ok_or_else(|| LifecycleError::TypeMismatch {
            expected: "int",
            actual: format!("{:?}", guard.as_value()),
        })
    }

    /// Attempts to mutate a field; fails deterministically with `Err(LifecycleError::BorrowConflict)`
    /// if any `ScopedViewGuard` / `ValueRef` borrow is active, or `Err(VmDropped/OwnerInvalidated)`
    /// if the VM or owner has been destroyed.
    pub fn try_set_field<V: IntoTypedValue>(
        &self,
        field: &str,
        value: V,
    ) -> Result<(), LifecycleError> {
        self.validate_live()?;
        if let Value::Struct(obj) = self.handle.as_value() {
            if obj.fields.try_borrow_mut().is_err() {
                return Err(LifecycleError::BorrowConflict {
                    target: "GenStructHandle",
                });
            }
        }
        self.handle.set_field(field, value)?;
        Ok(())
    }

    /// Prepares a zero-allocation fixed-scalar reader bound to a static field name.
    pub fn prepare_scalar_field(self, field: &'static str) -> Result<PreparedScalarHandle, LifecycleError> {
        // Validate field exists and is an int during setup (outside measured interval).
        let _ = self.read_int_field(field)?;
        Ok(PreparedScalarHandle {
            handle: self,
            field,
        })
    }
}

/// Prepared fixed-scalar reader over a weak-VM, generation-tagged struct handle.
///
/// Setup validates existence and type; each `read_i64` performs weak-VM upgrade,
/// generation check, owner-registry generation check, and zero-copy `RefCell` scalar read
/// with zero heap allocations.
pub struct PreparedScalarHandle {
    handle: GenStructHandle,
    field: &'static str,
}

impl PreparedScalarHandle {
    #[inline]
    pub fn handle(&self) -> &GenStructHandle {
        &self.handle
    }

    #[inline]
    pub fn read_i64(&self) -> Result<i64, LifecycleError> {
        self.handle.read_int_field(self.field)
    }

    #[inline]
    pub fn read_via_view<'a, V: LustStructView<'a>>(&'a self) -> Result<V, LifecycleError> {
        self.handle.read_view::<V>()
    }
}

/// Weak-VM, generation-tagged handle to a real Lust array.
///
/// Explicitly `!Send + !Sync`.
#[derive(Clone)]
pub struct GenArrayHandle {
    handle: ArrayHandle,
    vm_liveness: Weak<VmLiveness>,
    vm_generation: u64,
    owner_id: OwnerId,
    tenant_id: u32,
    _not_send_sync: PhantomData<*const ()>,
}

impl GenArrayHandle {
    #[inline]
    pub fn vm_generation(&self) -> u64 {
        self.vm_generation
    }

    #[inline]
    pub fn owner_id(&self) -> OwnerId {
        self.owner_id
    }

    #[inline]
    pub fn validate_live(&self) -> Result<(), LifecycleError> {
        VmLiveness::validate_handle(
            &self.vm_liveness,
            self.vm_generation,
            self.owner_id,
            self.tenant_id,
        )
    }

    pub fn checked_raw_handle(&self) -> Result<&ArrayHandle, LifecycleError> {
        self.validate_live()?;
        Ok(&self.handle)
    }

    pub fn len(&self) -> Result<usize, LifecycleError> {
        self.validate_live()?;
        Ok(self.handle.len())
    }

    pub fn is_empty(&self) -> Result<bool, LifecycleError> {
        Ok(self.len()? == 0)
    }

    pub fn borrow_get(&self, index: usize) -> Result<Option<ScopedViewGuard<'_>>, LifecycleError> {
        self.validate_live()?;
        Ok(self.handle.get(index).map(|value_ref| ScopedViewGuard {
            value_ref,
            vm_generation: self.vm_generation,
            owner_id: self.owner_id,
            _not_send_sync: PhantomData,
        }))
    }

    pub fn try_push(&self, value: Value) -> Result<(), LifecycleError> {
        self.validate_live()?;
        let can_mut = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            let _ = self.handle.borrow_mut();
        }))
        .is_ok();
        if !can_mut {
            return Err(LifecycleError::BorrowConflict {
                target: "GenArrayHandle",
            });
        }
        self.handle.push(value);
        Ok(())
    }
}

/// Weak-VM, generation-tagged handle to a real Lust map.
///
/// Explicitly `!Send + !Sync`.
#[derive(Clone)]
pub struct GenMapHandle {
    handle: MapHandle,
    vm_liveness: Weak<VmLiveness>,
    vm_generation: u64,
    owner_id: OwnerId,
    tenant_id: u32,
    _not_send_sync: PhantomData<*const ()>,
}

impl GenMapHandle {
    #[inline]
    pub fn vm_generation(&self) -> u64 {
        self.vm_generation
    }

    #[inline]
    pub fn owner_id(&self) -> OwnerId {
        self.owner_id
    }

    #[inline]
    pub fn validate_live(&self) -> Result<(), LifecycleError> {
        VmLiveness::validate_handle(
            &self.vm_liveness,
            self.vm_generation,
            self.owner_id,
            self.tenant_id,
        )
    }

    pub fn checked_raw_handle(&self) -> Result<&MapHandle, LifecycleError> {
        self.validate_live()?;
        Ok(&self.handle)
    }

    pub fn len(&self) -> Result<usize, LifecycleError> {
        self.validate_live()?;
        Ok(self.handle.len())
    }

    pub fn borrow_get<K: Into<ValueKey>>(
        &self,
        key: K,
    ) -> Result<Option<ScopedViewGuard<'_>>, LifecycleError> {
        self.validate_live()?;
        Ok(self.handle.get(key).map(|value_ref| ScopedViewGuard {
            value_ref,
            vm_generation: self.vm_generation,
            owner_id: self.owner_id,
            _not_send_sync: PhantomData,
        }))
    }

    pub fn try_insert<K: Into<ValueKey>>(
        &self,
        key: K,
        value: Value,
    ) -> Result<Option<Value>, LifecycleError> {
        self.validate_live()?;
        let can_mut = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            let _ = self.handle.borrow_mut();
        }))
        .is_ok();
        if !can_mut {
            return Err(LifecycleError::BorrowConflict {
                target: "GenMapHandle",
            });
        }
        Ok(self.handle.insert(key, value))
    }
}

/// Weak-VM, generation-tagged handle to a compiled Lust function.
///
/// Explicitly `!Send + !Sync`.
#[derive(Clone)]
pub struct GenFunctionHandle {
    handle: FunctionHandle,
    vm_liveness: Weak<VmLiveness>,
    vm_generation: u64,
    _not_send_sync: PhantomData<*const ()>,
}

impl GenFunctionHandle {
    #[inline]
    pub fn vm_generation(&self) -> u64 {
        self.vm_generation
    }

    #[inline]
    pub fn validate_live(&self) -> Result<(), LifecycleError> {
        VmLiveness::validate_vm_only(&self.vm_liveness, self.vm_generation)
    }

    pub fn call_typed_on<Args, R>(
        &self,
        vm: &mut LifetimeVm,
        args: Args,
    ) -> Result<R, LifecycleError>
    where
        Args: FunctionArgs,
        R: FromLustValue,
    {
        if vm.generation() != self.vm_generation {
            return Err(LifecycleError::GenerationMismatch {
                expected_vm_generation: vm.generation(),
                handle_generation: self.vm_generation,
            });
        }
        self.validate_live()?;
        Ok(self.handle.call_typed(vm.program_mut(), args)?)
    }
}

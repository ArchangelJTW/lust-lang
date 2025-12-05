use crate::bytecode::value::IteratorState;
use crate::bytecode::{LustMap, Value};
use crate::vm::task::TaskInstance;
use crate::vm::CallFrame;
use crate::vm::TaskSignal;
use crate::vm::VM;
use alloc::rc::{Rc, Weak};
use alloc::vec::Vec;
use core::cell::RefCell;
use hashbrown::{hash_map::Entry, HashMap, HashSet};
const COLLECT_INTERVAL: usize = 512;
const REGISTRATION_THRESHOLD: usize = 256;
#[derive(Default)]
pub struct CycleCollector {
    containers: HashMap<usize, ContainerEntry>,
    steps_since_collect: usize,
    pending_registrations: usize,
}

struct ContainerEntry {
    kind: ContainerKind,
    marked: bool,
}

enum ContainerKind {
    Array(Weak<RefCell<Vec<Value>>>),
    Map(Weak<RefCell<LustMap>>),
    Struct(Weak<RefCell<Vec<Value>>>),
    Iterator(Weak<RefCell<IteratorState>>),
}

enum ClearResult {
    Removed,
    Retain,
}

impl CycleCollector {
    pub fn new() -> Self {
        Self {
            containers: HashMap::new(),
            steps_since_collect: 0,
            pending_registrations: 0,
        }
    }

    pub fn register_value(&mut self, value: &Value) {
        match value {
            Value::Array(rc) => {
                if self.register_array(rc) {
                    self.pending_registrations += 1;
                }
            }

            Value::Map(rc) => {
                if self.register_map(rc) {
                    self.pending_registrations += 1;
                }
            }

            Value::Struct { fields, .. } => {
                if self.register_struct(fields) {
                    self.pending_registrations += 1;
                }
            }

            Value::Iterator(rc) => {
                if self.register_iterator(rc) {
                    self.pending_registrations += 1;
                }
            }

            Value::WeakStruct(weak) => {
                if let Some(strong) = weak.upgrade() {
                    self.register_value(&strong);
                }
            }

            _ => {}
        }
    }

    pub fn maybe_collect(&mut self, vm: &VM) {
        self.steps_since_collect = self.steps_since_collect.saturating_add(1);
        if self.containers.is_empty() {
            self.steps_since_collect = 0;
            self.pending_registrations = 0;
            return;
        }

        if self.steps_since_collect >= COLLECT_INTERVAL
            || self.pending_registrations >= REGISTRATION_THRESHOLD
        {
            self.collect(vm);
            self.steps_since_collect = 0;
            self.pending_registrations = 0;
        }
    }

    pub fn collect(&mut self, vm: &VM) {
        if self.containers.is_empty() {
            return;
        }

        let mut visited: HashSet<usize> = HashSet::new();
        self.mark_roots(vm, &mut visited);
        self.sweep();
    }

    fn mark_roots(&mut self, vm: &VM, visited: &mut HashSet<usize>) {
        for value in vm.globals.values() {
            self.mark_value(value, visited);
        }

        for frame in &vm.call_stack {
            self.mark_frame(frame, visited);
        }

        if let Some(value) = &vm.pending_return_value {
            self.mark_value(value, visited);
        }

        if let Some(signal) = &vm.pending_task_signal {
            self.mark_task_signal(signal, visited);
        }

        if let Some(signal) = &vm.last_task_signal {
            self.mark_task_signal(signal, visited);
        }

        for task in vm.task_manager.iter() {
            self.mark_task(task, visited);
        }
    }

    fn mark_task_signal(&mut self, signal: &TaskSignal, visited: &mut HashSet<usize>) {
        match signal {
            TaskSignal::Yield { value, .. } => self.mark_value(value, visited),
            TaskSignal::Stop { value } => self.mark_value(value, visited),
        }
    }

    fn mark_frame(&mut self, frame: &CallFrame, visited: &mut HashSet<usize>) {
        for value in frame.registers.iter() {
            self.mark_value(value, visited);
        }

        for value in frame.upvalues.iter() {
            self.mark_value(value, visited);
        }
    }

    fn mark_task(&mut self, task: &TaskInstance, visited: &mut HashSet<usize>) {
        for frame in &task.call_stack {
            self.mark_frame(frame, visited);
        }

        if let Some(frame) = task.initial_frame() {
            self.mark_frame(frame, visited);
        }
        if let Some(value) = &task.pending_return_value {
            self.mark_value(value, visited);
        }

        if let Some(value) = &task.last_yield {
            self.mark_value(value, visited);
        }

        if let Some(value) = &task.last_result {
            self.mark_value(value, visited);
        }
    }

    fn mark_value(&mut self, value: &Value, visited: &mut HashSet<usize>) {
        self.register_value(value);
        match value {
            Value::Array(rc) => self.mark_array(rc, visited),
            Value::Map(rc) => self.mark_map(rc, visited),
            Value::Struct { fields, .. } => self.mark_struct(fields, visited),
            Value::Enum {
                values: Some(rc), ..
            } => {
                for item in rc.iter() {
                    self.mark_value(item, visited);
                }
            }

            Value::Tuple(values) => {
                for item in values.iter() {
                    self.mark_value(item, visited);
                }
            }

            Value::Iterator(rc) => self.mark_iterator(rc, visited),
            Value::Closure { upvalues, .. } => {
                for up in upvalues.iter() {
                    let captured = up.get();
                    self.mark_value(&captured, visited);
                }
            }

            Value::WeakStruct(weak) => {
                if let Some(strong) = weak.upgrade() {
                    self.mark_value(&strong, visited);
                }
            }

            _ => {}
        }
    }

    fn mark_array(&mut self, rc: &Rc<RefCell<Vec<Value>>>, visited: &mut HashSet<usize>) {
        let ptr = Rc::as_ptr(rc) as usize;
        let entry = self
            .containers
            .entry(ptr)
            .or_insert_with(|| ContainerEntry {
                kind: ContainerKind::Array(Rc::downgrade(rc)),
                marked: false,
            });
        entry.kind = ContainerKind::Array(Rc::downgrade(rc));
        if !visited.insert(ptr) {
            entry.marked = true;
            return;
        }

        entry.marked = true;
        if let Ok(borrowed) = rc.try_borrow() {
            for value in borrowed.iter() {
                self.mark_value(value, visited);
            }
        }
    }

    fn mark_map(&mut self, rc: &Rc<RefCell<LustMap>>, visited: &mut HashSet<usize>) {
        let ptr = Rc::as_ptr(rc) as usize;
        let entry = self
            .containers
            .entry(ptr)
            .or_insert_with(|| ContainerEntry {
                kind: ContainerKind::Map(Rc::downgrade(rc)),
                marked: false,
            });
        entry.kind = ContainerKind::Map(Rc::downgrade(rc));
        if !visited.insert(ptr) {
            entry.marked = true;
            return;
        }

        entry.marked = true;
        if let Ok(borrowed) = rc.try_borrow() {
            for value in borrowed.values() {
                self.mark_value(value, visited);
            }
        }
    }

    fn mark_struct(&mut self, fields: &Rc<RefCell<Vec<Value>>>, visited: &mut HashSet<usize>) {
        let ptr = Rc::as_ptr(fields) as usize;
        let entry = self
            .containers
            .entry(ptr)
            .or_insert_with(|| ContainerEntry {
                kind: ContainerKind::Struct(Rc::downgrade(fields)),
                marked: false,
            });
        entry.kind = ContainerKind::Struct(Rc::downgrade(fields));
        if !visited.insert(ptr) {
            entry.marked = true;
            return;
        }

        entry.marked = true;
        if let Ok(borrowed) = fields.try_borrow() {
            for value in borrowed.iter() {
                self.mark_value(value, visited);
            }
        }
    }

    fn mark_iterator(&mut self, rc: &Rc<RefCell<IteratorState>>, visited: &mut HashSet<usize>) {
        let ptr = Rc::as_ptr(rc) as usize;
        let entry = self
            .containers
            .entry(ptr)
            .or_insert_with(|| ContainerEntry {
                kind: ContainerKind::Iterator(Rc::downgrade(rc)),
                marked: false,
            });
        entry.kind = ContainerKind::Iterator(Rc::downgrade(rc));
        if !visited.insert(ptr) {
            entry.marked = true;
            return;
        }

        entry.marked = true;
        if let Ok(borrowed) = rc.try_borrow() {
            match &*borrowed {
                IteratorState::Array { items, .. } => {
                    for value in items {
                        self.mark_value(value, visited);
                    }
                }

                IteratorState::MapPairs { items, .. } => {
                    for (_key, value) in items {
                        self.mark_value(value, visited);
                    }
                }
            }
        }
    }

    fn register_array(&mut self, rc: &Rc<RefCell<Vec<Value>>>) -> bool {
        let ptr = Rc::as_ptr(rc) as usize;
        match self.containers.entry(ptr) {
            Entry::Vacant(entry) => {
                entry.insert(ContainerEntry {
                    kind: ContainerKind::Array(Rc::downgrade(rc)),
                    marked: false,
                });
                true
            }

            Entry::Occupied(mut entry) => {
                entry.get_mut().kind = ContainerKind::Array(Rc::downgrade(rc));
                false
            }
        }
    }

    fn register_map(&mut self, rc: &Rc<RefCell<LustMap>>) -> bool {
        let ptr = Rc::as_ptr(rc) as usize;
        match self.containers.entry(ptr) {
            Entry::Vacant(entry) => {
                entry.insert(ContainerEntry {
                    kind: ContainerKind::Map(Rc::downgrade(rc)),
                    marked: false,
                });
                true
            }

            Entry::Occupied(mut entry) => {
                entry.get_mut().kind = ContainerKind::Map(Rc::downgrade(rc));
                false
            }
        }
    }

    fn register_struct(&mut self, fields: &Rc<RefCell<Vec<Value>>>) -> bool {
        let ptr = Rc::as_ptr(fields) as usize;
        match self.containers.entry(ptr) {
            Entry::Vacant(entry) => {
                entry.insert(ContainerEntry {
                    kind: ContainerKind::Struct(Rc::downgrade(fields)),
                    marked: false,
                });
                true
            }

            Entry::Occupied(mut entry) => {
                entry.get_mut().kind = ContainerKind::Struct(Rc::downgrade(fields));
                false
            }
        }
    }

    fn register_iterator(&mut self, iterator: &Rc<RefCell<IteratorState>>) -> bool {
        let ptr = Rc::as_ptr(iterator) as usize;
        match self.containers.entry(ptr) {
            Entry::Vacant(entry) => {
                entry.insert(ContainerEntry {
                    kind: ContainerKind::Iterator(Rc::downgrade(iterator)),
                    marked: false,
                });
                true
            }

            Entry::Occupied(mut entry) => {
                entry.get_mut().kind = ContainerKind::Iterator(Rc::downgrade(iterator));
                false
            }
        }
    }

    fn sweep(&mut self) {
        let mut to_remove = Vec::new();
        for (ptr, entry) in self.containers.iter_mut() {
            if entry.marked {
                entry.marked = false;
                continue;
            }

            match entry.kind.clear() {
                ClearResult::Removed => to_remove.push(*ptr),
                ClearResult::Retain => {}
            }
        }

        for ptr in to_remove {
            self.containers.remove(&ptr);
        }
    }
}

impl ContainerKind {
    fn clear(&self) -> ClearResult {
        match self {
            ContainerKind::Array(weak) => {
                if let Some(rc) = weak.upgrade() {
                    if let Ok(mut borrowed) = rc.try_borrow_mut() {
                        borrowed.clear();
                        ClearResult::Removed
                    } else {
                        ClearResult::Retain
                    }
                } else {
                    ClearResult::Removed
                }
            }

            ContainerKind::Map(weak) => {
                if let Some(rc) = weak.upgrade() {
                    if let Ok(mut borrowed) = rc.try_borrow_mut() {
                        borrowed.clear();
                        ClearResult::Removed
                    } else {
                        ClearResult::Retain
                    }
                } else {
                    ClearResult::Removed
                }
            }

            ContainerKind::Struct(weak) => {
                if let Some(rc) = weak.upgrade() {
                    if let Ok(mut borrowed) = rc.try_borrow_mut() {
                        for value in borrowed.iter_mut() {
                            *value = Value::Nil;
                        }

                        ClearResult::Removed
                    } else {
                        ClearResult::Retain
                    }
                } else {
                    ClearResult::Removed
                }
            }

            ContainerKind::Iterator(weak) => {
                if let Some(rc) = weak.upgrade() {
                    if let Ok(mut borrowed) = rc.try_borrow_mut() {
                        match &mut *borrowed {
                            IteratorState::Array { items, index } => {
                                items.clear();
                                *index = 0;
                            }

                            IteratorState::MapPairs { items, index } => {
                                items.clear();
                                *index = 0;
                            }
                        }

                        ClearResult::Removed
                    } else {
                        ClearResult::Retain
                    }
                } else {
                    ClearResult::Removed
                }
            }
        }
    }
}

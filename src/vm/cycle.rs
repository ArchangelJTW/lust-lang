use crate::bytecode::value::{IteratorState, Upvalue};
use crate::bytecode::{LustMap, Value};
use crate::vm::task::TaskInstance;
use crate::vm::{CallFrame, TaskSignal, VM};
use alloc::rc::{Rc, Weak};
use alloc::vec::Vec;
use core::cell::RefCell;
use hashbrown::{hash_map::Entry, HashMap, HashSet};

const COLLECT_INTERVAL: usize = 512;
const REGISTRATION_THRESHOLD: usize = 256;

type NodeKey = (u8, usize);
const NODE_ARRAY: u8 = 1;
const NODE_MAP: u8 = 2;
const NODE_STRUCT: u8 = 3;
const NODE_ITERATOR: u8 = 4;
const NODE_ENUM_VALUES: u8 = 5;
const NODE_TUPLE_VALUES: u8 = 6;
const NODE_CLOSURE_UPVALUES: u8 = 7;
const NODE_UPVALUE_CELL: u8 = 8;

#[derive(Default)]
pub struct CycleCollector {
    containers: HashMap<NodeKey, ContainerKind>,
    steps_since_collect: usize,
    pending_registrations: usize,
}

enum ContainerKind {
    Array(Weak<RefCell<Vec<Value>>>),
    Map(Weak<RefCell<LustMap>>),
    Struct(Weak<RefCell<Vec<Value>>>),
    Iterator(Weak<RefCell<IteratorState>>),
}

#[derive(Clone)]
enum NodeKind {
    Array(Weak<RefCell<Vec<Value>>>),
    Map(Weak<RefCell<LustMap>>),
    Struct(Weak<RefCell<Vec<Value>>>),
    Iterator(Weak<RefCell<IteratorState>>),
    EnumValues(Weak<Vec<Value>>),
    TupleValues(Weak<Vec<Value>>),
    ClosureUpvalues(Weak<Vec<Upvalue>>),
    UpvalueCell(Weak<RefCell<Value>>),
}

struct Node {
    kind: NodeKind,
    strong_count: usize,
    internal_incoming: usize,
    edges: Vec<NodeKey>,
}

enum ClearResult {
    Removed,
    Retain,
}

impl CycleCollector {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn register_value(&mut self, value: &Value) {
        self.discover_value(value, false);
    }

    pub(crate) fn register_graph(&mut self, value: &Value) {
        self.discover_value(value, false);
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
        self.discover_vm_roots(vm);
        self.collect_registered();
    }

    fn discover_vm_roots(&mut self, vm: &VM) {
        for value in vm.globals.values() {
            self.discover_value(value, true);
        }
        for frame in &vm.call_stack {
            self.discover_frame(frame);
        }
        if let Some(value) = &vm.pending_return_value {
            self.discover_value(value, true);
        }
        if let Some(signal) = &vm.pending_task_signal {
            self.discover_task_signal(signal);
        }
        if let Some(signal) = &vm.last_task_signal {
            self.discover_task_signal(signal);
        }
        for task in vm.task_manager.iter() {
            self.discover_task(task);
        }
    }

    fn discover_frame(&mut self, frame: &CallFrame) {
        for value in frame.registers.iter() {
            self.discover_value(value, true);
        }
        for value in &frame.upvalues {
            self.discover_value(value, true);
        }
    }

    fn discover_task_signal(&mut self, signal: &TaskSignal) {
        match signal {
            TaskSignal::Yield { value, .. } | TaskSignal::Stop { value } => {
                self.discover_value(value, true)
            }
        }
    }

    fn discover_task(&mut self, task: &TaskInstance) {
        for frame in &task.call_stack {
            self.discover_frame(frame);
        }
        if let Some(frame) = task.initial_frame() {
            self.discover_frame(frame);
        }
        if let Some(value) = &task.pending_return_value {
            self.discover_value(value, true);
        }
        if let Some(value) = &task.last_yield {
            self.discover_value(value, true);
        }
        if let Some(value) = &task.last_result {
            self.discover_value(value, true);
        }
    }

    fn discover_value(&mut self, value: &Value, scan_existing: bool) {
        let mut stack = vec![value.clone()];
        let mut visited = HashSet::new();
        while let Some(value) = stack.pop() {
            match value {
                Value::Array(rc) => {
                    let key = (NODE_ARRAY, Rc::as_ptr(&rc) as usize);
                    let registered = self.register_array(&rc);
                    if registered {
                        self.pending_registrations += 1;
                    }
                    if (registered || scan_existing) && visited.insert(key) {
                        if let Ok(values) = rc.try_borrow() {
                            stack.extend(values.iter().cloned());
                        }
                    }
                }
                Value::Map(rc) => {
                    let key = (NODE_MAP, Rc::as_ptr(&rc) as usize);
                    let registered = self.register_map(&rc);
                    if registered {
                        self.pending_registrations += 1;
                    }
                    if (registered || scan_existing) && visited.insert(key) {
                        if let Ok(map) = rc.try_borrow() {
                            for (map_key, value) in map.iter() {
                                let (original, hashed) = map_key.owned_values();
                                stack.push(original.clone());
                                stack.push(hashed.clone());
                                stack.push(value.clone());
                            }
                        }
                    }
                }
                Value::Struct { fields, .. } => {
                    let key = (NODE_STRUCT, Rc::as_ptr(&fields) as usize);
                    let registered = self.register_struct(&fields);
                    if registered {
                        self.pending_registrations += 1;
                    }
                    if (registered || scan_existing) && visited.insert(key) {
                        if let Ok(values) = fields.try_borrow() {
                            stack.extend(values.iter().cloned());
                        }
                    }
                }
                Value::Iterator(rc) => {
                    let key = (NODE_ITERATOR, Rc::as_ptr(&rc) as usize);
                    let registered = self.register_iterator(&rc);
                    if registered {
                        self.pending_registrations += 1;
                    }
                    if (registered || scan_existing) && visited.insert(key) {
                        if let Ok(iterator) = rc.try_borrow() {
                            match &*iterator {
                                IteratorState::Array { items, .. } => {
                                    stack.extend(items.iter().cloned());
                                }
                                IteratorState::MapPairs { items, .. } => {
                                    for (map_key, value) in items {
                                        let (original, hashed) = map_key.owned_values();
                                        stack.push(original.clone());
                                        stack.push(hashed.clone());
                                        stack.push(value.clone());
                                    }
                                }
                            }
                        }
                    }
                }
                Value::Tuple(values) => {
                    let key = (NODE_TUPLE_VALUES, Rc::as_ptr(&values) as usize);
                    if visited.insert(key) {
                        stack.extend(values.iter().cloned());
                    }
                }
                Value::Enum {
                    values: Some(values),
                    ..
                } => {
                    let key = (NODE_ENUM_VALUES, Rc::as_ptr(&values) as usize);
                    if visited.insert(key) {
                        stack.extend(values.iter().cloned());
                    }
                }
                Value::Closure { upvalues, .. } => {
                    let key = (NODE_CLOSURE_UPVALUES, Rc::as_ptr(&upvalues) as usize);
                    if visited.insert(key) {
                        stack.extend(upvalues.iter().map(Upvalue::get));
                    }
                }
                Value::WeakStruct(_) => {}
                _ => {}
            }
        }
    }

    fn collect_registered(&mut self) {
        if self.containers.is_empty() {
            return;
        }

        let Some(nodes) = self.snapshot_graph() else {
            return;
        };

        let mut live = HashSet::new();
        let mut stack = Vec::new();
        for (key, node) in &nodes {
            if node.strong_count > node.internal_incoming {
                stack.push(*key);
            }
        }

        while let Some(key) = stack.pop() {
            if !live.insert(key) {
                continue;
            }
            if let Some(node) = nodes.get(&key) {
                stack.extend(node.edges.iter().copied());
            }
        }

        self.sweep(&live);
    }

    fn snapshot_graph(&mut self) -> Option<HashMap<NodeKey, Node>> {
        let mut nodes = HashMap::new();

        // Seed every registered container. The temporary upgrade contributes one
        // strong reference, so exclude it from the exact count being recorded.
        for (key, container) in &self.containers {
            match container {
                ContainerKind::Array(weak) => {
                    if let Some(rc) = weak.upgrade() {
                        Self::insert_node(
                            &mut nodes,
                            *key,
                            Rc::strong_count(&rc) - 1,
                            NodeKind::Array(Rc::downgrade(&rc)),
                        );
                    }
                }
                ContainerKind::Map(weak) => {
                    if let Some(rc) = weak.upgrade() {
                        Self::insert_node(
                            &mut nodes,
                            *key,
                            Rc::strong_count(&rc) - 1,
                            NodeKind::Map(Rc::downgrade(&rc)),
                        );
                    }
                }
                ContainerKind::Struct(weak) => {
                    if let Some(rc) = weak.upgrade() {
                        Self::insert_node(
                            &mut nodes,
                            *key,
                            Rc::strong_count(&rc) - 1,
                            NodeKind::Struct(Rc::downgrade(&rc)),
                        );
                    }
                }
                ContainerKind::Iterator(weak) => {
                    if let Some(rc) = weak.upgrade() {
                        Self::insert_node(
                            &mut nodes,
                            *key,
                            Rc::strong_count(&rc) - 1,
                            NodeKind::Iterator(Rc::downgrade(&rc)),
                        );
                    }
                }
            }
        }

        let mut scanned = HashSet::new();
        while let Some(key) = nodes.keys().find(|key| !scanned.contains(*key)).copied() {
            scanned.insert(key);
            if !self.scan_node(key, &mut nodes) {
                return None;
            }
        }

        Some(nodes)
    }

    fn scan_node(&mut self, key: NodeKey, nodes: &mut HashMap<NodeKey, Node>) -> bool {
        match nodes[&key].kind.clone() {
            NodeKind::Array(weak) => {
                let Some(rc) = weak.upgrade() else {
                    return true;
                };
                let Ok(values) = rc.try_borrow() else {
                    return false;
                };
                for value in values.iter() {
                    self.scan_value(key, value, nodes);
                }
            }
            NodeKind::Map(weak) => {
                let Some(rc) = weak.upgrade() else {
                    return true;
                };
                let Ok(map) = rc.try_borrow() else {
                    return false;
                };
                for (map_key, value) in map.iter() {
                    let (original, hashed) = map_key.owned_values();
                    self.scan_value(key, original, nodes);
                    self.scan_value(key, hashed, nodes);
                    self.scan_value(key, value, nodes);
                }
            }
            NodeKind::Struct(weak) => {
                let Some(rc) = weak.upgrade() else {
                    return true;
                };
                let Ok(values) = rc.try_borrow() else {
                    return false;
                };
                for value in values.iter() {
                    self.scan_value(key, value, nodes);
                }
            }
            NodeKind::Iterator(weak) => {
                let Some(rc) = weak.upgrade() else {
                    return true;
                };
                let Ok(iterator) = rc.try_borrow() else {
                    return false;
                };
                match &*iterator {
                    IteratorState::Array { items, .. } => {
                        for value in items {
                            self.scan_value(key, value, nodes);
                        }
                    }
                    IteratorState::MapPairs { items, .. } => {
                        for (map_key, value) in items {
                            let (original, hashed) = map_key.owned_values();
                            self.scan_value(key, original, nodes);
                            self.scan_value(key, hashed, nodes);
                            self.scan_value(key, value, nodes);
                        }
                    }
                }
            }
            NodeKind::EnumValues(weak) | NodeKind::TupleValues(weak) => {
                let Some(values) = weak.upgrade() else {
                    return true;
                };
                for value in values.iter() {
                    self.scan_value(key, value, nodes);
                }
            }
            NodeKind::ClosureUpvalues(weak) => {
                let Some(upvalues) = weak.upgrade() else {
                    return true;
                };
                for upvalue in upvalues.iter() {
                    let cell = upvalue.cell();
                    let child = (NODE_UPVALUE_CELL, Rc::as_ptr(cell) as usize);
                    Self::add_edge(
                        nodes,
                        key,
                        child,
                        Rc::strong_count(cell),
                        NodeKind::UpvalueCell(Rc::downgrade(cell)),
                    );
                }
            }
            NodeKind::UpvalueCell(weak) => {
                let Some(cell) = weak.upgrade() else {
                    return true;
                };
                let Ok(value) = cell.try_borrow() else {
                    return false;
                };
                self.scan_value(key, &value, nodes);
            }
        }

        true
    }

    fn scan_value(&mut self, owner: NodeKey, value: &Value, nodes: &mut HashMap<NodeKey, Node>) {
        match value {
            Value::Array(rc) => {
                self.register_array(rc);
                let child = (NODE_ARRAY, Rc::as_ptr(rc) as usize);
                Self::add_edge(
                    nodes,
                    owner,
                    child,
                    Rc::strong_count(rc),
                    NodeKind::Array(Rc::downgrade(rc)),
                );
            }
            Value::Map(rc) => {
                self.register_map(rc);
                let child = (NODE_MAP, Rc::as_ptr(rc) as usize);
                Self::add_edge(
                    nodes,
                    owner,
                    child,
                    Rc::strong_count(rc),
                    NodeKind::Map(Rc::downgrade(rc)),
                );
            }
            Value::Struct { fields, .. } => {
                self.register_struct(fields);
                let child = (NODE_STRUCT, Rc::as_ptr(fields) as usize);
                Self::add_edge(
                    nodes,
                    owner,
                    child,
                    Rc::strong_count(fields),
                    NodeKind::Struct(Rc::downgrade(fields)),
                );
            }
            Value::Iterator(rc) => {
                self.register_iterator(rc);
                let child = (NODE_ITERATOR, Rc::as_ptr(rc) as usize);
                Self::add_edge(
                    nodes,
                    owner,
                    child,
                    Rc::strong_count(rc),
                    NodeKind::Iterator(Rc::downgrade(rc)),
                );
            }
            Value::Enum {
                values: Some(values),
                ..
            } => {
                let child = (NODE_ENUM_VALUES, Rc::as_ptr(values) as usize);
                Self::add_edge(
                    nodes,
                    owner,
                    child,
                    Rc::strong_count(values),
                    NodeKind::EnumValues(Rc::downgrade(values)),
                );
            }
            Value::Tuple(values) => {
                let child = (NODE_TUPLE_VALUES, Rc::as_ptr(values) as usize);
                Self::add_edge(
                    nodes,
                    owner,
                    child,
                    Rc::strong_count(values),
                    NodeKind::TupleValues(Rc::downgrade(values)),
                );
            }
            Value::Closure { upvalues, .. } => {
                let child = (NODE_CLOSURE_UPVALUES, Rc::as_ptr(upvalues) as usize);
                Self::add_edge(
                    nodes,
                    owner,
                    child,
                    Rc::strong_count(upvalues),
                    NodeKind::ClosureUpvalues(Rc::downgrade(upvalues)),
                );
            }
            Value::WeakStruct(_) => {}
            _ => {}
        }
    }

    fn insert_node(
        nodes: &mut HashMap<NodeKey, Node>,
        key: NodeKey,
        strong_count: usize,
        kind: NodeKind,
    ) {
        nodes.entry(key).or_insert(Node {
            kind,
            strong_count,
            internal_incoming: 0,
            edges: Vec::new(),
        });
    }

    fn add_edge(
        nodes: &mut HashMap<NodeKey, Node>,
        owner: NodeKey,
        child: NodeKey,
        strong_count: usize,
        kind: NodeKind,
    ) {
        Self::insert_node(nodes, child, strong_count, kind);
        nodes.get_mut(&owner).unwrap().edges.push(child);
        nodes.get_mut(&child).unwrap().internal_incoming += 1;
    }

    fn register_array(&mut self, rc: &Rc<RefCell<Vec<Value>>>) -> bool {
        self.register_container(
            (NODE_ARRAY, Rc::as_ptr(rc) as usize),
            ContainerKind::Array(Rc::downgrade(rc)),
        )
    }

    fn register_map(&mut self, rc: &Rc<RefCell<LustMap>>) -> bool {
        self.register_container(
            (NODE_MAP, Rc::as_ptr(rc) as usize),
            ContainerKind::Map(Rc::downgrade(rc)),
        )
    }

    fn register_struct(&mut self, rc: &Rc<RefCell<Vec<Value>>>) -> bool {
        self.register_container(
            (NODE_STRUCT, Rc::as_ptr(rc) as usize),
            ContainerKind::Struct(Rc::downgrade(rc)),
        )
    }

    fn register_iterator(&mut self, rc: &Rc<RefCell<IteratorState>>) -> bool {
        self.register_container(
            (NODE_ITERATOR, Rc::as_ptr(rc) as usize),
            ContainerKind::Iterator(Rc::downgrade(rc)),
        )
    }

    fn register_container(&mut self, key: NodeKey, kind: ContainerKind) -> bool {
        match self.containers.entry(key) {
            Entry::Vacant(entry) => {
                entry.insert(kind);
                true
            }
            Entry::Occupied(mut entry) => {
                *entry.get_mut() = kind;
                false
            }
        }
    }

    fn sweep(&mut self, live: &HashSet<NodeKey>) {
        let mut to_remove = Vec::new();
        for (key, container) in &self.containers {
            if live.contains(key) {
                continue;
            }
            if matches!(container.clear(), ClearResult::Removed) {
                to_remove.push(*key);
            }
        }

        for key in to_remove {
            self.containers.remove(&key);
        }
    }
}

impl ContainerKind {
    fn clear(&self) -> ClearResult {
        match self {
            ContainerKind::Array(weak) => {
                let Some(rc) = weak.upgrade() else {
                    return ClearResult::Removed;
                };
                let Ok(mut values) = rc.try_borrow_mut() else {
                    return ClearResult::Retain;
                };
                values.clear();
                ClearResult::Removed
            }
            ContainerKind::Map(weak) => {
                let Some(rc) = weak.upgrade() else {
                    return ClearResult::Removed;
                };
                let Ok(mut map) = rc.try_borrow_mut() else {
                    return ClearResult::Retain;
                };
                map.clear();
                ClearResult::Removed
            }
            ContainerKind::Struct(weak) => {
                let Some(rc) = weak.upgrade() else {
                    return ClearResult::Removed;
                };
                let Ok(mut fields) = rc.try_borrow_mut() else {
                    return ClearResult::Retain;
                };
                for value in fields.iter_mut() {
                    *value = Value::Nil;
                }
                ClearResult::Removed
            }
            ContainerKind::Iterator(weak) => {
                let Some(rc) = weak.upgrade() else {
                    return ClearResult::Removed;
                };
                let Ok(mut iterator) = rc.try_borrow_mut() else {
                    return ClearResult::Retain;
                };
                match &mut *iterator {
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
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::bytecode::value::WeakStructRef;
    use crate::bytecode::{StructLayout, Upvalue, ValueKey};
    use alloc::string::ToString;

    #[test]
    fn externally_held_direct_cycle_is_preserved() {
        let mut collector = CycleCollector::new();
        let value = Value::array(Vec::new());
        value.array_push(value.clone()).unwrap();
        collector.register_value(&value);

        collector.collect_registered();

        assert_eq!(value.array_len(), Some(1));
    }

    #[test]
    fn externally_held_tuple_and_enum_cycles_are_preserved() {
        let mut collector = CycleCollector::new();

        let tuple_array = Value::array(Vec::new());
        let tuple = Value::tuple(vec![tuple_array.clone()]);
        tuple_array.array_push(tuple.clone()).unwrap();
        collector.register_value(&tuple_array);
        let tuple_weak = match &tuple_array {
            Value::Array(rc) => Rc::downgrade(rc),
            _ => unreachable!(),
        };
        drop(tuple_array);

        let enum_array = Value::array(Vec::new());
        let enum_value = Value::enum_variant("Loop", "Value", vec![enum_array.clone()]);
        enum_array.array_push(enum_value.clone()).unwrap();
        collector.register_value(&enum_array);
        let enum_weak = match &enum_array {
            Value::Array(rc) => Rc::downgrade(rc),
            _ => unreachable!(),
        };
        drop(enum_array);

        collector.collect_registered();

        assert_eq!(tuple_weak.upgrade().unwrap().borrow().len(), 1);
        assert_eq!(enum_weak.upgrade().unwrap().borrow().len(), 1);
        drop((tuple, enum_value));
    }

    #[test]
    fn externally_held_upvalue_cell_preserves_cycle() {
        let mut collector = CycleCollector::new();
        let array = Value::array(Vec::new());
        let upvalue = Upvalue::new(Value::Nil);
        let closure = Value::Closure {
            function_idx: 0,
            upvalues: Rc::new(vec![upvalue.clone()]),
        };
        array.array_push(closure).unwrap();
        upvalue.set(array.clone());
        collector.register_value(&array);
        let weak = match &array {
            Value::Array(rc) => Rc::downgrade(rc),
            _ => unreachable!(),
        };
        drop(array);

        collector.collect_registered();

        assert_eq!(weak.upgrade().unwrap().borrow().len(), 1);
        drop(upvalue);
    }

    #[test]
    fn unreachable_closed_cycle_is_collected() {
        let mut collector = CycleCollector::new();
        let value = Value::array(Vec::new());
        value.array_push(value.clone()).unwrap();
        collector.register_value(&value);
        let weak = match &value {
            Value::Array(rc) => Rc::downgrade(rc),
            _ => unreachable!(),
        };
        drop(value);

        collector.collect_registered();

        assert!(weak.upgrade().is_none());
    }

    #[test]
    fn weak_struct_is_a_leaf() {
        let mut collector = CycleCollector::new();
        let layout = Rc::new(StructLayout::new(
            "Node".to_string(),
            Vec::new(),
            Vec::new(),
            Vec::new(),
            Vec::new(),
        ));
        let fields = Rc::new(RefCell::new(vec![Value::Nil]));
        let value = Value::Struct {
            name: "Node".to_string(),
            layout: layout.clone(),
            fields: fields.clone(),
        };
        fields.borrow_mut()[0] = value.clone();
        let weak_value = Value::WeakStruct(WeakStructRef::new("Node".to_string(), layout, &fields));
        collector.register_value(&value);
        drop((value, fields));

        collector.collect_registered();

        match weak_value {
            Value::WeakStruct(weak) => assert!(weak.upgrade().is_none()),
            _ => unreachable!(),
        }
    }

    #[test]
    fn both_owned_map_key_values_are_internal_edges() {
        let mut collector = CycleCollector::new();
        let array = Value::array(Vec::new());
        let map = Value::map(HashMap::default());
        map.map_set(
            ValueKey::with_hashed(array.clone(), array.clone()),
            array.clone(),
        )
        .unwrap();
        array.array_push(map.clone()).unwrap();
        collector.register_value(&array);
        collector.register_value(&map);
        let array_weak = match &array {
            Value::Array(rc) => Rc::downgrade(rc),
            _ => unreachable!(),
        };
        let map_weak = match &map {
            Value::Map(rc) => Rc::downgrade(rc),
            _ => unreachable!(),
        };
        drop((array, map));

        collector.collect_registered();

        assert!(array_weak.upgrade().is_none());
        assert!(map_weak.upgrade().is_none());
    }

    #[test]
    fn dropping_vm_collects_cycles_below_periodic_threshold() {
        let weak = {
            let mut vm = VM::new();
            let value = Value::array(Vec::new());
            value.array_push(value.clone()).unwrap();
            let weak = match &value {
                Value::Array(rc) => Rc::downgrade(rc),
                _ => unreachable!(),
            };
            vm.set_global("cycle", value);
            weak
        };

        assert!(weak.upgrade().is_none());
    }

    #[test]
    fn dropping_vm_collects_cycle_nested_in_function_constant() {
        let array = Value::array(Vec::new());
        let tuple = Value::tuple(vec![array.clone()]);
        array.array_push(tuple.clone()).unwrap();
        let weak = match &array {
            Value::Array(rc) => Rc::downgrade(rc),
            _ => unreachable!(),
        };

        let mut function = crate::bytecode::Function::new("constant_owner", 0, false);
        function.chunk.add_constant(tuple);
        let mut vm = VM::new();
        vm.load_functions(vec![function]);
        drop(array);
        drop(vm);

        assert!(weak.upgrade().is_none());
    }
}

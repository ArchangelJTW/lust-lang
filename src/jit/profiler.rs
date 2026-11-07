use alloc::vec::Vec;
use hashbrown::HashMap;
pub struct Profiler {
    backedges: HashMap<(usize, usize), u32>,
    hot_spots: Vec<HotSpot>,
    loop_hierarchy_cache: HashMap<(usize, usize), bool>,
}

#[derive(Debug, Clone)]
pub struct HotSpot {
    pub function_idx: usize,
    pub start_ip: usize,
    pub iterations: u32,
}

impl Profiler {
    pub fn new() -> Self {
        Self {
            backedges: HashMap::new(),
            hot_spots: Vec::new(),
            loop_hierarchy_cache: HashMap::new(),
        }
    }

    pub fn get_cached_loop_hierarchy(&self, func_idx: usize, loop_start_ip: usize) -> Option<bool> {
        self.loop_hierarchy_cache
            .get(&(func_idx, loop_start_ip))
            .copied()
    }

    pub fn set_cached_loop_hierarchy(
        &mut self,
        func_idx: usize,
        loop_start_ip: usize,
        is_hierarchy: bool,
    ) {
        self.loop_hierarchy_cache
            .insert((func_idx, loop_start_ip), is_hierarchy);
    }

    pub fn record_backedge(&mut self, func_idx: usize, ip: usize) -> u32 {
        let count = self
            .backedges
            .entry((func_idx, ip))
            .and_modify(|c| *c += 1)
            .or_insert(1);
        *count
    }

    pub fn is_hot(&self, func_idx: usize, ip: usize, threshold: u32) -> bool {
        self.backedges
            .get(&(func_idx, ip))
            .map(|&count| count >= threshold)
            .unwrap_or(false)
    }

    pub fn get_count(&self, func_idx: usize, ip: usize) -> u32 {
        self.backedges.get(&(func_idx, ip)).copied().unwrap_or(0)
    }

    pub fn mark_hot(&mut self, func_idx: usize, ip: usize) {
        let iterations = self.get_count(func_idx, ip);
        self.hot_spots.push(HotSpot {
            function_idx: func_idx,
            start_ip: ip,
            iterations,
        });
    }

    pub fn hot_spots(&self) -> &[HotSpot] {
        &self.hot_spots
    }

    pub fn reset(&mut self) {
        self.backedges.clear();
        self.hot_spots.clear();
        self.loop_hierarchy_cache.clear();
    }
}

impl Default for Profiler {
    fn default() -> Self {
        Self::new()
    }
}

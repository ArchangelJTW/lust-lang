use core::alloc::{AllocError, Allocator, Layout};
use lust::alloc_budget::{
    BudgetAllocError, LustBudgetAlloc, LustChunkArena, StaticSlabAllocator, VmAllocHandle,
    DEFAULT_CHUNK_SIZE,
};
use lust::data::budget::{BudgetError, BudgetTracker};
use lust::data::materialize::{AlignedStorage, MaterializeError};

#[test]
fn test_zst_allocations_charge_zero_bytes() {
    let alloc = LustBudgetAlloc::with_limit(64);

    // Spec requirement (1): Vec::<(), &LustBudgetAlloc>::with_capacity_in(100, &alloc)
    let mut v: Vec<(), &LustBudgetAlloc> = Vec::with_capacity_in(100, &alloc);
    for _ in 0..100 {
        v.push(());
    }
    assert_eq!(v.len(), 100);
    assert_eq!(alloc.used_bytes(), 0);
    assert_eq!(alloc.peak_bytes(), 0);
    drop(v);
    assert_eq!(alloc.used_bytes(), 0);

    // Box of ZST
    let b: Box<(), &LustBudgetAlloc> = Box::new_in((), &alloc);
    assert_eq!(alloc.used_bytes(), 0);
    drop(b);
    assert_eq!(alloc.used_bytes(), 0);

    // Direct ZST Allocator contract (over-aligned ZST, 0->0 grow, 0->0 shrink, deallocate no-op)
    let zst_layout = Layout::from_size_align(0, 64).unwrap();
    let ptr = alloc.allocate(zst_layout).expect("ZST allocate succeeds");
    assert_eq!(ptr.len(), 0);
    assert_eq!((ptr.as_ptr() as *mut u8 as usize) % 64, 0);
    assert_eq!(alloc.used_bytes(), 0);

    unsafe {
        let grown = alloc
            .grow(ptr.cast::<u8>(), zst_layout, zst_layout)
            .expect("ZST 0->0 grow succeeds");
        assert_eq!(grown.len(), 0);
        assert_eq!(alloc.used_bytes(), 0);

        let shrunk = alloc
            .shrink(grown.cast::<u8>(), zst_layout, zst_layout)
            .expect("ZST 0->0 shrink succeeds");
        assert_eq!(shrunk.len(), 0);
        assert_eq!(alloc.used_bytes(), 0);

        // Must NOT call Global.deallocate for size == 0
        alloc.deallocate(shrunk.cast::<u8>(), zst_layout);
    }
    assert_eq!(alloc.used_bytes(), 0);

    // Grow 0 -> 32, then shrink 32 -> 0
    let l32 = Layout::from_size_align(32, 8).unwrap();
    unsafe {
        let z = alloc.allocate(zst_layout).unwrap();
        let p32 = alloc.grow(z.cast::<u8>(), zst_layout, l32).unwrap();
        assert_eq!(alloc.used_bytes(), 32);
        let z2 = alloc.shrink(p32.cast::<u8>(), l32, zst_layout).unwrap();
        assert_eq!(z2.len(), 0);
        assert_eq!(alloc.used_bytes(), 0);
        alloc.deallocate(z2.cast::<u8>(), zst_layout);
    }
    assert_eq!(alloc.used_bytes(), 0);
    assert_eq!(alloc.peak_bytes(), 32);
}

#[test]
fn test_oom_rollback_on_grow_and_drop_refund() {
    // Spec requirement (2): limit 1024 bytes, grow fails -> used_bytes stays 512, drop -> 0
    let alloc = LustBudgetAlloc::with_limit(1024);

    {
        let mut v: Vec<u8, &LustBudgetAlloc> = Vec::new_in(&alloc);
        v.try_reserve_exact(512).expect("512 <= 1024 succeeds");
        assert_eq!(alloc.used_bytes(), 512);
        assert_eq!(alloc.peak_bytes(), 512);

        for i in 0..512usize {
            v.push((i & 0xFF) as u8);
        }
        assert_eq!(alloc.used_bytes(), 512);

        // Attempt to grow beyond 1024 bytes (512 + 513 = 1025 > 1024)
        let res = v.try_reserve_exact(513);
        assert!(res.is_err(), "growing past 1024-byte limit must fail");
        assert_eq!(
            alloc.used_bytes(),
            512,
            "OOM on grow must leave used_bytes at 512"
        );
        assert_eq!(alloc.peak_bytes(), 512);

        // Verify original data was preserved intact
        for i in 0..512usize {
            assert_eq!(v[i], (i & 0xFF) as u8);
        }
    }

    assert_eq!(alloc.used_bytes(), 0, "drop must refund all 512 bytes");
    assert_eq!(alloc.peak_bytes(), 512);

    // Also test direct raw Allocator::grow OOM rollback & shrink refund
    let l512 = Layout::from_size_align(512, 8).unwrap();
    let l1025 = Layout::from_size_align(1025, 8).unwrap();
    let l256 = Layout::from_size_align(256, 8).unwrap();
    let p = alloc.allocate(l512).unwrap();
    assert_eq!(alloc.used_bytes(), 512);
    unsafe {
        let err = alloc.grow(p.cast::<u8>(), l512, l1025);
        assert!(matches!(err, Err(AllocError)));
        assert_eq!(alloc.used_bytes(), 512);

        let shrunk = alloc.shrink(p.cast::<u8>(), l512, l256).unwrap();
        assert_eq!(alloc.used_bytes(), 256);

        alloc.deallocate(shrunk.cast::<u8>(), l256);
    }
    assert_eq!(alloc.used_bytes(), 0);
}

#[test]
fn test_vm_alloc_handle_cloning_and_budget() {
    let handle = VmAllocHandle::with_limit(2048);
    let handle2 = handle.clone();

    let mut v: Vec<u64, VmAllocHandle> = Vec::new_in(handle.clone());
    v.try_reserve_exact(64).expect("512 bytes <= 2048");
    assert_eq!(handle2.0.used_bytes(), 512);

    for i in 0..64u64 {
        v.push(i * 3);
    }
    assert_eq!(v[63], 189);

    // Exceeding limit fails cleanly
    assert!(v.try_reserve_exact(256).is_err());
    assert_eq!(handle.0.used_bytes(), 512);

    drop(v);
    assert_eq!(handle.0.used_bytes(), 0);
    assert_eq!(handle2.0.peak_bytes(), 512);
}

#[test]
fn test_lust_chunk_arena_bulk_free_and_budget() {
    let parent = LustBudgetAlloc::with_limit(DEFAULT_CHUNK_SIZE * 2);

    {
        let arena = LustChunkArena::new(&parent);
        assert_eq!(parent.used_bytes(), 0);

        // ZST inside arena does not allocate a chunk
        let z: Box<(), &LustChunkArena<'_>> = Box::new_in((), &arena);
        assert_eq!(parent.used_bytes(), 0);
        drop(z);

        // Allocate a tree of boxed nodes in the arena
        struct Node<'a> {
            val: i64,
            left: Option<Box<Node<'a>, &'a LustChunkArena<'a>>>,
            right: Option<Box<Node<'a>, &'a LustChunkArena<'a>>>,
        }

        fn build<'a>(
            depth: u32,
            val: i64,
            arena: &'a LustChunkArena<'a>,
        ) -> Box<Node<'a>, &'a LustChunkArena<'a>> {
            if depth == 0 {
                Box::new_in(
                    Node {
                        val,
                        left: None,
                        right: None,
                    },
                    arena,
                )
            } else {
                Box::new_in(
                    Node {
                        val,
                        left: Some(build(depth - 1, val * 2, arena)),
                        right: Some(build(depth - 1, val * 2 + 1, arena)),
                    },
                    arena,
                )
            }
        }

        fn sum(node: &Node<'_>) -> i64 {
            node.val
                + node.left.as_deref().map_or(0, sum)
                + node.right.as_deref().map_or(0, sum)
        }

        let root = build(8, 1, &arena);
        assert_eq!(sum(&root), (1i64..=511).sum::<i64>());
        // All 511 nodes (~20 KiB) fit in 1 default 256 KiB chunk
        assert_eq!(parent.used_bytes(), DEFAULT_CHUNK_SIZE);

        // Dropping individual boxes is a no-op until the arena itself drops
        drop(root);
        assert_eq!(parent.used_bytes(), DEFAULT_CHUNK_SIZE);

        // Over-aligned allocation works
        let aligned_layout = Layout::from_size_align(64, 64).unwrap();
        let ptr = arena.allocate(aligned_layout).unwrap();
        assert_eq!((ptr.as_ptr() as *mut u8 as usize) % 64, 0);

        // Oversized allocation exceeding remaining parent budget fails without corrupting parent
        let huge_layout = Layout::from_size_align(DEFAULT_CHUNK_SIZE * 2, 8).unwrap();
        assert!(arena.allocate(huge_layout).is_err());
        assert_eq!(parent.used_bytes(), DEFAULT_CHUNK_SIZE);
    }

    // Arena Drop releases every chunk back to parent
    assert_eq!(parent.used_bytes(), 0);
    assert_eq!(parent.peak_bytes(), DEFAULT_CHUNK_SIZE);
}

#[test]
fn test_static_slab_allocator_4096() {
    let slab = StaticSlabAllocator::<4096>::new();
    assert_eq!(slab.capacity(), 4096);
    assert_eq!(slab.used_bytes(), 0);

    // ZST does not advance offset
    let z: Box<(), &StaticSlabAllocator<4096>> = Box::new_in((), &slab);
    assert_eq!(slab.used_bytes(), 0);
    drop(z);

    {
        // Multiple simultaneous live mutable allocations (validates no &mut [u8; N] aliasing UB)
        let mut b1: Box<[u64; 16], &StaticSlabAllocator<4096>> = Box::new_in([1u64; 16], &slab);
        let mut b2: Box<[u64; 16], &StaticSlabAllocator<4096>> = Box::new_in([2u64; 16], &slab);
        let mut v: Vec<u32, &StaticSlabAllocator<4096>> = Vec::new_in(&slab);
        v.try_reserve_exact(64).expect("256 bytes fits in slab");
        for i in 0..64u32 {
            v.push(i);
        }

        // Mutate both boxes concurrently while vec is live
        b1[0] = 42;
        b2[15] = 99;
        assert_eq!(b1[0], 42);
        assert_eq!(b1[15], 1);
        assert_eq!(b2[0], 2);
        assert_eq!(b2[15], 99);
        assert_eq!(v.iter().sum::<u32>(), (0..64u32).sum::<u32>());

        // Over-aligned allocation (64-byte alignment)
        let l64 = Layout::from_size_align(64, 64).unwrap();
        let p64 = slab.allocate(l64).expect("64-byte aligned alloc fits");
        assert_eq!((p64.as_ptr() as *mut u8 as usize) % 64, 0);

        // Allocation exceeding remaining capacity fails without changing offset
        let used_before = slab.used_bytes();
        let too_big = Layout::from_size_align(4096, 8).unwrap();
        assert!(slab.allocate(too_big).is_err());
        assert_eq!(slab.used_bytes(), used_before);
    }

    // All allocations dropped; reset reclaims slab
    unsafe { slab.reset() };
    assert_eq!(slab.used_bytes(), 0);

    // Can allocate full 4096 bytes after reset (with alignment 1)
    let full = Layout::from_size_align(4096, 1).unwrap();
    let p = slab.allocate(full).expect("full 4096-byte slab fits");
    assert_eq!(p.len(), 4096);
    assert_eq!(slab.used_bytes(), 4096);
    assert!(slab.allocate(Layout::from_size_align(1, 1).unwrap()).is_err());
}

#[test]
fn test_aligned_storage_new_zeroed_in_and_drop_refund() {
    let handle = VmAllocHandle::with_limit(512);

    // Zero-sized storage charges 0 bytes and stays properly aligned
    {
        let z = AlignedStorage::new_zeroed_in(0, 16, handle.clone()).expect("ZST storage succeeds");
        assert_eq!(z.len(), 0);
        assert_eq!(z.align(), 16);
        assert_eq!((z.as_ptr() as usize) % 16, 0);
        assert!(z.as_slice().is_empty());
        assert_eq!(handle.0.used_bytes(), 0);
    }
    assert_eq!(handle.0.used_bytes(), 0);

    {
        let mut storage =
            AlignedStorage::new_zeroed_in(256, 32, handle.clone()).expect("256 <= 512 succeeds");
        assert_eq!(storage.len(), 256);
        assert_eq!(storage.align(), 32);
        assert_eq!((storage.as_ptr() as usize) % 32, 0);
        assert!(storage.as_slice().iter().all(|&b| b == 0));
        assert_eq!(handle.0.used_bytes(), 256);

        storage.as_mut_slice()[0] = 0xAB;
        storage.as_mut_slice()[255] = 0xCD;
        assert_eq!(storage.as_slice()[0], 0xAB);
        assert_eq!(storage.as_slice()[255], 0xCD);

        // Exceeding budget fails closed and does not leak quota
        let err = AlignedStorage::new_zeroed_in(300, 16, handle.clone()).unwrap_err();
        assert!(matches!(
            err,
            MaterializeError::BufferTooSmall {
                required: 556,
                actual: 512
            }
        ));
        assert_eq!(handle.0.used_bytes(), 256);
    }

    assert_eq!(handle.0.used_bytes(), 0, "Drop must refund 256 bytes");
    assert_eq!(handle.0.peak_bytes(), 256);
}

#[test]
fn test_budget_tracker_charge_and_release_with_alloc() {
    // From<BudgetAllocError> for BudgetError
    let conv: BudgetError = BudgetAllocError {
        limit: 100,
        requested: 150,
    }
    .into();
    assert_eq!(
        conv,
        BudgetError::BytesExceeded {
            limit: 100,
            requested: 150
        }
    );

    let alloc = LustBudgetAlloc::with_limit(500);
    let mut tracker = BudgetTracker::new(1000, 10_000, 32);

    // Synchronous charge
    tracker
        .charge_bytes_with_alloc(&alloc, 300)
        .expect("300 fits both tracker and alloc");
    assert_eq!(tracker.allocated_bytes(), 300);
    assert_eq!(alloc.used_bytes(), 300);

    // Alloc limit exceeded (300 + 300 = 600 > 500) -> transactional rollback of tracker!
    let err = tracker.charge_bytes_with_alloc(&alloc, 300).unwrap_err();
    assert_eq!(
        err,
        BudgetError::BytesExceeded {
            limit: 500,
            requested: 600
        }
    );
    assert_eq!(
        tracker.allocated_bytes(),
        300,
        "tracker must roll back on alloc failure"
    );
    assert_eq!(alloc.used_bytes(), 300);

    // Tracker limit tighter than alloc limit -> neither modified on failure
    let big_alloc = LustBudgetAlloc::with_limit(2000);
    let mut small_tracker = BudgetTracker::new(400, 10_000, 32);
    small_tracker
        .charge_bytes_with_alloc(&big_alloc, 250)
        .unwrap();
    let err2 = small_tracker
        .charge_bytes_with_alloc(&big_alloc, 200)
        .unwrap_err();
    assert_eq!(
        err2,
        BudgetError::BytesExceeded {
            limit: 400,
            requested: 450
        }
    );
    assert_eq!(small_tracker.allocated_bytes(), 250);
    assert_eq!(big_alloc.used_bytes(), 250);

    // Synchronous release
    tracker.release_bytes_with_alloc(&alloc, 300);
    assert_eq!(tracker.allocated_bytes(), 0);
    assert_eq!(alloc.used_bytes(), 0);
}

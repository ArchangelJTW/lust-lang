#ifndef LUST_LAYOUT_ABI_V1_H
#define LUST_LAYOUT_ABI_V1_H

#include <stdalign.h>
#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define LUST_ABI_VERSION_V1 1u

/* RFC Section 10 / 10.2 canonical status codes (fixed-width uint32_t) */
#define LUST_ABI_STATUS_OK 0u
#define LUST_ABI_STATUS_VALIDATION_FAILED 1u
#define LUST_ABI_STATUS_PARSE_FAILED 2u
#define LUST_ABI_STATUS_INVALID_HANDLE 3u
#define LUST_ABI_STATUS_INVALID_ABI 4u
#define LUST_ABI_STATUS_NEED_CAPACITY 5u
#define LUST_ABI_STATUS_INTERRUPTED 6u
#define LUST_ABI_STATUS_DEFECT 7u
#define LUST_ABI_STATUS_OUT_OF_RANGE 8u
#define LUST_ABI_STATUS_DENIED 9u

/* Storage kind tags matching CanonicalTypeTag */
#define LUST_STORAGE_BOOL 1u
#define LUST_STORAGE_INT64 2u
#define LUST_STORAGE_UINT64 3u
#define LUST_STORAGE_FLOAT64 4u
#define LUST_STORAGE_DECIMAL128 5u
#define LUST_STORAGE_OBJECT_ID 6u
#define LUST_STORAGE_DATETIME_UTC 7u
#define LUST_STORAGE_STRING_SLICE 8u
#define LUST_STORAGE_BINARY_SLICE 9u
#define LUST_STORAGE_NESTED_STRUCT 10u

/* Binding / ownership flags */
#define LUST_BINDING_FLAG_DIRECT (1ULL << 0)
#define LUST_BINDING_FLAG_OWNED (1ULL << 1)
#define LUST_BINDING_FLAG_JSON (1ULL << 2)
#define LUST_BINDING_FLAG_BINARY (1ULL << 3)
#define LUST_BINDING_FLAGS_KNOWN_MASK \
    (LUST_BINDING_FLAG_DIRECT | LUST_BINDING_FLAG_OWNED | LUST_BINDING_FLAG_JSON | LUST_BINDING_FLAG_BINARY)

#define LUST_FIELD_OWNED_SLICE (1ULL << 0)
#define LUST_FIELD_NULLABLE (1ULL << 1)
#define LUST_FIELD_OPTIONAL (1ULL << 2)

/* RFC Section 10 descriptors */
typedef struct LustOwnerHandle {
    uint64_t id;
    uint64_t generation;
} LustOwnerHandle;

typedef struct LustDataView {
    uint32_t abi_version;
    uint32_t struct_size;
    uint32_t representation;
    uint32_t flags;
    LustOwnerHandle owner;
    uint64_t node_or_offset;
    uint64_t byte_len;
} LustDataView;

typedef struct LustSchemaDescriptor {
    uint32_t abi_version;
    uint32_t struct_size;
    uint8_t schema_fingerprint[32];
    uint32_t flags;
    uint32_t reserved;
    uint64_t required_scratch_bytes;
    const void *opaque_plan;
} LustSchemaDescriptor;

/* RFC Section 10.2 struct binding descriptors */
typedef struct LustStructBindingInfoV1 {
    uint32_t abi_version;
    uint32_t info_size;
    uint64_t flags;
    uint64_t type_id;
    uint64_t schema_id;
    uint64_t layout_id;
    uint64_t output_size;
    uint64_t output_align;
    uint64_t field_count;
} LustStructBindingInfoV1;

typedef struct LustStructFieldInfoV1 {
    uint32_t info_size;
    uint32_t storage_kind;
    uint64_t slot;
    uint64_t byte_offset;
    uint64_t storage_size;
    uint64_t storage_align;
    uint64_t nested_layout_id;
    uint64_t ownership_flags;
} LustStructFieldInfoV1;

/* Canonical C field primitives matching src/data/layout.rs */
typedef struct LustObjectId {
    uint32_t words[3];
} LustObjectId;

typedef struct LustDecimal128 {
    alignas(16) uint8_t bytes[16];
} LustDecimal128;

typedef struct LustManagedSlice {
    uintptr_t ptr;
    uintptr_t len;
    uint64_t owner_id;
} LustManagedSlice;

/* Runtime LogicalLayoutV1 / ContiguousStruct mirrors */
typedef struct LustCRecord {
    uint8_t flag;
    int64_t id;
    LustObjectId oid;
    LustDecimal128 amount;
    double score;
    LustManagedSlice name;
    LustManagedSlice opt_note;
} LustCRecord;

typedef struct LustCAccount {
    int64_t id;
    uint64_t balance;
    double rate;
    LustManagedSlice owner;
} LustCAccount;

typedef struct LustCPacket {
    uint8_t valid;
    uint8_t opt_flag;
    int64_t ts;
    LustManagedSlice payload;
    LustManagedSlice opt_payload;
} LustCPacket;

typedef struct LustCInner {
    int64_t x;
    double y;
    uint8_t tag;
} LustCInner;

typedef struct LustCOuter {
    uint8_t lead;
    LustCInner inner;
    int64_t tail;
} LustCOuter;

typedef struct LustCUser {
    int64_t id;
    double score;
    uint8_t active;
    LustManagedSlice name;
} LustCUser;

/* Compile-time ABI descriptor assertions */
_Static_assert(sizeof(LustOwnerHandle) == 16, "LustOwnerHandle size");
_Static_assert(alignof(LustOwnerHandle) == 8, "LustOwnerHandle align");
_Static_assert(offsetof(LustOwnerHandle, id) == 0, "LustOwnerHandle.id");
_Static_assert(offsetof(LustOwnerHandle, generation) == 8, "LustOwnerHandle.generation");

_Static_assert(sizeof(LustDataView) == 48, "LustDataView size");
_Static_assert(alignof(LustDataView) == 8, "LustDataView align");
_Static_assert(offsetof(LustDataView, owner) == 16, "LustDataView.owner");
_Static_assert(offsetof(LustDataView, node_or_offset) == 32, "LustDataView.node_or_offset");
_Static_assert(offsetof(LustDataView, byte_len) == 40, "LustDataView.byte_len");

_Static_assert(sizeof(LustStructBindingInfoV1) == 64, "LustStructBindingInfoV1 size");
_Static_assert(alignof(LustStructBindingInfoV1) == 8, "LustStructBindingInfoV1 align");
_Static_assert(offsetof(LustStructBindingInfoV1, flags) == 8, "LustStructBindingInfoV1.flags");
_Static_assert(offsetof(LustStructBindingInfoV1, field_count) == 56, "LustStructBindingInfoV1.field_count");

_Static_assert(sizeof(LustStructFieldInfoV1) == 56, "LustStructFieldInfoV1 size");
_Static_assert(alignof(LustStructFieldInfoV1) == 8, "LustStructFieldInfoV1 align");
_Static_assert(offsetof(LustStructFieldInfoV1, slot) == 8, "LustStructFieldInfoV1.slot");
_Static_assert(offsetof(LustStructFieldInfoV1, byte_offset) == 16, "LustStructFieldInfoV1.byte_offset");

_Static_assert(sizeof(LustObjectId) == 12, "LustObjectId size");
_Static_assert(alignof(LustObjectId) == 4, "LustObjectId align");
_Static_assert(sizeof(LustDecimal128) == 16, "LustDecimal128 size");
_Static_assert(alignof(LustDecimal128) == 16, "LustDecimal128 align");

#if UINTPTR_MAX == 0xffffffffffffffffULL
_Static_assert(sizeof(LustSchemaDescriptor) == 64, "LustSchemaDescriptor size (64-bit)");
_Static_assert(alignof(LustSchemaDescriptor) == 8, "LustSchemaDescriptor align (64-bit)");
_Static_assert(offsetof(LustSchemaDescriptor, opaque_plan) == 56, "LustSchemaDescriptor.opaque_plan");

_Static_assert(sizeof(LustManagedSlice) == 24, "LustManagedSlice size (64-bit)");
_Static_assert(alignof(LustManagedSlice) == 8, "LustManagedSlice align (64-bit)");
_Static_assert(offsetof(LustManagedSlice, ptr) == 0, "LustManagedSlice.ptr");
_Static_assert(offsetof(LustManagedSlice, len) == 8, "LustManagedSlice.len");
_Static_assert(offsetof(LustManagedSlice, owner_id) == 16, "LustManagedSlice.owner_id");

_Static_assert(sizeof(LustCRecord) == 112, "LustCRecord size");
_Static_assert(alignof(LustCRecord) == 16, "LustCRecord align");
_Static_assert(offsetof(LustCRecord, flag) == 0, "LustCRecord.flag");
_Static_assert(offsetof(LustCRecord, id) == 8, "LustCRecord.id");
_Static_assert(offsetof(LustCRecord, oid) == 16, "LustCRecord.oid");
_Static_assert(offsetof(LustCRecord, amount) == 32, "LustCRecord.amount");
_Static_assert(offsetof(LustCRecord, score) == 48, "LustCRecord.score");
_Static_assert(offsetof(LustCRecord, name) == 56, "LustCRecord.name");
_Static_assert(offsetof(LustCRecord, opt_note) == 80, "LustCRecord.opt_note");

_Static_assert(sizeof(LustCAccount) == 48, "LustCAccount size");
_Static_assert(alignof(LustCAccount) == 8, "LustCAccount align");
_Static_assert(offsetof(LustCAccount, id) == 0, "LustCAccount.id");
_Static_assert(offsetof(LustCAccount, balance) == 8, "LustCAccount.balance");
_Static_assert(offsetof(LustCAccount, rate) == 16, "LustCAccount.rate");
_Static_assert(offsetof(LustCAccount, owner) == 24, "LustCAccount.owner");

_Static_assert(sizeof(LustCPacket) == 64, "LustCPacket size");
_Static_assert(alignof(LustCPacket) == 8, "LustCPacket align");
_Static_assert(offsetof(LustCPacket, valid) == 0, "LustCPacket.valid");
_Static_assert(offsetof(LustCPacket, opt_flag) == 1, "LustCPacket.opt_flag");
_Static_assert(offsetof(LustCPacket, ts) == 8, "LustCPacket.ts");
_Static_assert(offsetof(LustCPacket, payload) == 16, "LustCPacket.payload");
_Static_assert(offsetof(LustCPacket, opt_payload) == 40, "LustCPacket.opt_payload");

_Static_assert(sizeof(LustCInner) == 24, "LustCInner size");
_Static_assert(alignof(LustCInner) == 8, "LustCInner align");
_Static_assert(offsetof(LustCInner, x) == 0, "LustCInner.x");
_Static_assert(offsetof(LustCInner, y) == 8, "LustCInner.y");
_Static_assert(offsetof(LustCInner, tag) == 16, "LustCInner.tag");

_Static_assert(sizeof(LustCOuter) == 40, "LustCOuter size");
_Static_assert(alignof(LustCOuter) == 8, "LustCOuter align");
_Static_assert(offsetof(LustCOuter, lead) == 0, "LustCOuter.lead");
_Static_assert(offsetof(LustCOuter, inner) == 8, "LustCOuter.inner");
_Static_assert(offsetof(LustCOuter, tail) == 32, "LustCOuter.tail");

_Static_assert(sizeof(LustCUser) == 48, "LustCUser size");
_Static_assert(alignof(LustCUser) == 8, "LustCUser align");
_Static_assert(offsetof(LustCUser, id) == 0, "LustCUser.id");
_Static_assert(offsetof(LustCUser, score) == 8, "LustCUser.score");
_Static_assert(offsetof(LustCUser, active) == 16, "LustCUser.active");
_Static_assert(offsetof(LustCUser, name) == 24, "LustCUser.name");
#endif

#ifdef __cplusplus
}
#endif

#endif /* LUST_LAYOUT_ABI_V1_H */

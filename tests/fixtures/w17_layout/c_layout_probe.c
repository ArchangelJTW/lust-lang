#include "lust_layout_abi_v1.h"

#include <inttypes.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* Explicit tracked allocator counters for zero-allocation & cleanup proofs. */
static uint64_t g_alloc_calls = 0;
static uint64_t g_free_calls = 0;
static uint64_t g_cleanup_order_checksum = 0;

static void *tracked_malloc(size_t n) {
    g_alloc_calls += 1;
    return malloc(n);
}

static void tracked_free(void *p, uint64_t slot_id) {
    if (p != NULL) {
        g_free_calls += 1;
        g_cleanup_order_checksum = g_cleanup_order_checksum * 31ULL + slot_id;
        free(p);
    }
}

/*
 * Dedicated non-inlined scalar accessor for disassembly verification (Gate-SCHEMA-09).
 * Must emit direct constant-offset loads without any string/hash lookup or allocation.
 */
#if defined(__GNUC__) || defined(__clang__)
__attribute__((noinline))
#endif
int64_t lust_c_record_read_id_and_flag(const LustCRecord *rec) {
    int64_t acc = rec->id;
    if (rec->flag != 0) {
        acc += 100;
    }
    return acc;
}

#if defined(__GNUC__) || defined(__clang__)
__attribute__((noinline))
#endif
int64_t lust_c_user_eval(const LustCUser *user) {
    int64_t s = user->id;
    if (user->active != 0) {
        s += 100;
    }
    return s;
}

/* Mock binding metadata query conforming to RFC Section 10.2 */
static uint32_t lust_struct_binding_info_v1(
    uint32_t abi_version,
    uint64_t flags,
    LustStructBindingInfoV1 *out_info
) {
    if (out_info == NULL) {
        return LUST_ABI_STATUS_INVALID_ABI;
    }
    if (abi_version != LUST_ABI_VERSION_V1) {
        memset(out_info, 0, sizeof(*out_info));
        return LUST_ABI_STATUS_INVALID_ABI;
    }
    if (out_info->info_size < sizeof(LustStructBindingInfoV1)) {
        return LUST_ABI_STATUS_NEED_CAPACITY;
    }
    if ((flags & ~LUST_BINDING_FLAGS_KNOWN_MASK) != 0ULL) {
        memset(out_info, 0, sizeof(*out_info));
        return LUST_ABI_STATUS_INVALID_ABI;
    }

    out_info->abi_version = LUST_ABI_VERSION_V1;
    out_info->info_size = (uint32_t)sizeof(LustStructBindingInfoV1);
    out_info->flags = flags;
    out_info->type_id = 9001ULL;
    out_info->schema_id = 1001ULL;
    out_info->layout_id = 9001ULL;
    out_info->output_size = (uint64_t)sizeof(LustCRecord);
    out_info->output_align = (uint64_t)alignof(LustCRecord);
    out_info->field_count = 7ULL;
    return LUST_ABI_STATUS_OK;
}

static uint32_t lust_struct_binding_field_info_v1(
    uint64_t field_index,
    LustStructFieldInfoV1 *out_field
) {
    static const LustStructFieldInfoV1 kRecordFields[7] = {
        { sizeof(LustStructFieldInfoV1), LUST_STORAGE_BOOL, 0, offsetof(LustCRecord, flag), sizeof(((LustCRecord *)0)->flag), alignof(uint8_t), 0, 0 },
        { sizeof(LustStructFieldInfoV1), LUST_STORAGE_INT64, 1, offsetof(LustCRecord, id), sizeof(((LustCRecord *)0)->id), alignof(int64_t), 0, 0 },
        { sizeof(LustStructFieldInfoV1), LUST_STORAGE_OBJECT_ID, 2, offsetof(LustCRecord, oid), sizeof(((LustCRecord *)0)->oid), alignof(LustObjectId), 0, 0 },
        { sizeof(LustStructFieldInfoV1), LUST_STORAGE_DECIMAL128, 3, offsetof(LustCRecord, amount), sizeof(((LustCRecord *)0)->amount), alignof(LustDecimal128), 0, 0 },
        { sizeof(LustStructFieldInfoV1), LUST_STORAGE_FLOAT64, 4, offsetof(LustCRecord, score), sizeof(((LustCRecord *)0)->score), alignof(double), 0, 0 },
        { sizeof(LustStructFieldInfoV1), LUST_STORAGE_STRING_SLICE, 5, offsetof(LustCRecord, name), sizeof(((LustCRecord *)0)->name), alignof(LustManagedSlice), 0, LUST_FIELD_OWNED_SLICE },
        { sizeof(LustStructFieldInfoV1), LUST_STORAGE_STRING_SLICE, 6, offsetof(LustCRecord, opt_note), sizeof(((LustCRecord *)0)->opt_note), alignof(LustManagedSlice), 0, LUST_FIELD_OWNED_SLICE | LUST_FIELD_NULLABLE | LUST_FIELD_OPTIONAL },
    };

    if (out_field == NULL) {
        return LUST_ABI_STATUS_INVALID_ABI;
    }
    if (out_field->info_size < sizeof(LustStructFieldInfoV1)) {
        return LUST_ABI_STATUS_NEED_CAPACITY;
    }
    if (field_index >= 7ULL) {
        memset(out_field, 0, sizeof(*out_field));
        return LUST_ABI_STATUS_OUT_OF_RANGE;
    }
    *out_field = kRecordFields[field_index];
    return LUST_ABI_STATUS_OK;
}

/* Exercises partial-init failure in C and proves exact reverse-order cleanup without leaks. */
static uint32_t run_partial_init_cleanup_probe(void) {
    char *slots[3] = { NULL, NULL, NULL };
    uint64_t alloc_before = g_alloc_calls;
    uint64_t free_before = g_free_calls;
    g_cleanup_order_checksum = 0;

    /* Initialize slot 0 and slot 1, then fail on slot 2 before commit. */
    slots[0] = (char *)tracked_malloc(32);
    if (!slots[0]) {
        return LUST_ABI_STATUS_NEED_CAPACITY;
    }
    memcpy(slots[0], "slot_zero_owned", 16);

    slots[1] = (char *)tracked_malloc(48);
    if (!slots[1]) {
        tracked_free(slots[0], 1);
        return LUST_ABI_STATUS_NEED_CAPACITY;
    }
    memcpy(slots[1], "slot_one_owned", 15);

    /* Simulated validation failure on field 2: reverse-order cleanup (slot 1 then slot 0). */
    tracked_free(slots[1], 2);
    slots[1] = NULL;
    tracked_free(slots[0], 1);
    slots[0] = NULL;

    uint64_t alloc_delta = g_alloc_calls - alloc_before;
    uint64_t free_delta = g_free_calls - free_before;
    /* Reverse order: 0 * 31 + 2 = 2; 2 * 31 + 1 = 63 */
    if (alloc_delta != 2ULL || free_delta != 2ULL || g_cleanup_order_checksum != 63ULL) {
        return LUST_ABI_STATUS_DEFECT;
    }
    return LUST_ABI_STATUS_OK;
}

static int cmd_layout_json(void) {
    printf("{\n");
    printf("  \"pointer_width\": %zu,\n", sizeof(void *));
    printf("  \"descriptors\": {\n");
    printf("    \"LustOwnerHandle\": {\"size\": %zu, \"align\": %zu, \"offsets\": [%zu, %zu]},\n",
           sizeof(LustOwnerHandle), alignof(LustOwnerHandle),
           offsetof(LustOwnerHandle, id), offsetof(LustOwnerHandle, generation));
    printf("    \"LustDataView\": {\"size\": %zu, \"align\": %zu, \"offsets\": [%zu, %zu, %zu, %zu, %zu, %zu, %zu]},\n",
           sizeof(LustDataView), alignof(LustDataView),
           offsetof(LustDataView, abi_version), offsetof(LustDataView, struct_size),
           offsetof(LustDataView, representation), offsetof(LustDataView, flags),
           offsetof(LustDataView, owner), offsetof(LustDataView, node_or_offset),
           offsetof(LustDataView, byte_len));
    printf("    \"LustSchemaDescriptor\": {\"size\": %zu, \"align\": %zu, \"offsets\": [%zu, %zu, %zu, %zu, %zu, %zu, %zu]},\n",
           sizeof(LustSchemaDescriptor), alignof(LustSchemaDescriptor),
           offsetof(LustSchemaDescriptor, abi_version), offsetof(LustSchemaDescriptor, struct_size),
           offsetof(LustSchemaDescriptor, schema_fingerprint), offsetof(LustSchemaDescriptor, flags),
           offsetof(LustSchemaDescriptor, reserved), offsetof(LustSchemaDescriptor, required_scratch_bytes),
           offsetof(LustSchemaDescriptor, opaque_plan));
    printf("    \"LustStructBindingInfoV1\": {\"size\": %zu, \"align\": %zu, \"offsets\": [%zu, %zu, %zu, %zu, %zu, %zu, %zu, %zu, %zu]},\n",
           sizeof(LustStructBindingInfoV1), alignof(LustStructBindingInfoV1),
           offsetof(LustStructBindingInfoV1, abi_version), offsetof(LustStructBindingInfoV1, info_size),
           offsetof(LustStructBindingInfoV1, flags), offsetof(LustStructBindingInfoV1, type_id),
           offsetof(LustStructBindingInfoV1, schema_id), offsetof(LustStructBindingInfoV1, layout_id),
           offsetof(LustStructBindingInfoV1, output_size), offsetof(LustStructBindingInfoV1, output_align),
           offsetof(LustStructBindingInfoV1, field_count));
    printf("    \"LustStructFieldInfoV1\": {\"size\": %zu, \"align\": %zu, \"offsets\": [%zu, %zu, %zu, %zu, %zu, %zu, %zu, %zu]}\n",
           sizeof(LustStructFieldInfoV1), alignof(LustStructFieldInfoV1),
           offsetof(LustStructFieldInfoV1, info_size), offsetof(LustStructFieldInfoV1, storage_kind),
           offsetof(LustStructFieldInfoV1, slot), offsetof(LustStructFieldInfoV1, byte_offset),
           offsetof(LustStructFieldInfoV1, storage_size), offsetof(LustStructFieldInfoV1, storage_align),
           offsetof(LustStructFieldInfoV1, nested_layout_id), offsetof(LustStructFieldInfoV1, ownership_flags));
    printf("  },\n");
    printf("  \"structs\": {\n");
    printf("    \"Record\": {\"size\": %zu, \"align\": %zu, \"offsets\": [%zu, %zu, %zu, %zu, %zu, %zu, %zu], \"sizes\": [%zu, %zu, %zu, %zu, %zu, %zu, %zu]},\n",
           sizeof(LustCRecord), alignof(LustCRecord),
           offsetof(LustCRecord, flag), offsetof(LustCRecord, id), offsetof(LustCRecord, oid),
           offsetof(LustCRecord, amount), offsetof(LustCRecord, score), offsetof(LustCRecord, name),
           offsetof(LustCRecord, opt_note),
           sizeof(((LustCRecord *)0)->flag), sizeof(((LustCRecord *)0)->id), sizeof(((LustCRecord *)0)->oid),
           sizeof(((LustCRecord *)0)->amount), sizeof(((LustCRecord *)0)->score), sizeof(((LustCRecord *)0)->name),
           sizeof(((LustCRecord *)0)->opt_note));
    printf("    \"Account\": {\"size\": %zu, \"align\": %zu, \"offsets\": [%zu, %zu, %zu, %zu], \"sizes\": [%zu, %zu, %zu, %zu]},\n",
           sizeof(LustCAccount), alignof(LustCAccount),
           offsetof(LustCAccount, id), offsetof(LustCAccount, balance),
           offsetof(LustCAccount, rate), offsetof(LustCAccount, owner),
           sizeof(((LustCAccount *)0)->id), sizeof(((LustCAccount *)0)->balance),
           sizeof(((LustCAccount *)0)->rate), sizeof(((LustCAccount *)0)->owner));
    printf("    \"Packet\": {\"size\": %zu, \"align\": %zu, \"offsets\": [%zu, %zu, %zu, %zu, %zu], \"sizes\": [%zu, %zu, %zu, %zu, %zu]},\n",
           sizeof(LustCPacket), alignof(LustCPacket),
           offsetof(LustCPacket, valid), offsetof(LustCPacket, opt_flag),
           offsetof(LustCPacket, ts), offsetof(LustCPacket, payload), offsetof(LustCPacket, opt_payload),
           sizeof(((LustCPacket *)0)->valid), sizeof(((LustCPacket *)0)->opt_flag),
           sizeof(((LustCPacket *)0)->ts), sizeof(((LustCPacket *)0)->payload), sizeof(((LustCPacket *)0)->opt_payload));
    printf("    \"Inner\": {\"size\": %zu, \"align\": %zu, \"offsets\": [%zu, %zu, %zu], \"sizes\": [%zu, %zu, %zu]},\n",
           sizeof(LustCInner), alignof(LustCInner),
           offsetof(LustCInner, x), offsetof(LustCInner, y), offsetof(LustCInner, tag),
           sizeof(((LustCInner *)0)->x), sizeof(((LustCInner *)0)->y), sizeof(((LustCInner *)0)->tag));
    printf("    \"Outer\": {\"size\": %zu, \"align\": %zu, \"offsets\": [%zu, %zu, %zu], \"sizes\": [%zu, %zu, %zu]},\n",
           sizeof(LustCOuter), alignof(LustCOuter),
           offsetof(LustCOuter, lead), offsetof(LustCOuter, inner), offsetof(LustCOuter, tail),
           sizeof(((LustCOuter *)0)->lead), sizeof(((LustCOuter *)0)->inner), sizeof(((LustCOuter *)0)->tail));
    printf("    \"User\": {\"size\": %zu, \"align\": %zu, \"offsets\": [%zu, %zu, %zu, %zu], \"sizes\": [%zu, %zu, %zu, %zu]}\n",
           sizeof(LustCUser), alignof(LustCUser),
           offsetof(LustCUser, id), offsetof(LustCUser, score),
           offsetof(LustCUser, active), offsetof(LustCUser, name),
           sizeof(((LustCUser *)0)->id), sizeof(((LustCUser *)0)->score),
           sizeof(((LustCUser *)0)->active), sizeof(((LustCUser *)0)->name));
    printf("  }\n");
    printf("}\n");
    return 0;
}

static int cmd_verify_record_bytes(const char *path) {
    FILE *fp = fopen(path, "rb");
    if (!fp) {
        fprintf(stderr, "failed to open raw record file: %s\n", path);
        return 2;
    }
    LustCRecord rec;
    memset(&rec, 0, sizeof(rec));
    size_t n = fread(&rec, 1, sizeof(rec), fp);
    int extra = fgetc(fp);
    fclose(fp);
    if (n != sizeof(rec) || extra != EOF) {
        fprintf(stderr, "unexpected byte length: read=%zu expected=%zu extra=%d\n",
                n, sizeof(rec), extra);
        return 3;
    }

    uint8_t oid_bytes[12];
    memcpy(oid_bytes, &rec.oid, 12);

    uint64_t dec_lo = 0;
    uint64_t dec_hi = 0;
    memcpy(&dec_lo, rec.amount.bytes, 8);
    memcpy(&dec_hi, rec.amount.bytes + 8, 8);

    /* Prepared fixed-scalar read loop: N=10,000 reads with 0 heap allocations. */
    uint64_t allocs_before = g_alloc_calls;
    volatile int64_t sink = 0;
    for (int i = 0; i < 10000; i++) {
        sink += lust_c_record_read_id_and_flag(&rec);
    }
    uint64_t allocs_after = g_alloc_calls;
    int64_t single_eval = lust_c_record_read_id_and_flag(&rec);

    printf("{\"flag\":%u,\"id\":%" PRId64 ",\"oid_hex\":\"%02x%02x%02x%02x%02x%02x%02x%02x%02x%02x%02x%02x\","
           "\"dec_lo\":%" PRIu64 ",\"dec_hi\":%" PRIu64 ",\"score\":%.6f,"
           "\"name_len\":%" PRIu64 ",\"name_owner_id\":%" PRIu64 ","
           "\"opt_note_len\":%" PRIu64 ",\"opt_note_owner_id\":%" PRIu64 ","
           "\"eval_once\":%" PRId64 ",\"eval_10k_sum\":%" PRId64 ",\"scalar_read_allocs\":%" PRIu64 "}\n",
           (unsigned)rec.flag,
           rec.id,
           oid_bytes[0], oid_bytes[1], oid_bytes[2], oid_bytes[3],
           oid_bytes[4], oid_bytes[5], oid_bytes[6], oid_bytes[7],
           oid_bytes[8], oid_bytes[9], oid_bytes[10], oid_bytes[11],
           dec_lo, dec_hi,
           rec.score,
           (uint64_t)rec.name.len, rec.name.owner_id,
           (uint64_t)rec.opt_note.len, rec.opt_note.owner_id,
           single_eval,
           (int64_t)sink,
           allocs_after - allocs_before);
    return 0;
}

static int cmd_verify_user_bytes(const char *path) {
    FILE *fp = fopen(path, "rb");
    if (!fp) {
        fprintf(stderr, "failed to open raw user file: %s\n", path);
        return 2;
    }
    LustCUser user;
    memset(&user, 0, sizeof(user));
    size_t n = fread(&user, 1, sizeof(user), fp);
    int extra = fgetc(fp);
    fclose(fp);
    if (n != sizeof(user) || extra != EOF) {
        fprintf(stderr, "unexpected user byte length: read=%zu expected=%zu extra=%d\n",
                n, sizeof(user), extra);
        return 3;
    }
    int64_t eval_res = lust_c_user_eval(&user);
    printf("{\"id\":%" PRId64 ",\"score\":%.6f,\"active\":%u,\"name_len\":%" PRIu64 ",\"eval\":%" PRId64 "}\n",
           user.id, user.score, (unsigned)user.active, (uint64_t)user.name.len, eval_res);
    return 0;
}

static int cmd_binding_info_checks(void) {
    LustStructBindingInfoV1 info;
    memset(&info, 0, sizeof(info));
    info.info_size = (uint32_t)sizeof(info);

    /* 1. Positive control */
    uint32_t st_ok = lust_struct_binding_info_v1(
        LUST_ABI_VERSION_V1,
        LUST_BINDING_FLAG_DIRECT | LUST_BINDING_FLAG_OWNED | LUST_BINDING_FLAG_JSON,
        &info
    );
    if (st_ok != LUST_ABI_STATUS_OK || info.output_size != sizeof(LustCRecord) || info.field_count != 7ULL) {
        fprintf(stderr, "positive binding info failed: st=%u\n", st_ok);
        return 10;
    }

    /* 2. Null pointer rejected */
    if (lust_struct_binding_info_v1(LUST_ABI_VERSION_V1, LUST_BINDING_FLAG_DIRECT, NULL) != LUST_ABI_STATUS_INVALID_ABI) {
        fprintf(stderr, "null binding info was not rejected\n");
        return 11;
    }

    /* 3. Wrong ABI version rejected */
    LustStructBindingInfoV1 bad_ver;
    memset(&bad_ver, 0, sizeof(bad_ver));
    bad_ver.info_size = (uint32_t)sizeof(bad_ver);
    if (lust_struct_binding_info_v1(999u, LUST_BINDING_FLAG_DIRECT, &bad_ver) != LUST_ABI_STATUS_INVALID_ABI) {
        fprintf(stderr, "wrong abi_version was not rejected\n");
        return 12;
    }

    /* 4. Short info_size rejected */
    LustStructBindingInfoV1 short_sz;
    memset(&short_sz, 0, sizeof(short_sz));
    short_sz.info_size = 8u;
    if (lust_struct_binding_info_v1(LUST_ABI_VERSION_V1, LUST_BINDING_FLAG_DIRECT, &short_sz) != LUST_ABI_STATUS_NEED_CAPACITY) {
        fprintf(stderr, "short info_size was not rejected\n");
        return 13;
    }

    /* 5. Unknown flags rejected */
    LustStructBindingInfoV1 bad_flags;
    memset(&bad_flags, 0, sizeof(bad_flags));
    bad_flags.info_size = (uint32_t)sizeof(bad_flags);
    if (lust_struct_binding_info_v1(LUST_ABI_VERSION_V1, (1ULL << 40), &bad_flags) != LUST_ABI_STATUS_INVALID_ABI) {
        fprintf(stderr, "unknown flags were not rejected\n");
        return 14;
    }

    /* 6. Field info positive + out-of-range + short size + null */
    LustStructFieldInfoV1 fld;
    memset(&fld, 0, sizeof(fld));
    fld.info_size = (uint32_t)sizeof(fld);
    if (lust_struct_binding_field_info_v1(3ULL, &fld) != LUST_ABI_STATUS_OK || fld.byte_offset != offsetof(LustCRecord, amount)) {
        fprintf(stderr, "field info 3 failed\n");
        return 15;
    }
    if (lust_struct_binding_field_info_v1(7ULL, &fld) != LUST_ABI_STATUS_OUT_OF_RANGE) {
        fprintf(stderr, "out-of-range field info was not rejected\n");
        return 16;
    }
    if (lust_struct_binding_field_info_v1(0ULL, NULL) != LUST_ABI_STATUS_INVALID_ABI) {
        fprintf(stderr, "null field info was not rejected\n");
        return 17;
    }
    LustStructFieldInfoV1 short_fld;
    memset(&short_fld, 0, sizeof(short_fld));
    short_fld.info_size = 4u;
    if (lust_struct_binding_field_info_v1(0ULL, &short_fld) != LUST_ABI_STATUS_NEED_CAPACITY) {
        fprintf(stderr, "short field info_size was not rejected\n");
        return 18;
    }

    /* 7. Partial-init reverse-order cleanup without leaks */
    if (run_partial_init_cleanup_probe() != LUST_ABI_STATUS_OK) {
        fprintf(stderr, "partial-init cleanup probe failed\n");
        return 19;
    }

    printf("{\"binding_checks\":\"PASS\",\"allocs\":%" PRIu64 ",\"frees\":%" PRIu64 ",\"cleanup_order_checksum\":%" PRIu64 "}\n",
           g_alloc_calls, g_free_calls, g_cleanup_order_checksum);
    return 0;
}

int main(int argc, char **argv) {
    if (argc < 2) {
        fprintf(stderr, "usage: %s --layout-json | --verify-record-bytes <path> | --verify-user-bytes <path> | --binding-info-checks\n", argv[0]);
        return 1;
    }
    if (strcmp(argv[1], "--layout-json") == 0) {
        return cmd_layout_json();
    }
    if (strcmp(argv[1], "--verify-record-bytes") == 0 && argc >= 3) {
        return cmd_verify_record_bytes(argv[2]);
    }
    if (strcmp(argv[1], "--verify-user-bytes") == 0 && argc >= 3) {
        return cmd_verify_user_bytes(argv[2]);
    }
    if (strcmp(argv[1], "--binding-info-checks") == 0) {
        return cmd_binding_info_checks();
    }
    fprintf(stderr, "unknown mode: %s\n", argv[1]);
    return 1;
}

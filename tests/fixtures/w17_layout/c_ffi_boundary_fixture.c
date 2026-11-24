#include "lust_layout_abi_v1.h"

#include <errno.h>
#include <inttypes.h>
#include <signal.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

/* Declared store statuses (matching RFC Section 16.4 manifest) */
#define STORE_OK 0
#define STORE_MISSING 10
#define STORE_ERROR 11
#define STORE_EINTR 12
#define STORE_CANCELLED 13
#define STORE_PARTIAL_FAIL 14
#define STORE_UNKNOWN_SAMPLE 777

static uint64_t g_constructor_runs = 0;
static uint64_t g_native_entries = 0;
static uint64_t g_owner_allocs = 0;
static uint64_t g_owner_frees = 0;
static uint64_t g_cleanup_order_checksum = 0;
static volatile sig_atomic_t g_cancel_signal_seen = 0;

/* Real C constructor proving library/process initialization occurs iff permitted. */
#if defined(__GNUC__) || defined(__clang__)
__attribute__((constructor))
#endif
static void lust_w17_fixture_constructor(void) {
    g_constructor_runs += 1;
    const char *log_path = getenv("LUST_W17_CTOR_LOG");
    if (log_path != NULL && log_path[0] != '\0') {
        FILE *fp = fopen(log_path, "a");
        if (fp != NULL) {
            fprintf(fp, "CONSTRUCTOR_RUN=%" PRIu64 "\n", g_constructor_runs);
            fclose(fp);
        }
    }
}

static void on_term_signal(int sig) {
    (void)sig;
    g_cancel_signal_seen = 1;
}

typedef struct StoreRecord {
    int64_t value;
    uint64_t owner_token;
} StoreRecord;

static StoreRecord *store_record_alloc(int64_t val, uint64_t token) {
    StoreRecord *rec = (StoreRecord *)malloc(sizeof(StoreRecord));
    if (rec != NULL) {
        g_owner_allocs += 1;
        rec->value = val;
        rec->owner_token = token;
    }
    return rec;
}

static void store_record_free(StoreRecord *rec) {
    if (rec != NULL) {
        g_owner_frees += 1;
        g_cleanup_order_checksum = g_cleanup_order_checksum * 31ULL + rec->owner_token;
        free(rec);
    }
}

typedef struct CallbackContext {
    uint64_t registered;
    uint64_t invocations;
    uint64_t post_unregister_invocations;
    int64_t sum;
} CallbackContext;

typedef int (*LustCallbackFn)(CallbackContext *ctx, int64_t delta);

static int sample_callback(CallbackContext *ctx, int64_t delta) {
    if (!ctx->registered) {
        ctx->post_unregister_invocations += 1;
        return -1;
    }
    ctx->invocations += 1;
    ctx->sum += delta;
    return 0;
}

static void emit_result_json(
    const char *mode,
    int32_t raw_status,
    int has_payload,
    int64_t payload,
    int out_ptr_non_null,
    int captured_errno,
    int request_cancelled,
    uint64_t callbacks_ran,
    uint64_t post_unreg_callbacks
) {
    printf("{\"mode\":\"%s\",\"raw_status\":%" PRId32 ",\"has_payload\":%s,\"payload\":%" PRId64
           ",\"out_ptr_non_null\":%s,\"captured_errno\":%d,\"request_cancelled\":%s"
           ",\"constructor_runs\":%" PRIu64 ",\"native_entries\":%" PRIu64
           ",\"owner_allocs\":%" PRIu64 ",\"owner_frees\":%" PRIu64
           ",\"cleanup_order_checksum\":%" PRIu64
           ",\"callbacks_ran\":%" PRIu64 ",\"post_unreg_callbacks\":%" PRIu64 "}\n",
           mode,
           raw_status,
           has_payload ? "true" : "false",
           payload,
           out_ptr_non_null ? "true" : "false",
           captured_errno,
           request_cancelled ? "true" : "false",
           g_constructor_runs,
           g_native_entries,
           g_owner_allocs,
           g_owner_frees,
           g_cleanup_order_checksum,
           callbacks_ran,
           post_unreg_callbacks);
    fflush(stdout);
}

static int32_t native_store_read(
    const char *mode,
    int request_cancelled,
    StoreRecord **out_rec,
    int *out_immediate_errno
) {
    g_native_entries += 1;
    *out_rec = NULL;
    *out_immediate_errno = 0;

    if (strcmp(mode, "success_positive") == 0) {
        *out_rec = store_record_alloc(42001LL, 1ULL);
        return STORE_OK;
    }
    if (strcmp(mode, "success_negative") == 0) {
        *out_rec = store_record_alloc(-42LL, 2ULL);
        return STORE_OK;
    }
    if (strcmp(mode, "success_i64_min") == 0) {
        *out_rec = store_record_alloc(INT64_MIN, 3ULL);
        return STORE_OK;
    }
    if (strcmp(mode, "status_missing") == 0) {
        return STORE_MISSING;
    }
    if (strcmp(mode, "status_io_errno") == 0) {
        /* Set EACCES as the native syscall failure errno, capture immediately,
         * then clobber errno during simulated cleanup to prove immediate capture. */
        errno = EACCES;
        *out_immediate_errno = errno;
        errno = ENOENT;
        return STORE_ERROR;
    }
    if (strcmp(mode, "stale_errno_on_success") == 0) {
        errno = EPERM;
        *out_rec = store_record_alloc(77LL, 4ULL);
        return STORE_OK;
    }
    if (strcmp(mode, "stale_errno_on_missing") == 0) {
        errno = EPERM;
        return STORE_MISSING;
    }
    if (strcmp(mode, "status_eintr_uncancelled") == 0) {
        errno = EINTR;
        *out_immediate_errno = errno;
        return STORE_EINTR;
    }
    if (strcmp(mode, "status_cancelled_with_token") == 0) {
        (void)request_cancelled;
        return STORE_CANCELLED;
    }
    if (strcmp(mode, "status_cancelled_without_token") == 0) {
        return STORE_CANCELLED;
    }
    if (strcmp(mode, "invalid_success_null") == 0) {
        /* Violates success invariant: returns STORE_OK with NULL output pointer. */
        *out_rec = NULL;
        return STORE_OK;
    }
    if (strcmp(mode, "unknown_status") == 0) {
        return STORE_UNKNOWN_SAMPLE;
    }
    if (strcmp(mode, "partial_init_cleanup") == 0) {
        StoreRecord *r1 = store_record_alloc(100LL, 1ULL);
        StoreRecord *r2 = store_record_alloc(200LL, 2ULL);
        /* Step 3 fails: free in reverse order (r2 then r1) before returning error. */
        store_record_free(r2);
        store_record_free(r1);
        *out_rec = NULL;
        return STORE_PARTIAL_FAIL;
    }
    if (strcmp(mode, "mutant_leaked_owner") == 0) {
        /* Intentional mutant: allocates 2 owned handles, frees only 1 in ledger,
         * and releases raw memory only after recording the leak counter so ASan
         * doesn't mask the deterministic mutant counter assertion. */
        StoreRecord *r1 = store_record_alloc(100LL, 1ULL);
        StoreRecord *r2 = store_record_alloc(200LL, 2ULL);
        store_record_free(r2);
        /* Omit store_record_free(r1) from owner ledger; raw-free r1 silently. */
        free(r1);
        *out_rec = NULL;
        return STORE_PARTIAL_FAIL;
    }
    return STORE_UNKNOWN_SAMPLE;
}

int main(int argc, char **argv) {
    if (argc < 2) {
        fprintf(stderr, "usage: %s <mode> [cancel=0|1]\n", argv[0]);
        return 1;
    }
    const char *mode = argv[1];
    int request_cancelled = (argc >= 3 && strcmp(argv[2], "1") == 0) ? 1 : 0;

    if (strcmp(mode, "cancel_race") == 0) {
        g_native_entries += 1;
        signal(SIGTERM, on_term_signal);
        StoreRecord *r1 = store_record_alloc(11LL, 1ULL);
        StoreRecord *r2 = store_record_alloc(22LL, 2ULL);
        /* Cooperative cancellation race: spin briefly until cancel flag/signal arrives. */
        for (int i = 0; i < 40; i++) {
            if (request_cancelled || g_cancel_signal_seen) {
                break;
            }
            usleep(1000);
        }
        /* Release partial allocations in reverse order before returning STORE_CANCELLED. */
        store_record_free(r2);
        store_record_free(r1);
        emit_result_json(mode, STORE_CANCELLED, 0, 0, 0, 0, 1, 0, 0);
        return 0;
    }

    if (strcmp(mode, "callback_bounded") == 0) {
        g_native_entries += 1;
        CallbackContext *ctx = (CallbackContext *)calloc(1, sizeof(CallbackContext));
        if (!ctx) {
            return 2;
        }
        g_owner_allocs += 1;
        ctx->registered = 1;
        LustCallbackFn cb = sample_callback;
        int rc = 0;
        rc |= cb(ctx, 10);
        rc |= cb(ctx, 20);
        rc |= cb(ctx, 12);
        /* Explicit unregister acknowledgement before releasing context. */
        ctx->registered = 0;
        int64_t final_sum = ctx->sum;
        uint64_t ran = ctx->invocations;
        uint64_t post_unreg = ctx->post_unregister_invocations;
        free(ctx);
        g_owner_frees += 1;
        emit_result_json(mode, rc == 0 ? STORE_OK : STORE_ERROR, 1, final_sum, 1, 0, 0, ran, post_unreg);
        return 0;
    }

    if (strcmp(mode, "unbounded_worker") == 0) {
        g_native_entries += 1;
        /* Signal readiness to parent via stdout line, then block until terminated. */
        printf("WORKER_READY pid=%ld\n", (long)getpid());
        fflush(stdout);
        for (;;) {
            usleep(50000);
        }
        return 0;
    }

    if (strcmp(mode, "runaway_watchdog_probe") == 0) {
        g_native_entries += 1;
        /* Ignore SIGTERM so only external SIGKILL watchdog terminates this child. */
        signal(SIGTERM, SIG_IGN);
        printf("RUNAWAY_READY pid=%ld\n", (long)getpid());
        fflush(stdout);
        for (;;) {
            usleep(50000);
        }
        return 0;
    }

    StoreRecord *out_rec = NULL;
    int immediate_errno = 0;
    int32_t st = native_store_read(mode, request_cancelled, &out_rec, &immediate_errno);
    int has_payload = (out_rec != NULL) ? 1 : 0;
    int64_t payload = (out_rec != NULL) ? out_rec->value : 0;
    if (out_rec != NULL) {
        store_record_free(out_rec);
        out_rec = NULL;
    }
    emit_result_json(
        mode,
        st,
        has_payload,
        payload,
        has_payload,
        immediate_errno,
        request_cancelled,
        0,
        0
    );
    return 0;
}

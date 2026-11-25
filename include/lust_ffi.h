#ifndef LUST_FFI_H
#define LUST_FFI_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Extended FFI value tag */
typedef enum LustFfiValueTag {
    LUST_FFI_VALUE_NIL = 0,
    LUST_FFI_VALUE_BOOL = 1,
    LUST_FFI_VALUE_INT = 2,
    LUST_FFI_VALUE_FLOAT = 3,
    LUST_FFI_VALUE_STRING = 4,
    LUST_FFI_VALUE_HANDLE = 5, /* Extended: Opaque handle */
} LustFfiValueTag;

typedef struct LustHandle LustHandle;

typedef struct LustFfiValue {
    LustFfiValueTag tag;
    bool bool_value;
    int64_t int_value;
    double float_value;
    char *string_ptr;
    LustHandle *handle_ptr; /* Extended: Handle payload */
} LustFfiValue;

/* Disambiguated C FFI status */
typedef enum LustFfiResult {
    LUST_FFI_OK = 0,
    LUST_FFI_ERR_INVALID_HANDLE = 1,
    LUST_FFI_ERR_STALE_HANDLE = 2,
    LUST_FFI_ERR_TYPE_MISMATCH = 3,
    LUST_FFI_ERR_INDEX_OUT_OF_BOUNDS = 4,
    LUST_FFI_ERR_OUT_OF_MEMORY = 5,
    LUST_FFI_ERR_PROGRAM_MISMATCH = 6,
} LustFfiResult;

/* Handle Kind with LUST_HANDLE_INVALID sentinel */
typedef enum LustFfiHandleKind {
    LUST_HANDLE_INVALID = 0,
    LUST_HANDLE_ARRAY = 1,
    LUST_HANDLE_MAP = 2,
    LUST_HANDLE_STRUCT = 3,
} LustFfiHandleKind;

typedef struct EmbeddedBuilder EmbeddedBuilder;
typedef struct EmbeddedProgram EmbeddedProgram;

void lust_clear_last_error(void);
const char *lust_last_error_message(void);

void lust_string_free(char *ptr);
void lust_value_dispose(LustFfiValue *value);

EmbeddedBuilder *lust_builder_new(void);
void lust_builder_free(EmbeddedBuilder *builder);

bool lust_builder_add_module(EmbeddedBuilder *builder, const char *module_path, const char *source);
bool lust_builder_set_entry_module(EmbeddedBuilder *builder, const char *module_path);
bool lust_builder_set_base_dir(EmbeddedBuilder *builder, const char *base_dir);

EmbeddedProgram *lust_builder_compile(EmbeddedBuilder *builder);
void lust_program_free(EmbeddedProgram *program);

bool lust_program_run_entry(EmbeddedProgram *program);
bool lust_program_call(
    EmbeddedProgram *program,
    const char *function_name,
    const LustFfiValue *args,
    size_t args_len,
    LustFfiValue *out_value
);
bool lust_program_get_global(EmbeddedProgram *program, const char *name, LustFfiValue *out_value);
bool lust_program_set_global(EmbeddedProgram *program, const char *name, const LustFfiValue *value);

/* Handle lifecycle */
LustFfiResult lust_handle_retain(LustHandle *handle);
LustFfiResult lust_handle_release(LustHandle *handle);
LustFfiHandleKind lust_handle_kind(const LustHandle *handle);
bool lust_handle_is_valid(const LustHandle *handle);

/* Array handle operations */
LustFfiResult lust_array_new(EmbeddedProgram *program, size_t initial_capacity, LustHandle **out_handle);
LustFfiResult lust_array_len(const LustHandle *handle, size_t *out_len);
LustFfiResult lust_array_get(const LustHandle *handle, size_t index, LustFfiValue *out_value);
LustFfiResult lust_array_set(LustHandle *handle, size_t index, const LustFfiValue *value);
LustFfiResult lust_array_push(LustHandle *handle, const LustFfiValue *value);
LustFfiResult lust_array_pop(LustHandle *handle, LustFfiValue *out_value);

/* Map handle operations */
LustFfiResult lust_map_new(EmbeddedProgram *program, LustHandle **out_handle);
LustFfiResult lust_map_len(const LustHandle *handle, size_t *out_len);
LustFfiResult lust_map_get(const LustHandle *handle, const LustFfiValue *key, LustFfiValue *out_value);
LustFfiResult lust_map_set(LustHandle *handle, const LustFfiValue *key, const LustFfiValue *value);
LustFfiResult lust_map_remove(LustHandle *handle, const LustFfiValue *key, LustFfiValue *out_value);

/* Struct handle operations */
LustFfiResult lust_struct_new(EmbeddedProgram *program, const char *type_name, LustHandle **out_handle);
LustFfiResult lust_struct_type_name(const LustHandle *handle, const char **out_name);
LustFfiResult lust_struct_get_field(const LustHandle *handle, const char *field_name, LustFfiValue *out_value);
LustFfiResult lust_struct_set_field(LustHandle *handle, const char *field_name, const LustFfiValue *value);

/* Callback operations */
typedef LustFfiResult (*LustFfiCallbackFn)(
    void *user_data,
    const LustFfiValue *args,
    size_t args_len,
    LustFfiValue *out_result
);

bool lust_program_register_callback(
    EmbeddedProgram *program,
    const char *name,
    LustFfiCallbackFn callback,
    void *user_data,
    void (*cleanup)(void *user_data)
);

#ifdef __cplusplus
}
#endif

#endif /* LUST_FFI_H */

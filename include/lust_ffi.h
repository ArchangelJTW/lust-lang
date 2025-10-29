#ifndef LUST_FFI_H
#define LUST_FFI_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef enum LustFfiValueTag {
    LUST_FFI_VALUE_NIL = 0,
    LUST_FFI_VALUE_BOOL = 1,
    LUST_FFI_VALUE_INT = 2,
    LUST_FFI_VALUE_FLOAT = 3,
    LUST_FFI_VALUE_STRING = 4,
} LustFfiValueTag;

typedef struct LustFfiValue {
    LustFfiValueTag tag;
    bool bool_value;
    int64_t int_value;
    double float_value;
    char *string_ptr;
} LustFfiValue;

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

#ifdef __cplusplus
}
#endif

#endif /* LUST_FFI_H */

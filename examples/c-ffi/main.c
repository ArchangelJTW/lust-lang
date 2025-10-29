#include "lust_ffi.h"

#include <stdio.h>
#include <stdlib.h>

static void die_with_last_error(const char *context) {
    const char *message = lust_last_error_message();
    if (message == NULL) {
        message = "(no error message)";
    }
    fprintf(stderr, "%s failed: %s\n", context, message);
}

int main(void) {
    const char *module_path = "main";
    const char *module_source =
        "pub function add(a: int, b: int): int\n"
        "    return a + b\n"
        "end\n";

    EmbeddedBuilder *builder = lust_builder_new();
    if (builder == NULL) {
        die_with_last_error("lust_builder_new");
        return EXIT_FAILURE;
    }

    if (!lust_builder_set_base_dir(builder, "__ffi_example__")) {
        die_with_last_error("lust_builder_set_base_dir");
        lust_builder_free(builder);
        return EXIT_FAILURE;
    }

    if (!lust_builder_add_module(builder, module_path, module_source)) {
        die_with_last_error("lust_builder_add_module");
        lust_builder_free(builder);
        return EXIT_FAILURE;
    }

    if (!lust_builder_set_entry_module(builder, module_path)) {
        die_with_last_error("lust_builder_set_entry_module");
        lust_builder_free(builder);
        return EXIT_FAILURE;
    }

    EmbeddedProgram *program = lust_builder_compile(builder);
    if (program == NULL) {
        die_with_last_error("lust_builder_compile");
        return EXIT_FAILURE;
    }

    LustFfiValue args[2] = {
        {.tag = LUST_FFI_VALUE_INT, .int_value = 20},
        {.tag = LUST_FFI_VALUE_INT, .int_value = 22},
    };

    LustFfiValue result = {0};
    if (!lust_program_call(program, "main.add", args, 2, &result)) {
        die_with_last_error("lust_program_call");
        lust_program_free(program);
        return EXIT_FAILURE;
    }

    if (result.tag != LUST_FFI_VALUE_INT) {
        fprintf(stderr, "Unexpected result tag %d\n", result.tag);
        lust_value_dispose(&result);
        lust_program_free(program);
        return EXIT_FAILURE;
    }

    printf("20 + 22 = %lld\n", (long long)result.int_value);

    if (!lust_program_set_global(program, "main.answer", &result)) {
        die_with_last_error("lust_program_set_global");
        lust_value_dispose(&result);
        lust_program_free(program);
        return EXIT_FAILURE;
    }

    LustFfiValue stored = {0};
    if (!lust_program_get_global(program, "main.answer", &stored)) {
        die_with_last_error("lust_program_get_global");
        lust_value_dispose(&result);
        lust_program_free(program);
        return EXIT_FAILURE;
    }

    if (stored.tag == LUST_FFI_VALUE_INT) {
        printf("stored main.answer = %lld\n", (long long)stored.int_value);
    }

    lust_value_dispose(&stored);
    lust_value_dispose(&result);
    lust_program_free(program);
    return EXIT_SUCCESS;
}

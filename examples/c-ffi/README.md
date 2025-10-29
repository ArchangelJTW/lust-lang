# C FFI Example

Small C program that embeds Lust through the exported C ABI.

## Prerequisites

Build the Lust shared library first:

```
cargo build --release --lib
```

## Building the example

```
cc main.c -I../../include -L../../target/release -llust -o lust_ffi_example
```

Depending on your platform you may also need to add an rpath so the loader can find
`liblust` at runtime:

- Linux: `cc ... -Wl,-rpath,'$ORIGIN/../../target/release'`
- macOS: `cc ... -Wl,-rpath,@loader_path/../../target/release`
- Windows (MSVC): link against `lust.lib` and ensure `lust.dll` is alongside the executable.

## Running

```
./lust_ffi_example
```

Expected output:

```
20 + 22 = 42
stored main.answer = 42
```

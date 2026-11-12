UNAME_S := $(shell uname -s)
ifeq ($(UNAME_S),Darwin)
    LIBLUST := ../../target/release/liblust.dylib
    LDFLAGS += -Wl,-rpath,@loader_path/../../target/release
else
    LIBLUST := ../../target/release/liblust.so
    LDFLAGS += -Wl,-rpath,'$$ORIGIN/../../target/release'
endif

all: lust_ffi_example

$(LIBLUST):
	cargo build --release --lib

lust_ffi_example: main.c $(LIBLUST)
	$(CC) main.c $(CFLAGS) $(LDFLAGS) -I../../include -L../../target/release -llust -o lust_ffi_example

test: lust_ffi_example
	./lust_ffi_example

clean:
	rm -f lust_ffi_example *.o

.PHONY: all test clean

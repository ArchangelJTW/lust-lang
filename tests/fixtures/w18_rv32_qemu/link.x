ENTRY(_start)

MEMORY
{
  RAM : ORIGIN = 0x80000000, LENGTH = 16M
}

_stack_start = ORIGIN(RAM) + LENGTH(RAM);

SECTIONS
{
  .text ORIGIN(RAM) :
  {
    KEEP(*(.text.start));
    *(.text .text.*);
  } > RAM

  .rodata : ALIGN(4)
  {
    *(.rodata .rodata.*);
    *(.srodata .srodata.*);
    . = ALIGN(4);
  } > RAM

  .data : ALIGN(4)
  {
    *(.data .data.*);
    *(.sdata .sdata.*);
    . = ALIGN(4);
  } > RAM

  .bss (NOLOAD) : ALIGN(4)
  {
    __sbss = .;
    *(.bss .bss.*);
    *(.sbss .sbss.*);
    *(COMMON);
    . = ALIGN(4);
    __ebss = .;
  } > RAM

  /DISCARD/ :
  {
    *(.eh_frame .eh_frame.*);
  }
}

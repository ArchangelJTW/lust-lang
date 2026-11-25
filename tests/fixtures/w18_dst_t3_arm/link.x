ENTRY(Reset)

_stack_start = ORIGIN(RAM) + LENGTH(RAM);

SECTIONS
{
  .vector_table ORIGIN(FLASH) :
  {
    LONG(_stack_start);
    KEEP(*(.vector_table.reset_vector));
    KEEP(*(.vector_table.exceptions));
  } > FLASH

  .text :
  {
    *(.text .text.*);
  } > FLASH

  .rodata : ALIGN(4)
  {
    *(.rodata .rodata.*);
    . = ALIGN(4);
  } > FLASH

  .data : ALIGN(4)
  {
    __sdata = .;
    *(.data .data.*);
    . = ALIGN(4);
    __edata = .;
  } > RAM AT > FLASH

  __sidata = LOADADDR(.data);

  .bss (NOLOAD) : ALIGN(4)
  {
    __sbss = .;
    *(.bss .bss.*);
    *(COMMON);
    . = ALIGN(4);
    __ebss = .;
  } > RAM

  /DISCARD/ :
  {
    *(.ARM.exidx .ARM.exidx.*);
    *(.ARM.extab .ARM.extab.*);
  }
}

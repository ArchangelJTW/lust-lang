//! Example: Standalone VM and EmbeddedProgram Execution.
//! Demonstrates embedding Lust into a host application, compiling and executing
//! Lust script modules, invoking typed functions, and reading runtime state.

use lust::EmbeddedProgram;

fn main() -> lust::Result<()> {
    println!("--- [Standalone VM / EmbeddedProgram Example] ---");

    // 1. Lust module source with helper functions and global script initialization
    let script = r#"
        function compute_score(base: int, multiplier: int): int
            return base * multiplier + 10
        end

        function greet(name: string): string
            return "Hello from Lust, " .. name .. "!"
        end

        local initial_result = compute_score(20, 2)
        println("Script initialized: initial score = " .. tostring(initial_result))
    "#;

    // 2. Build and compile an embedded Lust program
    let mut program = EmbeddedProgram::builder()
        .module("main", script)
        .entry_module("main")
        .compile()?;

    // 3. Run the top-level entry script
    program.run_entry_script()?;
    println!("✓ Successfully executed entry script in standalone VM");

    // 4. Call a typed Lust function from Rust host
    let score: i64 = program.call_typed("main.compute_score", (50_i64, 2_i64))?;
    println!("✓ Called 'main.compute_score(50, 2)' -> {}", score);
    assert_eq!(score, 110, "50 * 2 + 10 must equal 110");

    // 5. Call string manipulation function
    let greeting: String = program.call_typed("main.greet", "Rust Host".to_string())?;
    println!("✓ Called 'main.greet(\"Rust Host\")' -> '{}'", greeting);
    assert_eq!(greeting, "Hello from Lust, Rust Host!");

    println!("All standalone VM embedding assertions passed!");
    Ok(())
}

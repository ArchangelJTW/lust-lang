use lust::{struct_field, EmbeddedProgram, StructInstance};

fn main() -> lust::Result<()> {
    let module = r#"
        struct Point
            x: int,
            y: int,
            name: string
        end

        enum Status
            Pending,
            Complete(int)
        end

        SCALE_FACTOR: int = 3

        extern {
            function host_scale(int): int
        }

        pub function translate(point: Point, dx: int, dy: int): Point
            println("Point Name: " .. point.name)
            return Point { x = point.x + dx, y = point.y + dy, name = point.name }
        end

        pub function summarize(values: Array<int>): int
            local total: int = 0
            for value in values do
                total = total + value
            end
            return total
        end

        pub function describe(status: Status): string
            if status is Pending then
                return "pending"
            elseif status is Complete(value) then
                return "done(" .. tostring(value) .. ")"
            end
            return "unknown"
        end

        pub function amplify(value: int): int
            return host_scale(value) * SCALE_FACTOR
        end

        pub function bump_scale(): ()
            SCALE_FACTOR = SCALE_FACTOR + 1
        end
    "#;

    let mut program = EmbeddedProgram::builder()
        .module("main", module)
        .entry_module("main")
        .compile()?;

    program.run_entry_script()?;

    program.register_typed_native(
        "host_scale",
        |value: i64| -> std::result::Result<i64, String> { Ok(value * 10) },
    )?;

    let initial_scale = program
        .get_typed_global::<i64>("main.SCALE_FACTOR")?
        .expect("SCALE_FACTOR should exist");
    println!("Initial SCALE_FACTOR = {initial_scale}");

    program.set_global_value("main.SCALE_FACTOR", 5_i64);

    let rust_set_scale = program
        .get_typed_global::<i64>("main.SCALE_FACTOR")?
        .expect("SCALE_FACTOR should exist");
    println!("SCALE_FACTOR after Rust update = {rust_set_scale}");

    let pending = program.enum_variant("main.Status", "Pending")?;

    println!(
        "Status: {}",
        program.call_typed::<_, String>("main.describe", pending)?
    );

    let complete = program.enum_variant_with("main.Status", "Complete", vec![4_i64])?;
    println!(
        "Status: {}",
        program.call_typed::<_, String>("main.describe", complete)?
    );

    let point = program.struct_instance(
        "main.Point",
        [
            struct_field("x", 3_i64),
            struct_field("y", 4_i64),
            struct_field("name", "FirstPoint"),
        ],
    )?;

    point.set_field("x", 8_i64)?;

    point.update_field("y", |value| match value {
        lust::Value::Int(current) => Ok(current + 5),
        other => Err(lust::LustError::RuntimeError {
            message: format!("expected int but saw {other:?}"),
        }),
    })?;

    let moved: StructInstance = program.call_typed("main.translate", (point, 2_i64, 5_i64))?;
    println!(
        "Translated point -> ({}, {})",
        moved.field::<i64>("x")?,
        moved.field::<i64>("y")?
    );

    let total: i64 = program.call_typed("main.summarize", vec![1_i64, 2_i64, 3_i64])?;
    println!("Summarize([1,2,3]) = {total}");

    let amplified: i64 = program.call_typed("main.amplify", 7_i64)?;
    println!("Amplify(7) with SCALE_FACTOR = {rust_set_scale} -> {amplified}");

    program.call_typed::<_, ()>("main.bump_scale", ())?;
    let bumped_scale = program
        .get_typed_global::<i64>("main.SCALE_FACTOR")?
        .expect("SCALE_FACTOR should exist");

    let amplified_after_bump: i64 = program.call_typed("main.amplify", 7_i64)?;
    println!("Amplify(7) after bump (SCALE_FACTOR = {bumped_scale}) -> {amplified_after_bump}");
    Ok(())
}

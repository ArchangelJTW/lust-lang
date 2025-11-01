use lust::{
    struct_field, ArrayHandle, EmbeddedProgram, FromLustValue, MapHandle, StructInstance,
    TableHandle, Value,
};

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

        arr_global: Array<int> = [1, 2, 3]

        map_global: Map<string, int> = {["one"] = 1, ["two"] = 2}

        table_global: Table = {}
        table_global:set("one", 1)
        table_global:set(2, "two")

        lust_point: Point = Point {
            x = 1,
            y = 2,
            name = "Lust Point"
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

    {
        let name_ref = point.borrow_field("name")?;
        println!(
            "Point Name (borrowed): {}",
            name_ref.as_string().unwrap_or("<unnamed>")
        );
    }

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

    if let Some(array) = program.get_typed_global::<ArrayHandle>("main.arr_global")? {
        array.push(Value::Int(4));
        let snapshot = array.with_ref(|values| {
            values
                .iter()
                .map(|v| v.as_int().unwrap())
                .collect::<Vec<_>>()
        });
        println!("Modified array = {:?}", snapshot);
    }

    if let Some(map) = program.get_typed_global::<MapHandle>("main.map_global")? {
        map.insert("three", Value::Int(3));
        let snapshot = map.with_ref(|view| {
            view.iter()
                .map(|(k, v)| (format!("{:?}", k), v.as_int().unwrap_or_default()))
                .collect::<Vec<_>>()
        });
        println!("Modified map = {:?}", snapshot);
    }

    if let Some(lust_point_value) = program.get_global_value("main.lust_point") {
        if let Ok(lust_point_struct) = StructInstance::from_value(lust_point_value) {
            println!(
                "Read Lust Point ({}, {}, \"{}\")",
                lust_point_struct.field::<i64>("x")?,
                lust_point_struct.field::<i64>("y")?,
                lust_point_struct.field::<String>("name")?
            );
        }
    }

    if let Some(table_value) = program.get_global_value("main.table_global") {
        if let Ok(table) = TableHandle::from_value(table_value.clone()) {
            table.insert("three", Value::Int(3));
            let snapshot = table.with_ref(|view| {
                view.iter()
                    .map(|(k, v)| (format!("{:?}", k), v.as_int().unwrap_or_default()))
                    .collect::<Vec<_>>()
            });
            println!("Modified table = {:?}", snapshot);
        } else if let Ok(map) = MapHandle::from_value(table_value) {
            map.insert("three", Value::Int(3));
            let snapshot = map.with_ref(|view| {
                view.iter()
                    .map(|(k, v)| (format!("{:?}", k), v.as_int().unwrap_or_default()))
                    .collect::<Vec<_>>()
            });
            println!("Modified table (as map) = {:?}", snapshot);
        }
    }

    Ok(())
}

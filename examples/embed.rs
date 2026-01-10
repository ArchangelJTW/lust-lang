use lust::embed::LustStructView as _;
use lust::{
    struct_field, ArrayHandle, AsyncDriver, AsyncTaskQueue, EmbeddedProgram, FromLustValue,
    FunctionHandle, LustStructView, MapHandle, NativeExport, NativeExportParam, StringRef,
    StructHandle, StructInstance, Value,
};

#[derive(LustStructView)]
#[lust(type = "main.Point")]
struct PointView<'a> {
    #[lust(field = "x")]
    x: lust::LustInt,
    #[lust(field = "y")]
    y: lust::LustInt,
    #[lust(field = "name")]
    name: StringRef<'a>,
}

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

        extern
            function host_scale(int): int
            function fetch_value(function(int)): Task
        end

        arr_global: Array<int> = [1, 2, 3]

        map_global: Map<string, int> = {["one"] = 1, ["two"] = 2}

        local lust_point: Point = Point {
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

        pub function get_async_value(): Task
            return fetch_value(function(value: int)
                println("callback invoked from Rust with " .. tostring(value))
            end)
        end

        pub function await_fetch(): Option<int>
            local job = fetch_value(function(value: int)
                println("callback (await) saw " .. tostring(value))
            end)
            while true do
                local info = task.info(job)
                if info.state is TaskStatus.Completed then
                    return Option.Some(info.last_result:unwrap())
                elseif info.state is TaskStatus.Failed then
                    println(info.error:unwrap_or("native async failure"))
                end
                task.yield(Option.None)
            end
            return Option.None
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

    program.vm_mut().record_exported_native(NativeExport::new(
        "main.host_scale",
        vec![NativeExportParam::new("value", "int")],
        "int",
    ));

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

    let translate_handle = program.function_handle("main.translate")?;
    let handle_point = program.struct_instance(
        "main.Point",
        [
            struct_field("x", 2_i64),
            struct_field("y", 3_i64),
            struct_field("name", "HandlePoint"),
        ],
    )?;
    let via_handle: StructInstance =
        translate_handle.call_typed(&mut program, (handle_point, 1_i64, 1_i64))?;
    println!(
        "Translated via handle -> ({}, {})",
        via_handle.field::<i64>("x")?,
        via_handle.field::<i64>("y")?
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

    if let Some(point_handle) = program.get_typed_global::<StructHandle>("main.lust_point")? {
        let view = PointView::from_handle(&point_handle)?;
        let name = view.name.as_str();
        println!("Derived PointView ({}, {}, \"{}\")", view.x, view.y, name);
    }

    let queue = AsyncTaskQueue::<FunctionHandle, lust::LustInt>::new();
    program
        .register_async_task_queue::<FunctionHandle, lust::LustInt>("fetch_value", queue.clone())?;

    program.vm_mut().record_exported_native(NativeExport::new(
        "main.fetch_value",
        vec![NativeExportParam::new("callback", "function(int)")],
        "Task",
    ));

    // Simulate Lust calling the native async function
    let lust_task = program.call_raw("main.get_async_value", Vec::new())?;
    let task_handle = match lust_task {
        Value::Task(handle) => handle,
        other => panic!("expected Task, got {other:?}"),
    };

    // Drive the pending job from Rust
    let pending_job = queue
        .pop() // pop_blocking can be used for continuous blocking polling
        .expect("fetch_value should enqueue a pending job");
    let callback = pending_job.args().clone();
    println!("Rust: invoking Lust callback with 77");
    callback.call_typed::<_, ()>(&mut program, 77_i64)?;
    let mut driver = AsyncDriver::new(&mut program);
    pending_job.complete_ok(77_i64);
    driver.pump_until_idle()?;

    // Inspect the task via the VM just like scripts would
    let (state, last_result, last_error) = {
        let vm = program.vm_mut();
        let task = vm.get_task_instance(task_handle).expect("async task info");
        (
            task.state.as_str().to_string(),
            task.last_result.clone(),
            task.error.clone(),
        )
    };
    println!("Async task state = {}", state);
    if let Some(err) = last_error {
        println!("Async task error = {}", err);
    }
    if let Some(value) = last_result.and_then(|v| v.as_int()) {
        println!("Async fetch returned {value}");
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

    if let Some(map_value) = program.get_global_value("main.map_global") {
        if let Ok(map) = MapHandle::from_value(map_value) {
            map.insert("three", Value::Int(3));
            let snapshot = map.with_ref(|view| {
                view.iter()
                    .map(|(k, v)| (format!("{:?}", k), v.as_int().unwrap_or_default()))
                    .collect::<Vec<_>>()
            });
            println!("Modified map = {:?}", snapshot);
        }
    }

    if let Ok(dir) = std::env::var("LUST_DUMP_EXTERNS") {
        let written = program
            .dump_externs_to_dir(&dir)
            .map_err(|err| lust::LustError::Unknown(format!("dump externs: {err}")))?;
        println!("Wrote {} extern stub file(s) under {}", written.len(), dir);
    }

    Ok(())
}

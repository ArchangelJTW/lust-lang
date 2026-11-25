# Архитектура и Комплексный План Внедрения Deterministic Simulation Testing (DST) для Lust Runtime (`lust-lang-fork`)

| Метаданные | Значение |
|---|---|
| **Версия документа** | 7.1 (DA Remediation — Specification Only) |
| **Статус** | Исправленная архитектурная спецификация; runtime gates NOT RUN |
| **Дата** | 2026-11-11 |
| **Целевой репозиторий** | `repos/lust-lang-fork` (ветка `perf/10x-jit-optimization-suite`) |
| **Методология** | Спецификация и рубрика `$mb3-dst` (D1–D8), FoundationDB DST Pattern, ZIO TestClock |
| **Базовый DRS** | **20 / 100 (`DST-HOSTILE`)** ($100 - 30_{D3} - 15_{D1} - 15_{D2} - 10_{D8} - 5_{D7} - 5_{TP}$) |
| **Целевой DRS** | **100 / 100 (`DST-READY`)** (расчётная траектория, требующая исполнения gates: 20 → 40 → 55 → 85 → 85 → 100) |

---

## 0. Область Применимости, Матрица Профилей и Автономный Replay Bundle

### 0.1. Матрица целевых профилей (Profile Matrix)
* **Tier-1 DST (Primary Execution & Concurrency Target)**:
  * *Среда*: `std`, 64-битные платформы (`x86_64-unknown-linux-gnu`, `aarch64-apple-darwin`), числовой профиль `std_64` (`LustInt = i64`, `LustFloat = f64`).
  * *Подсистемы*: Байткод-интерпретатор VM, стек структурированной конкурентности (Fibers, Scopes, Interrupt Masks), First-Class Effects (`effect_abi`), реактивный планировщик `RxScheduler` и координаторы завершения `ScopeJoinCoordinator` / `close_and_wait_quiescent`.
  * *Гарантия*: **Пошаговый Bit-for-Bit Replay** детерминированной трассы при фиксированном Replay Bundle на совместимой платформе.
* **Tier-2 DST (JIT & MIR Execution)**:
  * *Среда*: Cranelift JIT и нативные циклы.
  * *Требование*: Перенос контекста бюджетов (`GAS_REMAINING`, `MEMORY_USED`, `LAST_STATUS`, `MAX_MEMORY_LIMIT`) из глобальных атомиков в инкапсулированный `MirContext`.
  * *Гарантия*: Детерминированное чередование на границах вызовов runtime helpers и компилируемых safepoints.
* **Tier-3 DST (Embedded / no_std)**:
  * *Среда*: `no_std`, 32-битные платформы (`rv32`, `esp32c6`), числовой профиль `nostd_32` (`LustInt = i32`, `LustFloat = f32`).
  * *Гарантия*: **Семантическая эквивалентность переходов состояний**. Симулятор запускается на host-машине с типизацией `nostd_32` без привлечения 64-битных атомиков.

### 0.2. Автономный Replay Bundle и Execution Snapshot Lock
Автономность означает наличие всех исходных и входных данных без Git, сети и ранее заполненного Cargo cache. Совместимый toolchain и системный linker/SDK — явно проверяемые prerequisites; их версии и digest образа среды фиксируются в manifest. Сам `Cargo.lock` не содержит исходников зависимостей.

1. **Снимок входов сборки**:
   - Перед кампанией фиксируется весь исходный checkout: tracked-файлы в их текущем состоянии и явно включённые untracked inputs. Учитываются staged, unstaged и бинарные изменения. Hash дерева, а не один git SHA, идентифицирует исполняемые исходники.
   - Минимально включаются `src/`, `crates/`, `tests/`, `benchmarks/`, `examples/`, `include/`, `Cargo.toml`, `Cargo.lock`, `build.rs`, README и LICENSE-файлы, Cargo/toolchain configuration, fixtures и файлы из `include_str!` / `include_bytes!`. Рекурсивно учитываются manifest каждого workspace member и path dependencies; внешние пути и symlinks не должны уходить за границу снимка.
   - Перечень выше не является закрытым allowlist. Реальные targets определяются через `cargo metadata` для каждого заявленного профиля, включая автопоиск и `[[test]]` / `[[bench]]`. Build scripts и proc macros требуют учёта собственных входов.
   - `git status --ignored --porcelain --untracked-files=all` используется как инвентаризация, а не доказательство полноты. Игнорируемые исходные входы должны быть явно включены и хешированы либо отклонены с `UncommittedIgnoredSourceFile`. Сгенерированные выходы создаются в `OUT_DIR` из упакованных входов. `.git/`, build outputs и host credentials не упаковываются.
2. **Замыкание зависимостей**:
   - В bundle входят `vendor/` с транзитивными registry/git dependencies и Cargo source-replacement configuration. Path dependencies входят в исходное дерево. Запрещены внешние symlinks и незафиксированные загрузки build scripts.
   - Версии, features, target triple, numeric profile, rustc/Cargo, build flags и digest toolchain/SDK фиксируются в `manifest.json`. Переменные сборки передаются по явному allowlist; остальные не наследуются.
3. **Execution Snapshot Lock**:
   - Снимок формируется и запечатывается до компиляции и запуска кампании. Компиляция и выполнение идут только из его распакованной копии; `target/` и `OUT_DIR` находятся вне неизменяемых исходников. Hash входов проверяется до и после сборки; build script не может незаметно изменить их.
   - Изменение рабочей копии после фиксации не влияет на кампанию и её replay. Исходный snapshot сохраняется до завершения кампании; bundle не собирается заново из live checkout.
4. **Состав bundle**:
   - `manifest.json`, `source_tree.tar`, `vendor/`, source-replacement configuration и hashes каждого артефакта.
   - `initial_vfs.tar`, `env.json`, исходная программа/сценарий, seed, алгоритм и версия PRNG, параметры quantum, лимиты шагов и fault configuration.
   - `events.jsonl`: payload внешних ответов, request identity/sequence и точный порядок доставки при совпадающем шаге (`event_id`, `req_seq`, `ready_at_step`, `payload_bytes`).
   - Failing witness: код нарушенного инварианта, шаг и каноническая трасса либо её hash с описанием сериализации; версия scheduler/oracle и параметры shrinking. Результат shrinking воспроизводится с соответствующими ему входами.
5. **Приёмка replay (Gate 6)**:
   - Распаковать bundle в чистую изолированную директорию, проверить hashes и отсутствие выходящих symlinks. Использовать пустой `CARGO_HOME`, отключённую сеть и только объявленный toolchain/SDK.
   - Выполнить `cargo metadata --offline --locked`, `cargo check --offline --locked` и целевые тесты `cargo test --offline --locked` с точными features/targets кампании. Полнота исходников исключает missing-path ошибки, но не гарантирует успешную компиляцию или прохождение тестов.
   - Воспроизвести тот же invariant failure и witness trace; отдельно проверить, что правка live checkout после snapshot не меняет результат. Отсутствующая зависимость, вход или несовпадение trace блокируют gate.

### 0.3. Контракт Внешних Событий и FFI
В Tier-1 симуляции неконтролируемые фоновые потоки ОС полностью исключены:
1. Вызовы FFI и хоста разделяются на:
   - *Детерминированные моки*: немедленный возврат синтетического ответа через `MemFs` или `VirtualConsole`.
   - *Асинхронные события*: запрос помещается в очередь симулятора, а ответ доставляется строго по расписанию планировщика (из детерминированного PRNG в режиме exploration или из `events.jsonl` в режиме replay).
2. Любой немоделированный синхронный блокирующий вызов хоста (`BlockingHostCall`) запрещён внутри симуляции и приводит к немедленному fail-stop падению теста.

---

## 1. Исчерпывающая Инвентаризация Поверхностей (Surfaces Inventory по D1–D8)

### D1: Время и Таймеры (Clock & Timers)
1. **`src/effect_abi.rs:614` (`timeout`)**:
   *Шов*: разделение на чистые волоконные эффекты (управляемые через `TimerQueue` в `LustClock`) и изолированные хостовые вызовы.
2. **`src/vm/stdlib.rs:502` (`os.time`, `os.date`)**:
   *Шов*: проброс `WallClock::now_epoch_nanos()` через контекст `VM`.
3. **`src/vm/stdlib.rs:541` (`os.sleep`)**:
   *Шов*: перевод текущего волокна в состояние `TaskState::Yielded(TimerId)` в очереди виртуальных таймеров без блокировки потока ОС.
4. **`src/rx/scheduler.rs:62, 120` & `src/rx/lifecycle.rs:316`**:
   *Шов*: раздельные детерминированные протоколы ожидания квиесценции для закрытого и открытого скоупа.
5. **Разделение контрактов времени**:
   * `WallClock`: календарное время для `os.time()` / `os.date()`.
   * `MonotonicClock`: строго монотонный счётчик наносекунд для дедлайнов таймеров.
   * `TimerControl`: интерфейс управления виртуальным временем (доступен исключительно тестовому harness).

### D2: Энтропия и Генерация Случайных Чисел (PRNG)
1. **`src/vm/stdlib.rs:20`**: `static RNG: OnceLock<Mutex<StdRng>>`.
   *Шов*: локализация генератора внутри инстанса `VM` (`vm.rng: Box<dyn RngCore + Send>`). Конструктор `VM::with_seed(seed: u64)`.
2. **`src/vm/stdlib.rs:2944` (`math.randomseed`)**: Мутирует локальный RNG конкретной `VM`.

### D3: Хранилище, VFS и Файловая Система (Storage & I/O)
1. **Интерфейс `VirtualFileSystem` (ровно 12 методов)**:
   ```rust
   pub trait VirtualFileSystem: Send + Sync {
       fn read(&self, path: &Path) -> Result<Vec<u8>, std::io::Error>;
       fn read_to_string(&self, path: &Path) -> Result<String, std::io::Error>;
       fn write(&self, path: &Path, contents: &[u8]) -> Result<(), std::io::Error>;
       fn append(&self, path: &Path, contents: &[u8]) -> Result<(), std::io::Error>;
       fn create_dir_all(&self, path: &Path) -> Result<(), std::io::Error>;
       fn remove_file(&self, path: &Path) -> Result<(), std::io::Error>;
       fn remove_dir_all(&self, path: &Path) -> Result<(), std::io::Error>;
       fn rename(&self, from: &Path, to: &Path) -> Result<(), std::io::Error>;
       fn read_dir(&self, path: &Path) -> Result<Vec<PathBuf>, std::io::Error>;
       fn metadata(&self, path: &Path) -> Result<VfsMetadata, std::io::Error>;
       fn canonicalize(&self, path: &Path) -> Result<PathBuf, std::io::Error>;
       fn exists(&self, path: &Path) -> bool;
   }
   ```
2. **Семантика VFS и нормализация**:
   * Все пути приводятся к каноническому POSIX-виду относительно `/workspace`.
   * `read_dir` гарантирует детерминированную лексикографическую сортировку возвращаемых путей.
   * `canonicalize` резолвит сегменты `.` и `..` внутри виртуального корня.
3. **Ликвидация обходов**: Прямые системные вызовы `std::fs::canonicalize` в `src/bindgen.rs:122` и `Path::exists()` в `src/modules/mod.rs:504` перенаправляются в `vm.vfs`.
4. **Контроль хостового I/O: lint, facade guard и изоляция процесса**:
   * **Уровень компиляции (Compile-Time Blacklist Gate)**:
     Следующий явный baseline запрещает известные файловые входы через `clippy.toml`. Это не перехват системных вызовов и не исчерпывающее доказательство отсутствия I/O:
     ```toml
     disallowed-methods = [
         "std::fs::read",
         "std::fs::read_to_string",
         "std::fs::write",
         "std::fs::read_dir",
         "std::fs::create_dir",
         "std::fs::create_dir_all",
         "std::fs::remove_file",
         "std::fs::remove_dir",
         "std::fs::remove_dir_all",
         "std::fs::rename",
         "std::fs::copy",
         "std::fs::hard_link",
         "std::fs::soft_link",
         "std::fs::metadata",
         "std::fs::symlink_metadata",
         "std::fs::canonicalize",
         "std::fs::set_permissions",
         "std::path::Path::exists",
         "std::path::Path::try_exists",
         "std::path::Path::is_file",
         "std::path::Path::is_dir",
         "std::path::Path::is_symlink",
         "std::path::Path::metadata",
         "std::path::Path::symlink_metadata",
         "std::path::Path::canonicalize",
         "std::path::Path::read_dir",
         "std::path::Path::read_link",
         "std::fs::read_link",
         "std::fs::exists",
         "std::fs::DirBuilder::new",
         "std::fs::DirBuilder::create",
         "std::fs::File::open",
         "std::fs::File::create",
         "std::fs::File::create_new",
         "std::fs::File::options",
         "std::fs::OpenOptions::new",
         "std::fs::OpenOptions::open"
     ]
     disallowed-types = [
         "std::fs::File",
         "std::fs::OpenOptions",
         "std::fs::DirEntry",
         "std::fs::ReadDir",
         "std::fs::Metadata",
         "std::fs::Permissions",
         "std::fs::DirBuilder"
     ]
     ```
     Команда проверки production workspace: `cargo clippy --all-targets -- -D clippy::disallowed_methods -D clippy::disallowed_types`, отдельно для каждого поддержанного набора features/target. Toolchain закреплён; неизвестный или неразрешившийся lint path считается ошибкой конфигурации, а не успешным запретом.
     Единственное локальное разрешение host filesystem lint: `src/vfs/os_fs.rs`. Новые `allow`/`expect`, платформенные API (`std::os::*::fs`), raw descriptors, FFI и зависимости с I/O проверяются аудитом доступных из SUT путей; транзитивный код не покрывается этим Clippy gate автоматически.
   * **Runtime facade guard**:
     Каждая операция `OsFs` проверяет режим исполнения до host I/O, включая сохранённые ранее handles; попытка в симуляции даёт `DirectHostFsBypassPanic`. Проверки только при создании VM недостаточно. Guard не перехватывает прямой `std::fs`, libc или стороннюю библиотеку.
   * **Изоляция процесса**:
     Gate 3 запускает реальный SUT в дочернем процессе под внешним watchdog с проверенным для данной платформы запретом host filesystem I/O вне минимального bootstrap allowlist. После bootstrap доступы SUT идут через MemFs. Матрица Linux/macOS фиксирует конкретный backend и границы разрешений. Если backend отсутствует или bypass canary проходит, профиль получает `UNVERIFIED` и gate не закрывается; обычные chmod и переменная окружения sandbox не заменяют.
   * **Отрицательные тесты без конфликта с `--all-targets`**:
     - Отдельный fixture crate `tests/compile_fail/fs_gate/` исключён из production workspace и содержит явный `[[test]] name = "direct_std_fs", path = "direct_std_fs.rs"`. В нём используется тот же `clippy.toml`, совпадение проверяется по hash.
     - Из cwd fixture выполняется `cargo clippy --test direct_std_fs -- -D clippy::disallowed_methods -D clippy::disallowed_types`. Каждый canary проверяется отдельным прогоном: `fs::read`, конструкторы File/OpenOptions/DirBuilder, запрещённый тип, методы Path и платформенные входы. Требуются ненулевой exit и соответствующий lint code/span именно canary; `no test target`, ошибки конфигурации и посторонние ошибки не засчитываются. Разрешённый контрольный пример обязан собираться.
     - `tests/dst_vfs_bypass_negative.rs` отдельно проверяет facade panic и отклонение raw host I/O из SUT. Canary обращается к существующему host-only файлу, отсутствующему в MemFs; вне изоляции контроль подтверждает, что файл читается. Обход blacklist должен обнаруживаться независимо от OsFs.

### D4: Конкурентность, Шедулинг и Изоляция Процессного Состояния (Scheduling & State)
1. **Изоляция 4 глобальных атомиков MIR (`src/mir/runtime_helpers.rs:5`)**:
   * `GAS_REMAINING`, `MEMORY_USED`, `LAST_STATUS`, `MAX_MEMORY_LIMIT` переносятся в структуру `MirContext`.
   * Активный контекст хранится в потоковом RAII-стеке `CURRENT_MIR_STACK` для безопасной реентрантности FFI.
2. **Изоляция Lua Registry (`src/lua_compat/mod.rs:38`)**:
   * TLS-хранилище реестра и монотонный счётчик `NEXT_LUA_FUNCTION_ID` привязываются к инстансу `LuaCompatContext`.
3. **Полный контракт возобновляемого продолжения (FiberContinuation)**:
   Сохраняются все компоненты контекста:
   ```rust
   pub struct FiberContinuation {
       pub call_stack: Vec<Box<CallFrame>>, // стек вызовов с регистрами фреймов
       pub ip: usize,                       // точный instruction pointer
       pub pending_return_value: Option<Value>,
       pub pending_return_dest: Option<Register>,
       pub yield_dest: Option<Register>,
       pub fiber_context: FiberContext,     // маски прерываний, pending_interrupt, scope_id
       pub upvalues: Vec<Rc<RefCell<Upvalue>>>,
   }
   ```
4. **Раздельные Протоколы Квиесценции для Двух Entrypoint (Закрытый vs Открытый Скоуп)**:
   * **Случай А: `close_and_wait_quiescent` (Закрытый Скоуп — Permanent Quiescence)**:
     1. Под замком `registry.lock()`:
        - Атомарно устанавливается `reg.closed_scopes.insert(scope_id)` — регистрация новых участников для данной идентичности scope блокируется до конца её жизненного цикла (см. правило retirement ниже).
        - Проверка: если `active_participants == 0` — немедленный возврат `true`.
        - Иначе: регистрируется `WaiterEntry` в состоянии `Registered`. Если задан `timeout`, в таймерах регистрируется дедлайн.
     2. Освобождение мьютекса (`Mutex::unlock()`).
     3. Уступка управления: `Scheduler::yield_until_woken(waiter_id)`.
     4. При выходе последнего участника (`active == 0`) под замком `registry.lock()`: waiter переводится в `Woken`, таймер отменяется, волокно переводится в готовую очередь.
     5. При тайм-ауте под замком `registry.lock()`: если waiter ещё `Registered`, переводится в `TimedOut`. Если уже `Woken`, тайм-аут игнорируется.
     6. При пробуждении под замком `registry.lock()`: если `Woken` — возврат `true` (admission для данной идентичности остаётся закрытым); если `TimedOut` — финальная проверка `active == 0` и возврат результата.
   * **Случай Б: `wait_for_quiescence` (Открытый Скоуп — Transient Quiescence)**:
     1. Выполняет шаги 1–4 регистрации и ожидания под замком `registry.lock()`, но **без закрытия admission**.
     2. При пробуждении по `Woken` под замком `registry.lock()`: **обязательная повторная проверка `active_participants == 0`**:
        - Если `active == 0` — квиесценция подтверждена, возврат `true`.
        - Если до возобновления waiter зарегистрировался новый участник (`active > 0`): **немедленный возврат `true` запрещён!** Если дедлайн не истёк — под тем же замком waiter переводится обратно в `Registered`, восстанавливается таймер на **исходный абсолютный дедлайн** (шаг А.4 отменил предыдущий таймер), затем выполняется атомарная парковка по контракту ниже. Продлевать дедлайн на исходную duration запрещено. При истечении дедлайна возвращается `false`.
     3. `TimedOut` обрабатывается как в А.6: финальная проверка `active == 0` под замком. При `timeout = None` таймер не создаётся. После терминального результата waiter удаляется, оставшиеся timer/ready entries инвалидируются.
   * **Retirement и идентичность scope**: в текущем `src/rx/scheduler.rs::retire_scope` удаляется `closed_scopes`, а `lifecycle.rs::join` вызывает retirement. Поэтому нынешний код не обеспечивает постоянного запрета повторной регистрации. Внедрение обязано сохранить закрытый tombstone для той же идентичности либо использовать generation-qualified capability, отклоняющую устаревшие handles. Очистка памяти допустима только при невозможности обращения через старую идентичность; отсутствие записи не означает разрешение регистрации. Gate 2 проверяет `close → retire → register(old_scope)` и retirement до возобновления waiter. Это требование изменения реализации, а не утверждение о текущем runtime.
   * **Retained wake / атомарная парковка**: `yield_until_woken` сначала проверяет состояние waiter; `Woken`/`TimedOut` не паркуются. Проверка и установка parked-состояния атомарны относительно notify и timeout, с явно закреплённым порядком registry/scheduler locks. Запрещено удерживать registry lock при фактической уступке. Каждый цикл повторной регистрации имеет generation token; устаревшие timer/ready события игнорируются, enqueue выполняется не более одного раза на generation.
   * **Обязательный контрпример Gate 2**: deadline=10, notify при t=2 отменяет таймер; re-admission при t=3; resume при t=4 повторно паркует waiter; новый участник не выходит до t=10. Результат обязан быть `false` при deadline, без внешнего watchdog. Тест убивает мутант «не восстановить таймер»; отдельный мутант «вернуть true по Woken без recheck» выявляется тем же re-admission порядком. Это проверяется на реальном coordinator и scheduler, а не только на модели протокола.

### D5: Адресная память и Порядок Итерации (Memory Address Nondeterminism)
1. **Квалификация идентификаторов кэшей специализаций (`src/bytecode/value.rs:2169`)**:
   * Замена сырых указателей памяти `arr_ptr: usize` на композитный детерминированный ключ `(VmId, ArraySequenceId)`.
   * `VmId` выделяется монотонно планировщиком; `ArraySequenceId` выделяется монотонно внутри конкретной `VM`. Это гарантирует отсутствие коллизий в общем TLS-кэше при работе нескольких VM на одном потоке.

### D6: Окружение и Ввод-Вывод (Environment & Console)
1. **Виртуализация окружения (`src/vm/stdlib.rs:733`)**: Таблица `VirtualEnv` изолирует переменные процесса.
2. **Виртуализация консоли (`src/vm/stdlib.rs:449`)**: `VirtualConsole` перехватывает stdin/stdout/stderr в буферы симуляции.

---

## 2. Формальные Инварианты с «Зубами» (High-Teeth Invariants)

Каждый инвариант снабжён независимым предикатом завершимости (liveness) и проверяется fail-stop:

```
+-----------------------------------------------------------------------------------------------------+
|                                      FAIL-STOP INVARIANTS                                           |
+--------------------------+------------------------------------+-------------------------------------+
| Идентификатор            | Формулировка Safety                | Формулировка Liveness (Защита завис)|
+--------------------------+------------------------------------+-------------------------------------+
| INV-FIBER-01 (LIFO)      | Exactly-Once: каждый               | Completion Guarantee: к моменту     |
|                          | зарегистрированный ресурс          | закрытия скоупа счётчик выполненных |
|                          | финализируется строго 1 раз        | финализаторов обязан быть равен     |
|                          | в точном порядке LIFO.             | общему числу зарегистрированных (>0)|
+--------------------------+------------------------------------+-------------------------------------+
| INV-FIBER-02 (Masking)   | Preserved Delivery: отложенная     | Bound Safe Point: доставка отмены   |
|                          | отмена не теряется и сохраняется   | обязана произойти ровно на N=1      |
|                          | внутри маски любого уровня.        | инструкции после выхода из внешней  |
|                          |                                    | маски.                              |
+--------------------------+------------------------------------+-------------------------------------+
| INV-FIBER-03 (Teardown)  | Full Quiescence: Scope не может    | Termination Bound: все дочерние     |
|                          | стать Done, пока дочерние задачи   | задачи обязаны достичь терминала    |
|                          | не перейдут в Completed, Failed    | за ограниченный квант шагов         |
|                          | или Stopped (включая Yielded!).    | max_execution_steps (fail-stop).    |
+--------------------------+------------------------------------+-------------------------------------+
| INV-RX-01 (Quiescence)   | Zero-Leak: после завершения        | Active Rejection: сценарий          |
|                          | join() ни один callback не         | производит попытку записи после     |
|                          | выполняет запись в sink.           | join, проверяя отказ/дроп.          |
+--------------------------+------------------------------------+-------------------------------------+
| INV-BUDGET-01 (Gas)      | Progress Accounting: при каждом    | Non-Stall Invariant: суммарный газ  |
|                          | шаге вычислений gas_used строго    | растёт пропорционально числу        |
|                          | монотонно возрастает на вес опкода.| выполненных инструкций.             |
+--------------------------+------------------------------------+-------------------------------------+
```

### Независимый Лимит Шагов и Внешний Watchdog
Для гарантированной защиты от бесконечных циклов в симуляции:
1. **Счётчик шагов симулятора (`sim_step_counter`)**: Физический счётчик циклов шедулера в тестовом harness, полностью независимый от гостевого `gas_used` (который может быть сломан мутантом). При достижении `max_execution_steps` тест аварийно останавливается с вердиктом `EXHAUSTION_LIMIT_REACHED`.
2. **Внешний Watchdog Процесса**: Тестовый процесс оборачивается в системный watchdog с тайм-аутом 5 секунд. Срабатывание watchdog считается аварийным исходом (`FAIL: WATCHDOG_TIMEOUT`) и категорически не засчитывается как успешное обнаружение дефекта.

---

## 3. Точный Мутационный Контроль Приёмки (Mutation Acceptance Gate)

DST-тест считается валидным тогда и только тогда, когда строго доказана его чувствительность к целевым дефектам по следующей цепочке:

$$\text{Baseline GREEN} \longrightarrow \text{Инъекция мутанта} \longrightarrow \text{Подтверждение выполнения мутированной ветви} \longrightarrow \text{Падение конкретного оракула (RED)}$$

### Обязательные мутанты:
1. **Мутант M1 (Списание газа)**:
   - *Точка инъекции*: `src/vm/execution.rs` (закомментировано списание газа в цикле инструкций).
   - *Execution Witness*: подтверждение `MUTATION_M1_EXECUTION_COUNT > 0` в логе.
   - *Ожидаемый RED*: падение оракула `INV-BUDGET-01` (зафиксирован шаг прогресса без изменения газа).
2. **Мутант M2 (Пропуск финализатора при панике)**:
   - *Точка инъекции*: `src/vm/tasks.rs` (пропуск LIFO-финализации в ветке `TaskState::Failed`).
   - *Execution Witness*: подтверждение `MUTATION_M2_EXECUTION_COUNT > 0` в логе.
   - *Ожидаемый RED*: падение оракула `INV-FIBER-01-EXACTLY-ONCE` (число выполненных финализаторов меньше зарегистрированных).
3. **Мутант M3 (Гонка в Rx Admission)**:
   - *Точка инъекции*: `src/rx/scheduler.rs` (удалена проверка `reg.closed_scopes.contains(&scope_id)`).
   - *Execution Witness*: подтверждение вызова `register_participant` на закрытом скоупе.
   - *Ожидаемый RED*: падение оракула `INV-RX-01` (зафиксирован запуск callback после завершения join).

**Правило приёмки**: Падение из-за ошибки компиляции, общего тайм-аута процесса или паники инициализации **не засчитывается** как убийство мутанта.

---

## 4. Поэтапный План Внедрения и Устранение Зависимостей (Phased Implementation Roadmap)

Планировщик разделён на два этапа для исключения взаимных зависимостей:
* **Фаза 1–2: `MinimalSimScheduler`** — минимальный кооперативный цикл: поддержка парковки/пробуждения (`TaskState::Yielded(TimerId)`), очередь таймеров и шаг виртуального времени.
* **Фаза 4: `PreemptiveExplorationScheduler`** — полномасштабный планировщик вытеснения с quantum-шагами инструкций и псевдослучайным перемешиванием очередей.

```
+----------------------------------------------------------------------------------------------------+
|                                МАТРИЦА ЗАКРЫТИЯ ПОВЕРХНОСТЕЙ                                       |
+--------------------------------------+---------------------+---------------------------------------+
| Подсистема / Поверхность             | Закреплённая Фаза   | Проверяющий тест и Gate               |
+--------------------------------------+---------------------+---------------------------------------+
| VM RNG (static RNG)                  | Фаза 1              | `tests/dst_state_isolation.rs`        |
| MIR Atomics (GAS, MEM, STATUS, LIMIT)| Фаза 1              | `tests/dst_mir_isolation.rs`          |
| Lua Registry & ArrayId Namespacing   | Фаза 1              | `tests/dst_tls_reentrancy.rs`         |
| MinimalSimScheduler (Cooperative)    | Фаза 1              | `tests/dst_micro_step.rs`             |
| LustClock (Wall/Monotonic/TimerQueue)| Фаза 2              | `tests/dst_clock_determinism.rs`      |
| stdlib os.sleep & os.time            | Фаза 2              | `tests/dst_stdlib_time.rs`            |
| ScopeJoinCoordinator Quiescence      | Фаза 2              | `tests/dst_rx_quiescence.rs`          |
| VirtualFileSystem (все 12 методов)   | Фаза 3              | `tests/dst_vfs_contract.rs`           |
| Module Loader & Path Normalization   | Фаза 3              | `tests/dst_module_isolation.rs`       |
| VirtualEnv & VirtualConsole          | Фаза 3              | `tests/dst_env_console.rs`            |
| Clippy & Runtime Fail-Closed VFS Gate| Фаза 3              | `tests/dst_vfs_bypass_negative.rs`    |
| Fuel-based Fiber Preemption (DES)    | Фаза 4              | `tests/dst_fiber_quantum.rs`          |
| PreemptiveExplorationScheduler       | Фаза 4              | `tests/dst_fiber_interleaving.rs`     |
| BlockingHostCall Contract            | Фаза 4              | `tests/dst_host_call_isolation.rs`    |
| Invariant Engine (INV 01-03, RX, GAS)| Фаза 5              | `tests/dst_invariants_teeth.rs`       |
| FaultInjector (VFS, Interrupt, OOM)  | Фаза 5              | `tests/dst_fault_injection.rs`        |
| 100% Mutation Acceptance (M1-M3)     | Фаза 5              | `scripts/run_dst_mutation_gate.sh`    |
| CI Campaign (100k seeds + Shrinking) | Фаза 6              | `.github/workflows/dst_nightly.yml`  |
| Zero-Cost Verification (TOST B01-B15)| Фаза 6              | `benchmarks/effects_bench.rs`            |
+--------------------------------------+---------------------+---------------------------------------+
```

### Подробный регламент фаз:

#### Фаза 1: Изоляция Состояния и Микро-Харнесс (State Isolation & Minimal Loop)
* **Задачи**:
  1. Вынести `static RNG` в поле `VM.rng` с конструктором `VM::with_seed(u64)`.
  2. Перенести все 4 статических атомика MIR (`GAS_REMAINING`, `MEMORY_USED`, `LAST_STATUS`, `MAX_MEMORY_LIMIT`) в структуру `MirContext` с поддержкой реентрантного стека `CURRENT_MIR_STACK`.
  3. Добавить квалификацию `(VmId, ArraySequenceId)` в кэши специализаций `value.rs`.
  4. Привязать Lua Registry к `LuaCompatContext`.
  5. Реализовать `MinimalSimScheduler`: поддержка `TaskState::Yielded`, одноволоконный шаг и микро-оракул для проверки мутанта M1.
* **Критерий приёмки (Gate 1)**:
  - 4 параллельных VM на одном потоке с разными сидами работают на 100% изолированно.
  - Мутант M1 успешно убивается микро-оракулом при подтверждённом выполнении мутированной ветви.

#### Фаза 2: Разделение Времени и Таймеров
* **Задачи**:
  1. Внедрить разделение трейтов времени: `WallClock` (эпоха) и `MonotonicClock` (таймеры).
  2. Заменить вызов `std::thread::sleep` в `os.sleep` на приостановку волокна в очереди виртуальных таймеров через `MinimalSimScheduler`.
  3. Перевести `ScopeJoinCoordinator` и `close_and_wait_quiescent` на раздельные протоколы квиесценции (закрытый vs открытый скоуп) под единым замком.
* **Критерий приёмки (Gate 2)**:
  - Сценарий с задержкой 24 часа виртуального времени выполняется детерминированно за фиксированное число шагов планировщика (< 100 шагов для фиксированного сценария; wall-clock измеряется отдельно и не является детерминированным критерием).
  - Принудительно заданные порядки `notify-before-park`, `notify-after-park`, `timeout-before-notify`, `timeout-after-notify`, а также повторная регистрация в `wait_for_quiescence`, восстановление отменённого таймера на исходный deadline, stale-event rejection и `close → retire → register(old_scope)` подтверждают отсутствие зависаний и ложных квиесценций.

#### Фаза 3: Полное Замыкание VFS и Окружения
* **Задачи**:
  1. Реализовать `VirtualFileSystem` со всеми 12 методами (включая `canonicalize`, `append`, `rename`, `remove_dir_all`).
  2. Внедрить `MemFs` с детерминированной нормализацией путей и сортировкой директорий.
  3. Перенаправить все обращения `Path::exists()` и `canonicalize()` из `src/modules/` и `src/bindgen.rs` в VFS. Включить Clippy-правила `disallowed-methods` и `disallowed-types` на запрет `std::fs`.
  4. Внедрить `VirtualEnv` и `VirtualConsole`.
* **Критерий приёмки (Gate 3)**:
  - Каждый перечисленный direct-I/O canary отклоняется Clippy при проверке в изолированном fixture crate (`tests/compile_fail/fs_gate/`), а вызов `OsFs` в режиме симуляции вызывает панику `DirectHostFsBypassPanic` (проверено негативным тестом `dst_vfs_bypass_negative`).
  - Компиляция и запуск модульного проекта происходят в чистой `MemFs` под проверенной process isolation с host-only canary; bootstrap allowlist фиксируется в отчёте.

#### Фаза 4: Квантование и Чередование Волокон (DES)
* **Задачи**:
  1. Реализовать fuel-based квантование байткода с сохранением полного continuation (`FiberContinuation`).
  2. Добавить `PreemptiveExplorationScheduler`, перемешивающий порядок исполнения готовых волокон на основе `SimRng`.
  3. Зафиксировать контракт `BlockingHostCall`: внешние хостовые вызовы изолируются и не вносят недетерминизм в Tier-1.
* **Критерий приёмки (Gate 4)**:
  - Обнаружение искусственной гонки волокон на разделяемой таблице гарантируется в пределах 500 итераций сида с фиксацией точного падающего сида.

#### Фаза 5: Полнозубые Инварианты и Генерация Сбоев (Fault Injection)
* **Задачи**:
  1. Внедрить мониторинг инвариантов `INV-FIBER-01..03`, `INV-RX-01`, `INV-BUDGET-01` с обязательными предикатами завершимости (liveness bounds) и защитой от зависания через независимый `sim_step_counter`.
  2. Разработать инъектор сбоев `FaultConfiguration` (ошибки VFS, спонтанные отмены, отказы памяти).
  3. Провести полное мутационное тестирование: подтвердить убийство мутантов M1, M2, M3 конкретными оракулами при обязательном наличии execution witness.
* **Критерий приёмки (Gate 5)**:
  - 100% мутантов убиты соответствующими инвариантами; подтверждено выполнение мутированных ветвей.

#### Фаза 6: Масштабирование, CI и Статистическая Zero-Cost Верификация
* **Задачи**:
  1. Внедрить генерацию автономного `Replay Bundle` (`source_tree.tar` со всеми targets + `manifest.json` + `events.jsonl`) с автоматическим поиском минимального воспроизводящего примера (shrinking).
  2. Настроить ночные прогоны на 100 000 сидов в CI.
  3. Провести валидацию производительности production-сборки:
     * *Структурный критерий*: Мономономорфизация и нулевой оверхед — все трейты абстракций времени и VFS параметризуются статически (zero-sized marker types). В production (`cfg(not(feature = "sim"))`) все методы инлайнятся без `Box<dyn ...>` на hot-path.
     * *Статистический критерий (TOST Equivalence)*: Проверка эквивалентности двух односторонних t-тестов (Two One-Sided T-tests) на бенчмарках Criterion B01–B15. 95% доверительный интервал относительного изменения $[\Delta_{lower}, \Delta_{upper}]$ обязан целиком лежать внутри границы $[-1.5\%, +1.5\%]$. Если интервал пересекает границу допуска, результат `INCONCLUSIVE`; если целиком лежит вне допуска — `FAIL` эквивалентности. Оба результата блокируют performance gate. Baseline/candidate собираются одним toolchain с одинаковыми flags, входами и проверкой результатов; число независимых повторов и правило остановки фиксируются до измерений. Повторять тест до случайного PASS запрещено.
* **Критерий приёмки (Gate 6)**:
  - Официальное присвоение статуса **`DST-READY`**.

---

## 5. Корректный Расчёт DRS (DST-Readiness Score)

Расчёт ведётся строго по канонической формуле SoT рубрики `$mb3-dst`:
$$\text{DRS} = 100 - (D3_{FAIL} \times 30) - (D1_{FAIL} \times 15) - (D2_{FAIL} \times 15) - (D8_{VACUOUS} \times 10) - (D7_{UNINSTR} \times 5) - (TP_{hazards} \times 5)$$

Штрафы снимаются строго по мере завершения соответствующих фаз:

| Этап | $D1$ (Время) | $D2$ (PRNG) | $D3$ (VFS) | $D4$ (DES) | $TP$ (Изоляция/MIR) | $D7$ (Инварианты) | $D8$ (Сбои/Мутанты) | **Итоговый DRS** | Статус |
|---|---|---|---|---|---|---|---|---|---|
| **Текущее состояние** | FAIL (-15) | FAIL (-15) | FAIL (-30) | 0 | FAIL (-5) | UNINSTR (-5) | VACUOUS (-10) | **20** | `DST-HOSTILE` |
| **После Фазы 1** | FAIL (-15) | **PASS (0)** | FAIL (-30) | 0 | **PASS (0)** | UNINSTR (-5) | VACUOUS (-10) | **40** | `DST-BLOCKED` |
| **После Фазы 2** | **PASS (0)** | **PASS (0)** | FAIL (-30) | 0 | **PASS (0)** | UNINSTR (-5) | VACUOUS (-10) | **55** | `DST-BLOCKED` |
| **После Фазы 3** | **PASS (0)** | **PASS (0)** | **PASS (0)** | 0 | **PASS (0)** | UNINSTR (-5) | VACUOUS (-10) | **85** | `Core seams implemented; gates pending`|
| **После Фазы 4** | **PASS (0)** | **PASS (0)** | **PASS (0)** | 0 | **PASS (0)** | UNINSTR (-5) | VACUOUS (-10) | **85** | `Core seams implemented; gates pending`|
| **После Фаз 5–6**| **PASS (0)** | **PASS (0)** | **PASS (0)** | 0 | **PASS (0)** | **PASS (0)** | **PASS (0)** | **100**| `DST-READY (Full)`|

*Пояснения к расчёту*:
- $D4$ (DES) имеет вес 0 в формуле вычета DRS, но является функциональным предусловием для перехода от базового Core к полномасштабной симуляции.
- $D7$ и $D8$ окончательно переходят в `PASS` в Фазе 5 после завершения полного инвариантного мониторинга и 100% покрытия мутационного гейта, что доводит DRS с 85 до 100.

### 5.1. Граница принятия спецификации

DRS — расчётная оценка по перечисленным поверхностям, не доказательство детерминизма или безопасности. Статус DST-READY присваивается только после всех обязательных gates для конкретного профиля и воспроизводимых артефактов. На момент редакции 7.1 runtime gates, автономный replay, мутационные кампании и performance equivalence **NOT RUN** в рамках этого DA. Отсутствие реализации запланированного механизма само по себе не дефект спецификации; закрытие замечания на уровне контракта не закрывает runtime gate.

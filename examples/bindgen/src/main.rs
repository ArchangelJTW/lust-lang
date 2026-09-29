pub mod game_bindings {
    include!(concat!(env!("OUT_DIR"), "/game_bindings.rs"));
}

fn main() -> lust::Result<()> {
    let scripts_dir = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("scripts");
    // Resolve `use player.*` from scripts/player.lust without listing it manually.
    let mut program = lust::EmbeddedProgram::builder()
        .with_base_dir(scripts_dir.clone())
        .entry_module("main")
        .compile()?;

    let mut bindings = game_bindings::GameBindings::from_program(&mut program);
    let mut player = game_bindings::Player::new(&mut bindings, 42)?;
    let label = player.display_label()?;
    let description = player.describe()?;
    player = player.rename("renamed-player")?;
    let id = player.id()?;
    let name = player.name()?;
    let fetched = bindings.get_player(7)?;
    let fetched = bindings.rename_player(fetched, "through-free-function")?;

    let mut other_program = lust::EmbeddedProgram::builder()
        .with_base_dir(scripts_dir)
        .entry_module("main")
        .compile()?;
    let mut other_bindings = game_bindings::GameBindings::from_program(&mut other_program);
    let foreign_player = other_bindings.get_player(99)?;
    assert!(bindings
        .rename_player(foreign_player, "must-not-cross-programs")
        .is_err());

    let record = bindings.get_record(8)?;
    let record_player = record.player()?;
    let record_label = record_player.display_label()?;
    let optional_name = bindings.lookup_name(id)?;
    let doubled_id = bindings.double_id(id)?;
    let status = bindings.get_status()?;
    let ready = status.is_ready(&mut bindings)?;
    let status = bindings.echo_status(status)?;
    let status_text = match status {
        game_bindings::GameStatus::Ready => "ready".to_string(),
        game_bindings::GameStatus::Score(score) => format!("score-{score}"),
        game_bindings::GameStatus::Player(player) => format!("player-{}", player.id()?),
    };

    println!(
        "player={name} ({label}; {description}), id={id}, doubled={doubled_id}, optional={optional_name:?}, other={}, nested={record_label}, ready={ready}, status={status_text}",
        fetched.name()?
    );
    Ok(())
}

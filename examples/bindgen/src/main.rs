pub mod game_bindings {
    include!(concat!(env!("OUT_DIR"), "/game_bindings.rs"));
}

fn main() -> lust::Result<()> {
    let source = include_str!("../scripts/main.lust");
    let mut program = lust::EmbeddedProgram::builder()
        .module("main", source)
        .entry_module("main")
        .compile()?;

    let mut bindings = game_bindings::GameBindings::from_program(&mut program);
    let player = bindings.get_player(42)?;
    let player = bindings.rename_player(player, "renamed-player")?;
    let id = player.id()?;
    let name = player.name()?;
    let optional_name = bindings.lookup_name(id)?;
    let doubled_id = bindings.double_id(id)?;
    let status = bindings.get_status()?;
    let status = bindings.echo_status(status)?;
    let status_text = match status {
        game_bindings::GameStatus::Ready => "ready".to_string(),
        game_bindings::GameStatus::Score(score) => format!("score-{score}"),
        game_bindings::GameStatus::Player(player) => format!("player-{}", player.id()?),
    };

    println!(
        "player={name}, id={id}, doubled={doubled_id}, optional={optional_name:?}, status={status_text}"
    );
    Ok(())
}

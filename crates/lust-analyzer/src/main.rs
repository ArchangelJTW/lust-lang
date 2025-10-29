mod analysis;
mod backend;
mod diagnostics;
mod semantic_tokens;
mod utils;
#[tokio::main]
async fn main() {
    backend::run().await;
}

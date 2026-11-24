//! napi-rs addon. The VM is constructed, used, and dropped on its actor thread.
use std::sync::{mpsc, OnceLock};
use napi::{Error, Result, Status};
use napi_derive::napi;

enum Request { Eval(String, mpsc::SyncSender<std::result::Result<String, String>>), Stop }
static ACTOR: OnceLock<mpsc::Sender<Request>> = OnceLock::new();

fn actor() -> &'static mpsc::Sender<Request> {
    ACTOR.get_or_init(|| {
        let (tx, rx) = mpsc::channel::<Request>();
        std::thread::Builder::new().name("lust-napi-rs-vm".into()).spawn(move || {
            while let Ok(req) = rx.recv() {
                match req {
                    Request::Eval(source, reply) => {
                        // EmbeddedProgram and its !Send VM never leave this actor thread.
                        let value = lust::EmbeddedProgram::builder()
                            .module("main", &source)
                            .entry_module("main")
                            .compile()
                            .map_err(|e| format!("compile error: {e}"))
                            .and_then(|mut program| {
                                program
                                    .vm_mut()
                                    .call("main.main", vec![])
                                    .map_err(|e| format!("runtime error: {e}"))
                            })
                            .map(|v| format!("{v:?}"));
                        let _ = reply.send(value);
                    }
                    Request::Stop => break,
                }
            }
        }).expect("spawn Lust VM actor");
        tx
    })
}

fn eval(source: String) -> Result<String> {
    let (tx, rx) = mpsc::sync_channel(1);
    actor().send(Request::Eval(source, tx)).map_err(|_| Error::new(Status::GenericFailure, "VM actor stopped"))?;
    rx.recv().map_err(|_| Error::new(Status::GenericFailure, "VM actor dropped reply"))?
        .map_err(|message| Error::new(Status::GenericFailure, message))
}

#[napi(js_name = "evalLust")]
pub fn eval_lust(source: String) -> Result<String> { eval(source) }

#[napi(js_name = "evalLustAsync")]
pub async fn eval_lust_async(source: String) -> Result<String> { eval(source) }

#[napi]
pub fn shutdown() -> Result<()> {
    if let Some(actor) = ACTOR.get() { actor.send(Request::Stop).map_err(|_| Error::new(Status::GenericFailure, "VM actor already stopped"))?; }
    Ok(())
}

// Kept crate-wide on purpose: the SQLite repositories own a `rusqlite::Connection`
// (Send, !Sync) and are shared as `Arc<dyn …>` on a single-threaded LocalSet.
// Removing this flags ~20 sites across the app/infra layers; the real fix (DB
// actor / spawn_blocking, dropping the unsound `unsafe impl Send/Sync` on
// NeurolitheApp) is the Phase 3 storage-runtime redesign (PLAN 3.3, ARC-5).
#![allow(clippy::arc_with_non_send_sync)]
pub mod application;
pub mod daemon;
pub mod domain;
pub mod infrastructure;
pub mod interfaces;

fn main() {
    match crate::interfaces::cli::main_entry() {
        Ok(code) => std::process::exit(code),
        Err(e) => {
            eprintln!("Error: {e:#}");
            std::process::exit(1);
        }
    }
}

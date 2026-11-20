fn main() -> std::process::ExitCode {
    match tts_with_rvc_rs::worker::run_native(std::env::args().skip(1)) {
        Ok(()) => std::process::ExitCode::SUCCESS,
        Err(error) => {
            eprintln!("RVC worker: {error}");
            std::process::ExitCode::FAILURE
        }
    }
}

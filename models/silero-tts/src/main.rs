use std::io::{self, BufReader, BufWriter};

fn main() {
    let args: Vec<_> = std::env::args().skip(1).collect();
    match args.as_slice() {
        [s, rest @ ..] if s == "--ipc-stdio" => {
            if let Err(error) = silero_tts::ipc::run(rest) {
                eprintln!("Silero IPC: {error}");
                std::process::exit(1);
            }
        }
        [] => serve(),
        [s] if s == "--stdio" => serve(),
        [s] if s == "--capabilities" => println!("{}", silero_tts::worker::capabilities()),
        [s] if s == "--help" || s == "-h" => {
            println!("silero-tts --ipc-stdio --backend flex --checkpoint ABSOLUTE_PATH --input-dir ABSOLUTE_DIR --output-dir ABSOLUTE_DIR\n       silero-tts [--stdio | --capabilities | --help]\n\nJSONL inference worker; commands are read from stdin and responses written to stdout.");
        }
        _ => {
            eprintln!("Unknown argument. Use --help to list available modes.");
            std::process::exit(2);
        }
    }
}

fn serve() {
    let stdin = io::stdin();
    let stdout = io::stdout();
    if let Err(error) = silero_tts::worker::serve(BufReader::new(stdin.lock()), BufWriter::new(stdout.lock())) {
        eprintln!("Silero worker I/O failure: {error}");
        std::process::exit(1);
    }
}

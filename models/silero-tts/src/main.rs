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
        [s] if s == "--catalogue" => print_catalogue(None),
        [s, language, model] if s == "--model-info" => print_catalogue(Some((language, model))),
        [s] if s == "--capabilities" => println!("{}", silero_tts::worker::capabilities()),
        [s] if s == "--help" || s == "-h" => {
            println!("silero-tts --ipc-stdio --backend flex --checkpoint ABSOLUTE_PATH --input-dir ABSOLUTE_DIR --output-dir ABSOLUTE_DIR\n       silero-tts [--stdio | --capabilities | --catalogue | --model-info LANGUAGE MODEL | --help]\n\nJSONL inference worker; commands are read from stdin and responses written to stdout.");
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

fn print_catalogue(selection: Option<(&str,&str)>) {
    use silero_tts::catalogue::Catalogue;
    let result = (|| -> anyhow::Result<serde_json::Value> {
        let c = Catalogue::embedded()?;
        let verified: Vec<serde_json::Value> = serde_json::from_str(include_str!("../verified_models.json"))?;
        let model_value = |m: &silero_tts::catalogue::ModelEntry| -> anyhow::Result<serde_json::Value> {
            let mut value = serde_json::to_value(m)?;
            value["verified_checkpoint"] = serde_json::Value::Bool(verified.iter().any(|entry| entry["model_id"] == m.model_id && entry["checkpoint"]["url"] == m.url));
            Ok(value)
        };
        match selection {
            Some((language, model)) => model_value(c.resolve(language, model)?),
            None => Ok(serde_json::json!({"source": c.source, "source_sha256": c.source_sha256,
                "models": c.models.iter().map(model_value).collect::<anyhow::Result<Vec<_>>>()?})),
        }
    })();
    match result { Ok(value) => println!("{value}"), Err(error) => { eprintln!("Silero catalogue: {error:#}"); std::process::exit(1); } }
}

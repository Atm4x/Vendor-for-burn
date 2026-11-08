#![warn(missing_docs)]
#![cfg_attr(docsrs, feature(doc_cfg))]

//! Import and export ONNX models with [Burn](https://burn.dev).
//!
//! `burn-onnx` turns an ONNX model into ordinary Burn code: a Rust source file
//! that defines a `Model` struct with a typed `forward` method, plus a `.bpk`
//! (Burnpack) file holding its weights. The generated model has no ONNX runtime
//! dependency and runs on every Burn backend, from WebAssembly and `no_std`
//! microcontrollers to CUDA, Metal, and Vulkan. Going the other way, the
//! experimental [`export`][export-mod] module captures a Burn module's forward pass and
//! writes it out as an ONNX file.
//!
//! # Importing a model
//!
//! Conversion normally runs in a build script, so the generated code stays in
//! sync with the `.onnx` file. Add the dependencies to `Cargo.toml`. The
//! generated code uses `burn-store` to load its weights, so your crate needs it
//! too, and `burn` needs at least one backend feature enabled:
//!
//! ```toml
//! [dependencies]
//! burn = { version = "0.22", features = ["flex"] }
//! burn-store = "0.22"
//!
//! [build-dependencies]
//! burn-onnx = "0.22"
//! ```
//!
//! Generate the code in `build.rs`:
//!
//! ```no_run
//! # #[cfg(feature = "import")] {
//! use burn_onnx::ModelGen;
//!
//! ModelGen::new()
//!     .input("src/model/my_model.onnx")
//!     .out_dir("model/")
//!     .run_from_script();
//! # }
//! ```
//!
//! Include it from your crate (here `src/model/mod.rs`). The file is named
//! after the `.onnx` input:
//!
//! ```ignore
//! pub mod my_model {
//!     include!(concat!(env!("OUT_DIR"), "/model/my_model.rs"));
//! }
//! ```
//!
//! And run it:
//!
//! ```ignore
//! use burn::tensor::{Device, Tensor};
//! use crate::model::my_model::Model;
//!
//! let device = Device::default();
//! let model = Model::default(); // loads the .bpk written by the build script
//! let input = Tensor::<4>::zeros([1, 3, 224, 224], &device);
//! let output = model.forward(input);
//! ```
//!
//! ## Loading weights
//!
//! Which constructors the generated `Model` gets depends on the
//! [`LoadStrategy`][load-strategy] chosen at build time:
//!
//! | Constructor                         | Strategies                  | Loads weights from                         |
//! | ----------------------------------- | --------------------------- | ------------------------------------------ |
//! | `Model::default()`                  | `File`, `Embedded`          | the build-time `.bpk` path, or the binary  |
//! | `Model::from_file(path, &device)`   | `File`                      | a `.bpk` file at runtime                   |
//! | `Model::from_embedded(&device)`     | `Embedded`                  | bytes compiled into the binary             |
//! | `Model::from_bytes(bytes, &device)` | `File`, `Embedded`, `Bytes` | in-memory `.bpk` contents (`burn::tensor::Bytes`) |
//!
//! `Model::default()` uses the default device and, with `LoadStrategy::File`,
//! the `.bpk` path recorded at generation time: an absolute path inside
//! `OUT_DIR` for [`ModelGen::run_from_script`][run-from-script], or the
//! `out_dir` exactly as given for `run_from_cli` and `onnx2burn`, where a
//! relative path resolves against the working directory at runtime. That is
//! fine for development; ship the `.bpk` alongside your binary and call
//! `from_file` (or use `Embedded`) for anything you distribute.
//!
//! `Model::new(&device)` only builds the module structure: layers get fresh
//! random parameters and ONNX constants read as zeros. Never run a model built
//! with `new` alone; use one of the loading constructors above.
//!
//! ## Command line
//!
//! The `onnx2burn` binary runs the same conversion outside a build script, which
//! is handy for reading or hand-editing the generated code:
//!
//! ```sh
//! cargo install burn-onnx
//! onnx2burn model.onnx ./out
//! ```
//!
//! # Operator coverage and extensions
//!
//! The import pipeline is built on [`onnx-ir`](https://docs.rs/onnx-ir), which
//! parses and type-checks the graph and handles every opset from 1 to 24. The
//! [supported operators table] lists what converts out of the box. Anything
//! else (vendor domains such as `com.microsoft`, custom exporter ops, or
//! operators not implemented yet) can be supplied by registering a
//! [`CustomOp`][custom-op]; [`OpOverride`][op-override] replaces the code
//! generated for a built-in operator. Everything a hook needs lives in [`ext`][ext-mod].
//!
//! # Feature flags
//!
//! | Feature  | Default | Description                                                      |
//! | -------- | :-----: | ---------------------------------------------------------------- |
//! | `import` | yes     | ONNX to Burn code generation ([`ModelGen`][model-gen]) and the `onnx2burn` CLI |
//! | `mmap`   | yes     | Memory-map `.onnx` files while parsing instead of reading them in |
//! | `export` | no      | Burn to ONNX export ([`OnnxExporter`][exporter])                 |
//!
//! [export-mod]: https://docs.rs/burn-onnx/latest/burn_onnx/export/index.html
//! [exporter]: https://docs.rs/burn-onnx/latest/burn_onnx/export/struct.OnnxExporter.html
//! [supported operators table]: https://github.com/tracel-ai/burn-onnx/blob/main/SUPPORTED-ONNX-OPS.md
// The import API exists only with the `import` feature; link to docs.rs without it.
#![cfg_attr(
    feature = "import",
    doc = "[model-gen]: crate::ModelGen
[run-from-script]: crate::ModelGen::run_from_script
[load-strategy]: crate::LoadStrategy
[ext-mod]: crate::ext
[custom-op]: crate::ext::CustomOp
[op-override]: crate::ext::OpOverride"
)]
#![cfg_attr(
    not(feature = "import"),
    doc = "[model-gen]: https://docs.rs/burn-onnx/latest/burn_onnx/struct.ModelGen.html
[run-from-script]: https://docs.rs/burn-onnx/latest/burn_onnx/struct.ModelGen.html#method.run_from_script
[load-strategy]: https://docs.rs/burn-onnx/latest/burn_onnx/enum.LoadStrategy.html
[ext-mod]: https://docs.rs/burn-onnx/latest/burn_onnx/ext/index.html
[custom-op]: https://docs.rs/burn-onnx/latest/burn_onnx/ext/trait.CustomOp.html
[op-override]: https://docs.rs/burn-onnx/latest/burn_onnx/ext/trait.OpOverride.html"
)]

#[cfg(feature = "import")]
#[macro_use]
extern crate derive_new;

/// ONNX-to-Burn import and code generation.
#[cfg(feature = "import")]
#[cfg_attr(docsrs, doc(cfg(feature = "import")))]
pub mod import;

#[cfg(feature = "import")]
pub use import::*;

#[cfg(feature = "export")]
#[cfg_attr(docsrs, doc(cfg(feature = "export")))]
pub mod export;

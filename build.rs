fn main() {
    // Build proto files
    println!("cargo:rerun-if-changed=proto/federated_ai.proto");
    prost_build::Config::new()
        .out_dir("src/")
        .compile_protos(&["proto/federated_ai.proto"], &["proto/"])
        .expect("Failed to compile proto files");
}

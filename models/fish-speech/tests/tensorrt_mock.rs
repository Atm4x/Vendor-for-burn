#![cfg(feature = "tensorrt-mock")]

#[test]
fn creates_mock_network() -> trtx::Result<()> {
    let logger = trtx::Logger::stderr()?;
    let mut builder = trtx::Builder::new(&logger)?;
    builder.create_network(0)?;
    Ok(())
}

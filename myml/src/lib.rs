// src/lib.rs
use pyo3::prelude::*;

pub mod activation;
pub mod matrix;
pub mod network;

#[pymodule]
fn myml(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<network::Network>()?;
    m.add("SIGMOID", activation::ActivationId::Sigmoid as u8)?;
    m.add("TANH",    activation::ActivationId::Tanh    as u8)?;
    m.add("RELU",    activation::ActivationId::Relu    as u8)?;
    Ok(())
}

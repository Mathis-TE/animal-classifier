use pyo3::prelude::*;

pub mod activation;
pub mod matrix;
pub mod network;

#[pymodule]
fn myml_core(_py: Python, m: &Bound<'_, PyModule>) -> PyResult<()> {

    m.add_class::<network::Network>()?;

    m.add("SIGMOID", activation::ActivationKind::Sigmoid as u8)?;
    m.add("TANH",    activation::ActivationKind::Tanh    as u8)?;
    m.add("RELU",    activation::ActivationKind::Relu    as u8)?;
    Ok(())
}

use pyo3::prelude::*;
use pyo3::exceptions::PyValueError;

use std::vec;

use super::activation::{Activation, ActivationId, SIGMOID, TANH, RELU};
use super::matrix::Matrix;

#[pyclass]
pub struct Network {
    layers: Vec<usize>,
    weights: Vec<Matrix>,
    biases: Vec<Matrix>,
    data: Vec<Matrix>,
    learning_rate: f64,
    activation: Activation,
    //on choisit la fonction d'activation (Sigmoid, tanh ou relu)
    activation_id: ActivationId,
}
#[pymethods]
impl Network {

    #[new]
    pub fn new(layers: Vec<usize>, learning_rate: f64, activation_id: Option<u8>) -> Network {
        if layers.len() < 2 {
            panic!("Network must have at least 2 layers");
        }

        let id = match activation_id.unwrap_or(0){
            0 => ActivationId::Sigmoid,
            1 => ActivationId::Tanh,
            2 => ActivationId::Relu,
            _ => panic!("Invalid activation id"),
        };
        let act = match id {
            ActivationId::Sigmoid => SIGMOID,
            ActivationId::Tanh => TANH,
            ActivationId::Relu => RELU,
        };

        let mut weights = vec![];
        let mut biases = vec![];

        // Initialisation des poids et des biais
        for i in 0..layers.len()-1{
            weights.push(Matrix::random(layers[i+1], layers[i]));
            biases.push(Matrix::random(layers[i+1], 1));
        }

        Network {
            layers,
            weights,
            biases,
            data: vec![],
            learning_rate,
            activation: act,
            activation_id: id,
        }
    }

    /// Propagation avant
    pub fn feed_forward(&mut self, inputs: Vec<f64>) -> Vec<f64>{
        if inputs.len() != self.layers[0]{
            panic!("Incompatible input dimensions");
        }
        let mut current = Matrix::from(vec![inputs]).transpose();
        self.data = vec![current.clone()];

        for i in 0..self.layers.len()-1{
            current = self.weights[i]
            .multiply(&current)
            .add(&self.biases[i])
            .map(&self.activation.function);
        self.data.push(current.clone());
        }
    current.data.iter().map(|row| row[0]).collect()
    }


    pub fn train(&mut self, inputs: Vec<Vec<f64>>, targets: Vec<Vec<f64>>, epochs: u16) -> Vec<f64> {
        if (inputs.len() != targets.len()) || (inputs.is_empty()) {
            panic!("Inputs and targets must have the same length and should not be empty");
        }

        let n_inputs = inputs.len() as f64;
        let n_targets = targets[0].len() as f64;
        let mut history = Vec::with_capacity(epochs as usize);

        let step = (epochs / 100).max(1);

        for e in 1..=epochs {
            let mut sum_squared_errors = 0.0;

            for j in 0..inputs.len() {
                let outputs = self.feed_forward(inputs[j].clone());

                for (out, targ) in outputs.iter().zip(targets[j].iter()) {
                    let diff = targ - out;
                    sum_squared_errors += diff * diff;
                }
                self.back_propagate(outputs, targets[j].clone());
            }

            let mse = sum_squared_errors / (n_inputs * n_targets);
            history.push(mse);

            if epochs <= 100 || e % step == 0 {
                println!("Epoch {e} / {epochs} - MSE: {mse:.6}");
                continue;
            }
        }

        history
    }


    /// Retourne les dimensions des couches
    pub fn layers(&self) -> Vec<usize>{
        self.layers.clone()
    }

    /// Retourne les dimensions des poids
    pub fn weights_shapes(&self) -> Vec<(usize, usize)> {
        self.weights.iter().map(|w| (w.rows, w.cols)).collect()
    }

    /// Retourne les dimensions des biais
    pub fn biases_shapes(&self) -> Vec<(usize, usize)> {
        self.biases.iter().map(|b| (b.rows, b.cols)).collect()
    }

    /// Retourne l'ID de l'activation pour ne pas mettre en string
    pub fn activation_id(&self) -> u8 {
        self.activation_id as u8
    }

    pub fn set_activation_id(&mut self, id: u8) -> PyResult<()> {
        self.activation_id = match id {
            0 => ActivationId::Sigmoid,
            1 => ActivationId::Tanh,
            2 => ActivationId::Relu,
            _ => return Err(PyValueError::new_err("Invalid activation id")),
        };
        self.activation = match self.activation_id {
            ActivationId::Sigmoid => SIGMOID,
            ActivationId::Tanh    => TANH,
            ActivationId::Relu    => RELU,
        };
        Ok(())
    
    }

    pub fn get_state(&self) -> (
        Vec<usize>,             //layers
        f64,                    //leaning_rate
        u8,                     //activation id
        Vec<Vec<Vec<f64>>>,     //weights
        Vec<Vec<Vec<f64>>>,     //biases

    ){
        let w: Vec<Vec<Vec<f64>>> = self.weights.iter().map(|m| m.data.clone()).collect();
        let b: Vec<Vec<Vec<f64>>> = self.biases.iter().map(|m| m.data.clone()).collect();

        (
            self.layers.clone(),
            self.learning_rate,
            self.activation_id as u8,
            w,
            b,
        )
    }
    #[staticmethod]
    pub fn from_state(
        layers: Vec<usize>,
        learning_rate: f64,
        activation_id: u8,
        weights: Vec<Vec<Vec<f64>>>,
        biases: Vec<Vec<Vec<f64>>>,
    ) -> PyResult<Self> {
        if layers.len() < 2 {
            return Err(PyValueError::new_err("layers must have at least 2 entries"));
        }
        if weights.len() != layers.len()-1 || biases.len() != layers.len()-1 {
            return Err(PyValueError::new_err("weights/biases length mismatch with layers"));
        }

        
        let (id, act) = match activation_id {
            0 => (ActivationId::Sigmoid, SIGMOID),
            1 => (ActivationId::Tanh,    TANH),
            2 => (ActivationId::Relu,    RELU),
            _ => return Err(PyValueError::new_err("Invalid activation")),
        };

        let mut w_mat = Vec::with_capacity(weights.len());
        let mut b_mat = Vec::with_capacity(biases.len());
        for i in 0..weights.len() {
            let wi = &weights[i];
            if wi.len() != layers[i+1] || wi[0].len() != layers[i] {
                return Err(PyValueError::new_err(format!("weight[{i}] has wrong shape (expected {}x{})", layers[i+1], layers[i])));
            }
            w_mat.push(Matrix::from(wi.clone()));

            let bi = &biases[i];
            if bi.len() != layers[i+1] || bi[0].len() != 1 {
                return Err(PyValueError::new_err(format!("bias[{i}] has wrong shape (expected {}x1)", layers[i+1])));
            }
            b_mat.push(Matrix::from(bi.clone()));
        }

        Ok(Self {
            layers,
            weights: w_mat,
            biases: b_mat,
            data: vec![],
            learning_rate,
            activation: act,
            activation_id: id,
        })
    }

}

impl Network {
    /// Rétropropagation
    pub fn back_propagate(&mut self, outputs: Vec<f64>, targets: Vec<f64>){
        if targets.len() != self.layers[self.layers.len()-1]{
            panic!("Incompatible target dimensions");
        }

        let parsed = Matrix::from(vec![outputs]).transpose();
        let targets = Matrix::from(vec![targets]).transpose();

        let errors = targets.subtract(&parsed);
        let derivative_act = parsed.map(&self.activation.derivative);
        let mut delta = derivative_act.dot_multiply(&errors);

        for i in (0..self.layers.len()-1).rev(){
            let grad_w = delta
                        .multiply(&self.data[i].transpose());


            let prop_errors = self.weights[i].transpose().multiply(&delta);

            self.weights[i] = self.weights[i]
                                .add(&grad_w
                                .map(&|x| x * self.learning_rate
                                ));

            self.biases[i]  = self.biases[i]
                                .add(&delta
                                .map(&|x| x * self.learning_rate
                                ));

            if i > 0 {
                let dact_prev = self.data[i].map(&self.activation.derivative);
                delta = dact_prev.dot_multiply(&prop_errors);   
            }

        }
    }
    
}
use pyo3::prelude::*;
use pyo3::exceptions::PyValueError;

use std::vec;

use super::activation::{Activation, ActivationKind, SIGMOID, TANH, RELU};
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
    activation_kind: ActivationKind,
}
#[pymethods]
impl Network {
    //retourne un type générique
    #[new]
    pub fn new(layers: Vec<usize>, learning_rate: f64, activation_kind: Option<u8>) -> Network {
        if layers.len() < 2 {
            panic!("Network must have at least 2 layers");
        }

        let kind = match activation_kind.unwrap_or(0){
            0 => ActivationKind::Sigmoid,
            1 => ActivationKind::Tanh,
            2 => ActivationKind::Relu,
            _ => panic!("Invalid activation kind"),
        };
        let act = match kind {
            ActivationKind::Sigmoid => SIGMOID,
            ActivationKind::Tanh => TANH,
            ActivationKind::Relu => RELU,
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
            activation_kind: kind,
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



    pub fn train(&mut self, inputs: Vec<Vec<f64>>, targets: Vec<Vec<f64>>, epochs: u16){
        for i in 1..=epochs{
            if epochs < 100 || i % (epochs / 100) == 0{
                println!("Epoch {} of {}", i, epochs);
            }
            for j in 0..inputs.len(){
                let outputs = self.feed_forward(inputs[j].clone());
                self.back_propagate(outputs, targets[j].clone());
            }
        }
    }

    //MSE
// Dans #[pymethods] impl Network
pub fn train_return_mse(&mut self, inputs: Vec<Vec<f64>>, targets: Vec<Vec<f64>>, epochs: u16) -> Vec<f64> {
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
    pub fn activation_kind(&self) -> u8 {
        self.activation_kind as u8
    }

    pub fn set_activation_kind(&mut self, id: u8) -> PyResult<()> {
        self.activation_kind = match id {
            0 => ActivationKind::Sigmoid,
            1 => ActivationKind::Tanh,
            2 => ActivationKind::Relu,
            _ => return Err(PyValueError::new_err("Invalid activation kind")),
        };
        self.activation = match self.activation_kind {
            ActivationKind::Sigmoid => SIGMOID,
            ActivationKind::Tanh    => TANH,
            ActivationKind::Relu    => RELU,
        };
        Ok(())
    
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

        let mut errors = targets.subtract(&parsed);
        let mut gradients = parsed.map(&self.activation.derivative);

        for i in (0..self.layers.len()-1).rev(){
            gradients = gradients
                            .dot_multiply(&errors)
                            .map(&|x| x * self.learning_rate);

            self.weights[i] = self.weights[i]
                                .add(&gradients
                                .multiply(&self.data[i].transpose()));

            self.biases[i] = self.biases[i].add(&gradients);
            //on va multiplier les erreurs avec les poids 
            errors = self.weights[i].transpose().multiply(&errors);
            gradients = self.data[i].map(&self.activation.derivative);

        }
    }
    
}
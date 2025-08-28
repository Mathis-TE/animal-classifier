use std::f64::consts::E;

#[derive(Clone,Copy)]
pub struct Activation {
    pub function: fn(f64) -> f64,
    pub derivative: fn(f64) -> f64,
}

#[repr(u8)]
#[derive(Clone,Copy)]
pub enum ActivationKind {
    Sigmoid = 0,
    Tanh = 1,
    Relu = 2,
}

#[inline]
fn sigmoid(x:f64) -> f64 {
    1.0 / (1.0 + E.powf(-x))
}

#[inline]
fn d_sigmoid(x:f64) -> f64 {
    x * (1.0 - x)
}

#[inline]
pub fn tanh(x:f64) -> f64 {
    x.tanh()
}

#[inline]
fn d_tanh(x:f64) -> f64 {
    1.0 - x.tanh().powi(2)
}

#[inline]
fn relu(x:f64) -> f64 {
    if x > 0.0 { x } else { 0.0 }
}

#[inline]
fn d_relu(x:f64) -> f64 {
    if x > 0.0 { 1.0 } else { 0.0 }
}


pub const SIGMOID: Activation = Activation {
    function: sigmoid,
    derivative: d_sigmoid,
};

pub const TANH: Activation = Activation {
    function: tanh,
    derivative: d_tanh,
};

pub const RELU: Activation = Activation {
    function: relu,
    derivative: d_relu,
};

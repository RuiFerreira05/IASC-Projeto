use rand::random_range;

use crate::phi::{Phi};

pub struct Neuronio {
  pub w: Vec<f64>,
  pub b: f64,
  pub phi: Phi
}

impl Neuronio {
  pub fn iniciar(d: usize, phi: Phi) -> Self {
    Neuronio {
      w: (0..d).map(|_| random_range(-1.0..=1.0)).collect(),
      b: random_range(-1.0..=1.0),
      phi
    }
  }

  pub fn propagar(&self, x: &[f64]) -> f64 {
    assert_eq!(x.len(), self.w.len(), "Input size must match weights (Expected input size: {}. Given input Size: {})", self.w.len(), x.len());

    let dot: f64 = self.w.iter().zip(x).map(|(w_i, x_i)| w_i * x_i).sum();
    let h: f64 = dot + self.b;

    self.phi.aplicar(h)
  }
}

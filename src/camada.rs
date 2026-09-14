use crate::{neuronio::Neuronio, phi::{self, Phi}};

pub trait Camada {
  fn propagar(&self, x: &[f64]) -> Vec<f64>;
}

pub struct CamadaEntrada {}

impl CamadaEntrada {
  pub fn iniciar(_ds: usize) -> Self {
    CamadaEntrada {}
  }
}

impl Camada for CamadaEntrada {
  fn propagar(&self, x: &[f64]) -> Vec<f64> {
    x.to_vec()
  }
}

pub struct CamadaDensa {
  neuronios: Vec<Neuronio>
}

impl CamadaDensa {
  pub fn iniciar(de: usize, ds: usize, phi: Phi) -> Self {
    CamadaDensa {
      neuronios: (0..ds).map(|_| Neuronio::iniciar(de, phi)).collect()
    }
  }
}

impl Camada for CamadaDensa {
  fn propagar(&self, x: &[f64]) -> Vec<f64> {
    self.neuronios.iter().map(|neuronio| neuronio.propagar(x)).collect()
  }
}

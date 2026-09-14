pub enum Phi {
  Degrau,
  Tan
}

impl Phi {
  pub fn aplicar(&self, h: f64) -> f64 {
    match self {
        Phi::Degrau => {
          if h >= 0.0 {
            1.0
          } else {
            0.0
          }
        },
        Phi::Tan => {
          h.tanh()
        },
    }
  }
}

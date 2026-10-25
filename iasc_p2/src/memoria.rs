use crate::ambiente::{Acao, Estado};
use std::collections::HashMap;

#[allow(non_snake_case)]
pub struct Memoria {
    valor_omissao: f32,
    Q: HashMap<(Estado, Acao), f32>,
}

impl Memoria {
    pub fn iniciar(valor_omissao: f32) -> Self {
        Self {
            valor_omissao,
            Q: HashMap::new(),
        }
    }

    pub fn iniciar_default() -> Self {
        Self::iniciar(0.0)
    }

    pub fn q(&self, s: &Estado, a: Acao) -> f32 {
        *self.Q.get(&(s.clone(), a)).unwrap_or(&self.valor_omissao)
    }

    pub fn actualizar(&mut self, s: Estado, a: Acao, q: f32) {
        self.Q.insert((s, a), q);
    }
}

use std::collections::HashMap;

#[derive(PartialEq, Eq, Hash)]
struct Estado {}

#[derive(PartialEq, Eq, Hash)]
struct Acao {}

struct Memoria {
    valor_omissao: f32,
    Q: HashMap<(Estado, Acao), f32>,
}

impl Default for Memoria {
    fn default() -> Self {
        Self {
            valor_omissao: 0.0,
            Q: HashMap::new(),
        }
    }
}

impl Memoria {
    pub fn iniciar(valor_emissao: f32) -> Self {
        Self {
            valor_omissao: valor_emissao,
            Q: HashMap::new(),
        }
    }

    pub fn q(&self, s: Estado, a: Acao) -> f32 {
        *self.Q.get(&(s, a)).unwrap_or(&self.valor_omissao)
    }
}

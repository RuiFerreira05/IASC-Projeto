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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_valor_omissao() {
        let mem = Memoria::iniciar(5.0);
        let s = Estado::from("s0");
        let a = Acao(0);
        assert_eq!(mem.q(&s, a), 5.0);
    }

    #[test]
    fn test_iniciar_default() {
        let mem = Memoria::iniciar_default();
        let s = Estado::from("s0");
        let a = Acao(0);
        assert_eq!(mem.q(&s, a), 0.0);
    }

    #[test]
    fn test_actualizar_e_obter() {
        let mut mem = Memoria::iniciar_default();
        let s = Estado::from("s1");
        let a = Acao(1);

        mem.actualizar(s.clone(), a, 2.5);
        assert_eq!(mem.q(&s, a), 2.5);

        // Outro par deve continuar com o valor por omissão
        let a2 = Acao(2);
        assert_eq!(mem.q(&s, a2), 0.0);
    }

    #[test]
    fn test_sobrescrever_valor() {
        let mut mem = Memoria::iniciar_default();
        let s = Estado::from("s1");
        let a = Acao(0);

        mem.actualizar(s.clone(), a, 1.0);
        assert_eq!(mem.q(&s, a), 1.0);

        mem.actualizar(s.clone(), a, 3.5);
        assert_eq!(mem.q(&s, a), 3.5);
    }
}

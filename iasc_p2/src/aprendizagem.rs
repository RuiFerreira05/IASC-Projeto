use crate::{
    ambiente::{Acao, Estado},
    memoria::Memoria,
};

pub struct Sarsa {
    alfa: f32,
    gama: f32,
}

impl Sarsa {
    pub fn iniciar(alfa: f32, gama: f32) -> Self {
        Self { alfa, gama }
    }

    pub fn aprender(
        &self,
        s: Estado,
        a: Acao,
        r: f32,
        s_linha: Estado,
        a_linha: Acao,
        r#final: bool,
        memoria: &mut Memoria,
    ) {
        let j = if r#final {
            r
        } else {
            r + self.gama * memoria.q(&s_linha, a_linha)
        };

        let q = memoria.q(&s, a) + self.alfa * (j - memoria.q(&s, a));
        memoria.actualizar(s, a, q);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_aprender_passo_nao_terminal() {
        let sarsa = Sarsa::iniciar(0.5, 0.5);
        let mut mem = Memoria::iniciar_default();

        let s0 = Estado::from("s0");
        let a0 = Acao(0);
        let s1 = Estado::from("s1");
        let a1 = Acao(1);

        // Q(s1, a1) = 4.0
        mem.actualizar(s1.clone(), a1, 4.0);

        // r = 2.0, final = false
        // J = 2.0 + 0.5 * 4.0 = 4.0
        // Q(s0, a0) = 0.0 + 0.5 * (4.0 - 0.0) = 2.0
        sarsa.aprender(s0.clone(), a0, 2.0, s1, a1, false, &mut mem);

        assert!((mem.q(&s0, a0) - 2.0).abs() < 1e-5);
    }

    #[test]
    fn test_aprender_passo_terminal() {
        let sarsa = Sarsa::iniciar(0.5, 0.9);
        let mut mem = Memoria::iniciar_default();

        let s0 = Estado::from("s0");
        let a0 = Acao(0);
        let s_term = Estado::from("s_term");
        let a_dummy = Acao(0);

        // Q(s_term, a_dummy) = 100.0, mas não deve ser considerado porque é terminal
        mem.actualizar(s_term.clone(), a_dummy, 100.0);

        // r = 10.0, final = true
        // J = r = 10.0
        // Q(s0, a0) = 0.0 + 0.5 * (10.0 - 0.0) = 5.0
        sarsa.aprender(s0.clone(), a0, 10.0, s_term, a_dummy, true, &mut mem);

        assert!((mem.q(&s0, a0) - 5.0).abs() < 1e-5);
    }
}

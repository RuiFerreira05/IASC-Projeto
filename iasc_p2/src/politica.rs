use rand::{
    random_range,
    seq::{IndexedRandom, SliceRandom},
};

use crate::{
    ambiente::{Acao, Estado},
    memoria::Memoria,
};

pub struct Greedy {
    epsilon: f32,
}

impl Greedy {
    pub fn iniciar(epsilon: f32) -> Self {
        Self { epsilon }
    }

    pub fn selecionar_acao(&self, s: &Estado, accoes: &[Acao], memoria: &Memoria) -> Acao {
        if random_range(0.0..=1.0) > self.epsilon {
            Self::accao_sofrega(s, accoes, memoria)
        } else {
            Self::accao_exploracao(accoes)
        }
    }

    pub fn accao_sofrega(s: &Estado, accoes: &[Acao], memoria: &Memoria) -> Acao {
        let mut candidates = accoes.to_vec();
        candidates.shuffle(&mut rand::rng());

        let mut max = f32::NEG_INFINITY;
        let mut argmax: Acao = candidates[0];

        for a in &candidates {
            let q = memoria.q(s, *a);
            if q > max {
                max = q;
                argmax = *a;
            }
        }

        argmax
    }

    pub fn accao_exploracao(accoes: &[Acao]) -> Acao {
        let mut rng = rand::rng();
        *accoes
            .choose(&mut rng)
            .expect("A lista de acções não pode estar vazia")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_accao_sofrega_escolhe_maior_q() {
        let mut mem = Memoria::iniciar_default();
        let s = Estado::from("s0");
        let a0 = Acao(0);
        let a1 = Acao(1);
        let a2 = Acao(2);

        mem.actualizar(s.clone(), a0, 1.0);
        mem.actualizar(s.clone(), a1, 5.0);
        mem.actualizar(s.clone(), a2, 2.0);

        let accoes = vec![a0, a1, a2];
        let escolhida = Greedy::accao_sofrega(&s, &accoes, &mem);
        assert_eq!(escolhida, a1);
    }

    #[test]
    fn test_accao_sofrega_valores_negativos() {
        let mut mem = Memoria::iniciar_default();
        let s = Estado::from("s0");
        let a0 = Acao(0);
        let a1 = Acao(1);

        mem.actualizar(s.clone(), a0, -10.0);
        mem.actualizar(s.clone(), a1, -2.0);

        let accoes = vec![a0, a1];
        let escolhida = Greedy::accao_sofrega(&s, &accoes, &mem);
        assert_eq!(escolhida, a1);
    }

    #[test]
    fn test_accao_exploracao_retorna_acao_valida() {
        let a0 = Acao(0);
        let a1 = Acao(1);
        let accoes = vec![a0, a1];

        let escolhida = Greedy::accao_exploracao(&accoes);
        assert!(accoes.contains(&escolhida));
    }

    #[test]
    fn test_selecionar_acao_epsilon_zero_sempre_sofrega() {
        let mut mem = Memoria::iniciar_default();
        let s = Estado::from("s0");
        let a0 = Acao(0);
        let a1 = Acao(1);

        mem.actualizar(s.clone(), a0, 0.0);
        mem.actualizar(s.clone(), a1, 10.0);

        let politica = Greedy::iniciar(0.0);
        let accoes = vec![a0, a1];

        for _ in 0..20 {
            assert_eq!(politica.selecionar_acao(&s, &accoes, &mem), a1);
        }
    }
}

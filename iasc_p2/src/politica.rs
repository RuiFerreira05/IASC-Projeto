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

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

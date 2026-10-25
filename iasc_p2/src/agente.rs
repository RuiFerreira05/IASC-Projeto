use crate::{
    ambiente::{Acao, Ambiente, Estado},
    aprendizagem::Sarsa,
    memoria::Memoria,
    politica::Greedy,
};

pub struct Agente {
    pub memoria: Memoria,
    pub politica: Greedy,
    pub algoritmo_aprendizagem: Sarsa,
    pub accoes: Vec<Acao>,
}

impl Agente {
    pub fn iniciar(
        memoria: Memoria,
        politica: Greedy,
        algoritmo_aprendizagem: Sarsa,
        accoes: Vec<Acao>,
    ) -> Self {
        Self {
            memoria,
            politica,
            algoritmo_aprendizagem,
            accoes,
        }
    }

    pub fn selecionar_acao(&self, s: &Estado) -> Acao {
        self.politica
            .selecionar_acao(s, &self.accoes, &self.memoria)
    }

    pub fn treinar_episodio(&mut self, ambiente: &mut impl Ambiente) {
        let mut s = ambiente.iniciar();
        let mut a = self.selecionar_acao(&s);

        loop {
            let (s_linha, r, r#final) = ambiente.passo(a);
            let a_linha = self.selecionar_acao(&s_linha);

            self.algoritmo_aprendizagem.aprender(
                s,
                a,
                r,
                s_linha.clone(),
                a_linha,
                r#final,
                &mut self.memoria,
            );

            if r#final {
                break;
            }

            s = s_linha;
            a = a_linha;
        }
    }
}

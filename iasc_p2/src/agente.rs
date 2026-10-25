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

#[cfg(test)]
mod tests {
    use super::*;

    struct MockAmbienteCurto {
        estado: i32,
    }

    impl Ambiente for MockAmbienteCurto {
        fn iniciar(&mut self) -> Estado {
            self.estado = 0;
            Estado::from("0")
        }

        fn passo(&mut self, _a: Acao) -> (Estado, f32, bool) {
            self.estado += 1;
            let terminal = self.estado >= 2;
            (Estado::from(self.estado.to_string()), 1.0, terminal)
        }
    }

    #[test]
    fn test_agente_selecionar_acao() {
        let mut mem = Memoria::iniciar_default();
        let s0 = Estado::from("0");
        let a0 = Acao(0);
        let a1 = Acao(1);

        mem.actualizar(s0.clone(), a1, 10.0);

        let politica = Greedy::iniciar(0.0); // 100% guloso
        let sarsa = Sarsa::iniciar(0.1, 0.9);
        let agente = Agente::iniciar(mem, politica, sarsa, vec![a0, a1]);

        assert_eq!(agente.selecionar_acao(&s0), a1);
    }

    #[test]
    fn test_agente_treinar_episodio_atualiza_memoria() {
        let mem = Memoria::iniciar_default();
        let politica = Greedy::iniciar(0.0);
        let sarsa = Sarsa::iniciar(0.5, 0.9);
        let accoes = vec![Acao(0)];

        let mut agente = Agente::iniciar(mem, politica, sarsa, accoes);
        let mut env = MockAmbienteCurto { estado: 0 };

        agente.treinar_episodio(&mut env);

        // O ambiente executou 2 passos até terminal:
        // Estado 0 e Estado 1 foram visitados e atualizados na memória.
        let q_s0 = agente.memoria.q(&Estado::from("0"), Acao(0));
        assert!(q_s0 > 0.0);
    }
}

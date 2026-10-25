use iasc_p2::{
    agente::Agente,
    ambiente::{Acao, Ambiente, Estado},
    aprendizagem::Sarsa,
    memoria::Memoria,
    politica::Greedy,
};

struct LinhaMundo {
    pos: i32,
    meta: i32,
}

impl LinhaMundo {
    const ESQ: Acao = Acao(0);
    const DIR: Acao = Acao(1);

    fn new(meta: i32) -> Self {
        Self { pos: 0, meta }
    }
}

impl Ambiente for LinhaMundo {
    fn iniciar(&mut self) -> Estado {
        self.pos = 0;
        Estado::from(self.pos.to_string())
    }

    fn passo(&mut self, a: Acao) -> (Estado, f32, bool) {
        match a {
            Self::ESQ => self.pos = (self.pos - 1).max(0),
            Self::DIR => self.pos = (self.pos + 1).min(self.meta),
            _ => panic!("Ação desconhecida"),
        }

        let terminal = self.pos == self.meta;
        let recompensa = if terminal { 10.0 } else { -1.0 };
        (Estado::from(self.pos.to_string()), recompensa, terminal)
    }
}

#[test]
fn test_sarsa_aprendizagem() {
    let mut env = LinhaMundo::new(3);
    let accoes = vec![LinhaMundo::ESQ, LinhaMundo::DIR];

    let memoria = Memoria::iniciar_default();
    let politica = Greedy::iniciar(0.1);
    let sarsa = Sarsa::iniciar(0.5, 0.9);

    let mut agente = Agente::iniciar(memoria, politica, sarsa, accoes);

    for _ in 0..200 {
        agente.treinar_episodio(&mut env);
    }

    let q_dir_em_0 = agente.memoria.q(&Estado::from("0"), LinhaMundo::DIR);
    let q_esq_em_0 = agente.memoria.q(&Estado::from("0"), LinhaMundo::ESQ);

    assert!(
        q_dir_em_0 > q_esq_em_0,
        "O agente deve preferir mover-se para a direita (Q_dir: {}, Q_esq: {})",
        q_dir_em_0,
        q_esq_em_0
    );
}

#[test]
fn test_sarsa_aprendizagem_com_agente_default() {
    let mut env = LinhaMundo::new(3);
    let accoes = vec![LinhaMundo::ESQ, LinhaMundo::DIR];

    let mut agente = Agente::iniciar_default(accoes);

    for _ in 0..300 {
        agente.treinar_episodio(&mut env);
    }

    let q_dir_em_0 = agente.memoria.q(&Estado::from("0"), LinhaMundo::DIR);
    let q_esq_em_0 = agente.memoria.q(&Estado::from("0"), LinhaMundo::ESQ);

    assert!(
        q_dir_em_0 > q_esq_em_0,
        "O agente default deve aprender a mover-se para a direita (Q_dir: {}, Q_esq: {})",
        q_dir_em_0,
        q_esq_em_0
    );
}

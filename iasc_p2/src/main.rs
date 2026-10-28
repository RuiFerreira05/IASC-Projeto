// O estado atual deste ficheiro representa um pequeno teste para verificação da implementação e usabilidade da biblioteca, e não estará presente na ultima entrega.
// A aplicação anteriormente presente neste ficheiro foi removida devido a problemas de compatibilidade com o novo sistema modularizado de Learning e Policy no Agente.

use iasc_p2::{
    agente::Agente,
    ambiente::{Acao, Ambiente, Estado},
    aprendizagem::q_learning::QLearning,
    memoria::Memoria,
    politica::greedy::Greedy,
};

fn main() {
    let left = Acao(0);
    let right = Acao(1);
    let accoes = vec![right, left];

    let politica = Greedy::iniciar(0.1);
    // let algoritmo_aprendizagem = Sarsa::iniciar(0.5, 0.9);
    let algoritmo_aprendizagem = QLearning::iniciar(0.5, 0.9);

    let memoria = Memoria::iniciar_default();
    let mut agente = Agente::iniciar(memoria, politica, algoritmo_aprendizagem, accoes);

    let mut ambiente = LinhaAmbiente::new(5);
    for _ in 0..=1000 {
        agente.treinar_episodio(&mut ambiente);
    }

    println!("---");
    for estado_i in 0..=5 {
        for acao_i in 0..=1 {
            let estado = Estado::novo(estado_i.to_string());
            let acao = Acao(acao_i);
            let acao_s = if acao_i == 0 { "LEFT" } else { "RIGHT" };
            let q = agente.memoria.q(&estado, acao);

            println!("q({estado_i}, {acao_s}) = {q}")
        }
        println!("---")
    }
}

struct LinhaAmbiente {
    dim: u32,
    current_cell: u32,
}

impl Ambiente for LinhaAmbiente {
    fn iniciar(&mut self) -> Estado {
        self.current_cell = 0;
        Estado::novo("0".to_string())
    }

    fn passo(&mut self, a: Acao) -> (Estado, f32, bool) {
        match a {
            Acao(1) => {
                if self.current_cell < self.dim {
                    self.current_cell += 1
                }
            }
            Acao(0) => {
                if self.current_cell > 0 {
                    self.current_cell -= 1
                }
            }
            _ => (),
        }

        let estado = Estado::novo(self.current_cell.to_string());
        if self.current_cell == self.dim {
            return (estado, 10.0, true);
        } else {
            return (estado, 0.0, false);
        }
    }
}

impl LinhaAmbiente {
    fn new(dim: u32) -> Self {
        Self {
            dim,
            current_cell: 0,
        }
    }
}

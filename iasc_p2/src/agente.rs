use crate::{
    ambiente::{Acao, Ambiente, Estado},
    aprendizagem::{
        aprendizagem::{Aprendizagem, Transicao},
        sarsa::Sarsa,
    },
    memoria::Memoria,
    politica::{greedy::Greedy, politica::Politica},
};

/**
Representação do Agente de Aprendizagem por Reforço.

O agente é a entidade computacional autónoma que interage com o ambiente, aprende
comportamentos com base no reforço obtido e toma decisões sequenciais com o objetivo
de maximizar o retorno cumulativo ao longo do tempo.

Arquitetura do Agente:
- Memória de Aprendizagem (memoria): Mantém as estimativas tabulares da função de valor
  de ação Q(s, a) para os pares (estado, ação) explorados.
- Política de Decisão (politica): Estratégia de seleção de ações que gere o dilema entre
  exploração e aproveitamento (estratégia ε-greedy).
- Algoritmo de Aprendizagem (algoritmo_aprendizagem): Mecanismo de atualização por
  diferença temporal (SARSA) responsável pelo ajuste iterativo dos valores Q(s, a).
- Espaço de Ações (accoes): Conjunto finito de ações admissíveis que o agente pode executar.

Ciclo de Controlo do Algoritmo SARSA (por Episódio):
1. Iniciar o ambiente e obter o estado inicial: s = ambiente.iniciar().
2. Selecionar a ação inicial com base na política: a = selecionar_acao(s).
3. Repetir por cada passo até atingir um estado terminal:
   - Executar a ação a no ambiente, observando o novo estado s', a recompensa instantânea r
     e o indicador de terminação final.
   - Selecionar a próxima ação a' para o estado s' utilizando a mesma política: a' = selecionar_acao(s').
   - Executar o passo de aprendizagem: aprender(s, a, r, s', a', final).
   - Se o estado s' for terminal, terminar o episódio.
   - Caso contrário, transitar de estado e ação: s ← s', a ← a'.
*/
pub struct Agente<P: Politica, A: Aprendizagem> {
    /// Memória de aprendizagem tabular que armazena a função de valor Q(s, a).
    pub memoria: Memoria,

    /// Política de seleção de ação (por exemplo, ε-greedy) que orienta o comportamento do agente.
    pub politica: P,

    /// Algoritmo de aprendizagem por reforço por diferença temporal (SARSA).
    pub algoritmo_aprendizagem: A,

    /// Conjunto de ações admissíveis no ambiente.
    pub accoes: Vec<Acao>,
}

impl<P: Politica, A: Aprendizagem> Agente<P, A> {
    /**
    Inicializa um agente de aprendizagem por reforço com os componentes fornecidos.

    Parâmetros:
    - memoria: Instância da memória tabular Q(s, a).
    - politica: Instância da política de seleção de ação (ε-greedy).
    - algoritmo_aprendizagem: Instância do algoritmo SARSA.
    - accoes: Vetor de ações admissíveis no ambiente.
    */
    pub fn iniciar(
        memoria: Memoria,
        politica: P,
        algoritmo_aprendizagem: A,
        accoes: Vec<Acao>,
    ) -> Self {
        Self {
            memoria,
            politica,
            algoritmo_aprendizagem,
            accoes,
        }
    }

    /**
    Seleciona uma ação para o estado s através da política de decisão do agente e do conhecimento na memória.
    */
    pub fn selecionar_acao(&self, s: &Estado) -> Acao {
        self.politica
            .selecionar_acao(s, &self.accoes, &self.memoria)
    }

    /**
    Executa o treino completo de um episódio de interação entre o agente e o ambiente segundo o algoritmo SARSA.

    Etapas do algoritmo:
    1. Obtenção do estado inicial s através de `ambiente.iniciar()`.
    2. Seleção da primeira ação a = selecionar_acao(s) segundo a política ε-greedy.
    3. Ciclo iterativo de transições:
       - O ambiente executa o passo com a ação a, retornando (s', r, final).
       - O agente seleciona a próxima ação a' para o estado s' segundo a política ε-greedy.
       - O algoritmo SARSA atualiza a estimativa Q(s, a) na memória com a experiência (s, a, r, s', a', final).
       - Caso final seja verdadeiro, o episódio é concluído (break).
       - Caso contrário, os apontadores de estado e ação avançam: s = s', a = a'.
    */
    pub fn treinar_episodio(&mut self, ambiente: &mut impl Ambiente) {
        let mut s = ambiente.iniciar();
        let mut a = self.selecionar_acao(&s);
        loop {
            let (s_linha, r, fim) = ambiente.passo(a);
            let a_linha = self.selecionar_acao(&s_linha);
            let transicao = Transicao {
                s: &s,
                a,
                r,
                s_linha: &s_linha,
                a_linha: Some(a_linha),
                fim,
                accoes: &self.accoes,
            };
            self.algoritmo_aprendizagem
                .aprender(transicao, &mut self.memoria);
            if fim {
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

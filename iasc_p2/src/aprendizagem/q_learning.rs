use crate::{
    aprendizagem::aprendizagem::{Aprendizagem, Transicao},
    memoria::Memoria,
};

/**
Implementação do algoritmo de aprendizagem por reforço Q-Learning.

O Q-Learning é um método de controlo por Diferença Temporal (Temporal Difference - TD)
que opera em regime "Off-Policy":
- Controlo Off-Policy: Utiliza políticas distintas para comportamento e para atualização de valor.
  Enquanto a seleção de ações para execução no ambiente segue uma política com exploração
  (como ε-greedy), a estimativa do retorno futuro assume a escolha da ação ótima que maximiza
  o valor de ação no estado seguinte:
      max_{a*} Q(s', a*)
- Convergência: Permite ao agente aprender diretamente a função de valor ótima Q*(s, a) e a
  política ótima π*(s) = argmax_a Q*(s, a), mesmo enquanto explora o ambiente com ações subótimas.

Regra de Atualização:
    Q(s, a) ← Q(s, a) + α * (J - Q(s, a))

onde o retorno estimado (TD Target) J é dado por:
1. Passo com continuidade (estado s' não terminal):
   J = r + γ * max_{a* ∈ A} Q(s', a*)
2. Passo final (estado s' terminal):
   J = r
*/
pub struct QLearning {
    /// Fator de aprendizagem α ∈ [0, 1].
    alfa: f32,

    /// Fator de desconto temporal γ ∈ [0, 1].
    gama: f32,
}

impl QLearning {
    /**
    Inicializa o algoritmo Q-Learning com os fatores de aprendizagem e de desconto temporal.
    */
    pub fn iniciar(alfa: f32, gama: f32) -> Self {
        Self { alfa, gama }
    }
}

impl Aprendizagem for QLearning {
    fn aprender(&mut self, transicao: Transicao, memoria: &mut Memoria) {
        let j = if transicao.fim {
            transicao.r
        } else {
            // Obter a ação que maximiza Q(s', a*) sobre as ações admissíveis
            let max_q = transicao
                .accoes
                .iter()
                .map(|&a_cand| memoria.q(transicao.s_linha, a_cand))
                .fold(f32::NEG_INFINITY, f32::max);

            let valor_futuro = if max_q == f32::NEG_INFINITY {
                0.0
            } else {
                max_q
            };
            transicao.r + self.gama * valor_futuro
        };

        let q_antigo = memoria.q(transicao.s, transicao.a);
        let q_novo = q_antigo + self.alfa * (j - q_antigo);
        memoria.actualizar(transicao.s.clone(), transicao.a, q_novo);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ambiente::{Acao, Estado};

    #[test]
    fn test_q_learning_atualizacao_nao_terminal() {
        let mut mem = Memoria::iniciar_default();
        let s0 = Estado::from("s0");
        let s1 = Estado::from("s1");
        let a0 = Acao(0);
        let a1 = Acao(1);
        let accoes = vec![a0, a1];

        // Q(s1, a0) = 4.0, Q(s1, a1) = 10.0 => max_a Q(s1, a) = 10.0
        mem.actualizar(s1.clone(), a0, 4.0);
        mem.actualizar(s1.clone(), a1, 10.0);

        let mut ql = QLearning::iniciar(0.5, 0.9);
        let transicao = Transicao {
            s: &s0,
            a: a0,
            r: 2.0,
            s_linha: &s1,
            a_linha: None, // Off-policy: não depende de a'
            fim: false,
            accoes: &accoes,
        };

        ql.aprender(transicao, &mut mem);

        // J = 2.0 + 0.9 * 10.0 = 11.0
        // Q_novo = 0.0 + 0.5 * (11.0 - 0.0) = 5.5
        assert_eq!(mem.q(&s0, a0), 5.5);
    }

    #[test]
    fn test_q_learning_atualizacao_terminal() {
        let mut mem = Memoria::iniciar_default();
        let s0 = Estado::from("s0");
        let s_fim = Estado::from("s_fim");
        let a0 = Acao(0);
        let accoes = vec![a0];

        let mut ql = QLearning::iniciar(0.5, 0.9);
        let transicao = Transicao {
            s: &s0,
            a: a0,
            r: 10.0,
            s_linha: &s_fim,
            a_linha: None,
            fim: true,
            accoes: &accoes,
        };

        ql.aprender(transicao, &mut mem);

        // J = r = 10.0
        // Q_novo = 0.0 + 0.5 * (10.0 - 0.0) = 5.0
        assert_eq!(mem.q(&s0, a0), 5.0);
    }
}

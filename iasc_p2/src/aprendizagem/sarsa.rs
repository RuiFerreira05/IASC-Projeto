use crate::{
    aprendizagem::aprendizagem::{Aprendizagem, Transicao},
    memoria::Memoria,
};

/**
Implementação do algoritmo de aprendizagem por reforço SARSA (State-Action-Reward-State-Action).

O SARSA é um método de controlo por Diferença Temporal (Temporal Difference - TD)
que opera em regime "On-Policy":
- Controlo On-Policy: Utiliza a mesma política (por exemplo, ε-greedy) tanto para selecionar
  a ação executada no ambiente como para selecionar a próxima ação a' utilizada na estimativa
  do retorno futuro. O agente aprende o valor da política que está efetivamente a executar,
  incluindo o custo das ações exploratórias.
- Amostragem da Experiência: Cada passo de aprendizagem baseia-se na quíntupla de transição:
  s → a → r → s' → a'
  onde o agente no estado s executa a ação a, observa a recompensa r e o novo estado s',
  e escolhe a ação subsequente a' segundo a sua política atual.

Regra de Atualização por Diferença Temporal (TD):
A estimativa do valor de ação Q(s, a) é atualizada iterativamente corrigindo o valor anterior
com base no erro de estimativa (diferença temporal):
    Nova Estimativa ← Estimativa + α * (Retorno Estimado - Estimativa)
    Q(s, a) ← Q(s, a) + α * (J - Q(s, a))

onde:
- J: Retorno estimado ou valor alvo (TD Target).
- J - Q(s, a): Erro de estimativa ou erro da diferença temporal (TD Error).
- α ∈ [0, 1]: Fator ou taxa de aprendizagem (learning rate).
- γ ∈ [0, 1]: Fator de desconto temporal (discount factor).

Retorno em Passos Contínuos vs. Estados Terminais:
1. Passo com continuidade (estado s' não terminal):
   J = r + γ * Q(s', a')
   O retorno estimado combina a recompensa imediata r com o valor descontado da ação seguinte.
2. Passo final (estado s' terminal):
   J = r
   Nos estados terminais não existem ações seguintes possíveis; por conseguinte, o retorno
   futuro é nulo e o valor alvo depende exclusivamente da recompensa imediata r observada.

Propriedades dos Parâmetros:
- Fator de aprendizagem α:
  - α → 0: Conserva as estimativas anteriores, conferindo maior relevância às experiências antigas.
  - α → 1: Descarta o histórico anterior, conferindo relevância quase total à experiência mais recente.
  - Em ambientes não estacionários, um valor fixo de α permite ponderar exponencialmente as recompensas no tempo.
- Fator de desconto γ:
  - γ = 0: Agente míope, otimizando apenas a recompensa imediata r.
  - γ → 1: Agente com visão a longo prazo, atribuindo grande relevância às recompensas futuras diferidas no tempo.
*/
pub struct Sarsa {
    /// Fator de aprendizagem α ∈ [0, 1], que determina a dimensão do ajuste efetuado na estimativa Q(s, a).
    alfa: f32,

    /// Fator de desconto temporal γ ∈ [0, 1], que pondera a relevância das recompensas futuras.
    gama: f32,
}

impl Sarsa {
    /**
    Inicializa o algoritmo de aprendizagem SARSA com as taxas de aprendizagem e de desconto temporal especificadas.

    Parâmetros:
    - alfa: Fator de aprendizagem α ∈ [0, 1].
    - gama: Fator de desconto temporal γ ∈ [0, 1].
    */
    pub fn iniciar(alfa: f32, gama: f32) -> Self {
        Self { alfa, gama }
    }
}

impl Aprendizagem for Sarsa {
    fn aprender(&mut self, transicao: Transicao, memoria: &mut Memoria) {
        let j = if transicao.fim {
            transicao.r
        } else {
            let a_linha = transicao.a_linha.expect("SARSA requires next action a'");
            transicao.r + self.gama * memoria.q(transicao.s_linha, a_linha)
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
    fn test_sarsa_atualizacao_nao_terminal() {
        let mut mem = Memoria::iniciar_default();
        let s0 = Estado::from("s0");
        let s1 = Estado::from("s1");
        let a0 = Acao(0);
        let a1 = Acao(1);
        let accoes = vec![a0, a1];

        // Q(s1, a0) = 4.0, Q(s1, a1) = 10.0
        mem.actualizar(s1.clone(), a0, 4.0);
        mem.actualizar(s1.clone(), a1, 10.0);

        let mut sarsa = Sarsa::iniciar(0.5, 0.9);

        // No SARSA (On-Policy), o retorno usa a_linha selecionada (neste caso a0 = 4.0 em vez de max_a = 10.0)
        let transicao = Transicao {
            s: &s0,
            a: a0,
            r: 2.0,
            s_linha: &s1,
            a_linha: Some(a0),
            fim: false,
            accoes: &accoes,
        };

        sarsa.aprender(transicao, &mut mem);

        // J = 2.0 + 0.9 * Q(s1, a0) = 2.0 + 0.9 * 4.0 = 5.6
        // Q_novo = 0.0 + 0.5 * (5.6 - 0.0) = 2.8
        assert_eq!(mem.q(&s0, a0), 2.8);
    }

    #[test]
    fn test_sarsa_atualizacao_terminal() {
        let mut mem = Memoria::iniciar_default();
        let s0 = Estado::from("s0");
        let s_fim = Estado::from("s_fim");
        let a0 = Acao(0);
        let accoes = vec![a0];

        let mut sarsa = Sarsa::iniciar(0.5, 0.9);
        let transicao = Transicao {
            s: &s0,
            a: a0,
            r: 10.0,
            s_linha: &s_fim,
            a_linha: None,
            fim: true,
            accoes: &accoes,
        };

        sarsa.aprender(transicao, &mut mem);

        // J = r = 10.0
        // Q_novo = 0.0 + 0.5 * (10.0 - 0.0) = 5.0
        assert_eq!(mem.q(&s0, a0), 5.0);
    }
}

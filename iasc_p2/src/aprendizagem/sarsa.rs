use crate::{
    ambiente::{Acao, Estado},
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

    /**
    Inicializa o algoritmo SARSA com os hiperparâmetros predefinidos (α = 0.1 e γ = 0.9).

    Configuração padrão de referência:
    - α = 0.1: Taxa de aprendizagem moderada que assegura atualizações estáveis da função Q(s, a).
    - γ = 0.9: Fator de desconto temporal que assegura uma visão a longo prazo na consideração
      dos retornos futuros diferidos.
    */
    pub fn iniciar_default() -> Self {
        Self::iniciar(0.1, 0.9)
    }
}

impl Aprendizagem for Sarsa {
    fn aprender(&mut self, transicao: Transicao, memoria: &mut Memoria) {
        let j = if transicao.r#final {
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

// ########### TESTES ###########

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_iniciar_default() {
        let sarsa = Sarsa::iniciar_default();
        assert_eq!(sarsa.alfa, 0.1);
        assert_eq!(sarsa.gama, 0.9);
    }
}

use crate::{
    ambiente::{Acao, Estado},
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

    /**
    Executa um passo de atualização incremental da estimativa de valor Q(s, a) com base na experiência observada.

    Etapas do processamento:
    1. Cálculo do retorno estimado (valor alvo J):
       - Se o passo for terminal (final == true):
         J = r
       - Caso contrário (passo contínuo):
         J = r + γ * Q(s', a')
    2. Cálculo da nova estimativa de valor Q(s, a) por diferença temporal:
       Q(s, a) = Q(s, a) + α * (J - Q(s, a))
    3. Atualização do valor correspondente na memória de aprendizagem:
       memoria.actualizar(s, a, Q(s, a))

    Parâmetros:
    - s: Estado de partida da transição.
    - a: Ação executada no estado s.
    - r: Recompensa instantânea obtida com a transição.
    - s_linha: Novo estado s' observado após a execução da ação a.
    - a_linha: Próxima ação a' selecionada pela política no estado s_linha.
    - final: Indicador booleano de estado terminal (fim de episódio).
    - memoria: Referência mutável para a memória de aprendizagem onde reside a tabela Q(s, a).
    */
    #[allow(clippy::too_many_arguments)]
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_iniciar_default() {
        let sarsa = Sarsa::iniciar_default();
        assert_eq!(sarsa.alfa, 0.1);
        assert_eq!(sarsa.gama, 0.9);
    }

    #[test]
    fn test_aprender_passo_nao_terminal() {
        let sarsa = Sarsa::iniciar(0.5, 0.5);
        let mut mem = Memoria::iniciar_default();

        let s0 = Estado::from("s0");
        let a0 = Acao(0);
        let s1 = Estado::from("s1");
        let a1 = Acao(1);

        // Q(s1, a1) = 4.0
        mem.actualizar(s1.clone(), a1, 4.0);

        // r = 2.0, final = false
        // J = 2.0 + 0.5 * 4.0 = 4.0
        // Q(s0, a0) = 0.0 + 0.5 * (4.0 - 0.0) = 2.0
        sarsa.aprender(s0.clone(), a0, 2.0, s1, a1, false, &mut mem);

        assert!((mem.q(&s0, a0) - 2.0).abs() < 1e-5);
    }

    #[test]
    fn test_aprender_passo_terminal() {
        let sarsa = Sarsa::iniciar(0.5, 0.9);
        let mut mem = Memoria::iniciar_default();

        let s0 = Estado::from("s0");
        let a0 = Acao(0);
        let s_term = Estado::from("s_term");
        let a_dummy = Acao(0);

        // Q(s_term, a_dummy) = 100.0, mas não deve ser considerado porque é terminal
        mem.actualizar(s_term.clone(), a_dummy, 100.0);

        // r = 10.0, final = true
        // J = r = 10.0
        // Q(s0, a0) = 0.0 + 0.5 * (10.0 - 0.0) = 5.0
        sarsa.aprender(s0.clone(), a0, 10.0, s_term, a_dummy, true, &mut mem);

        assert!((mem.q(&s0, a0) - 5.0).abs() < 1e-5);
    }
}

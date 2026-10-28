use rand::{
    random_range,
    seq::{IndexedRandom, SliceRandom},
};

use crate::{
    ambiente::{Acao, Estado},
    memoria::Memoria,
    politica::politica::Politica,
};

/**
Representação da política de seleção de ação ε-greedy (epsilon-greedy).

Em Aprendizagem por Reforço, o agente depara-se constantemente com o dilema fundamental
entre exploração e aproveitamento (Exploration vs. Exploitation):
- Exploração (Exploration): O agente seleciona ações alternativas com o objetivo de descobrir
  novos estados e avaliar melhor ações cujas estimativas de recompensa sejam ainda incertas.
  Permite acumular experiência e evitar a convergência prematura para políticas subótimas.
- Aproveitamento (Exploitation): O agente seleciona a ação identificada como sendo a melhor
  segundo o conhecimento adquirido até ao momento (ação sôfrega ou greedy), com o intuito
  de maximizar o retorno imediato e futuro acumulado:
  a* = argmax_a Q(s, a)

Estratégia ε-greedy:
Para gerir este compromisso, a estratégia ε-greedy estabelece um mecanismo estocástico
regulado pelo parâmetro ε ∈ [0, 1] (probabilidade de exploração):
- Com probabilidade 1 - ε: Realiza aproveitamento do conhecimento, selecionando a ação sôfrega
  a* com maior estimativa de valor Q(s, a) (ação greedy).
- Com probabilidade ε: Realiza exploração do ambiente, selecionando uma ação aleatória
  uniformemente distribuída entre todas as ações admissíveis (at ≠ a* em geral).

Tratamento de Empates (Tie-breaking):
Quando múltiplas ações partilham o mesmo valor máximo de Q(s, a), a política baralha previamente
as ações candidatas (shuffle), assegurando um desempate aleatório não tendencioso.
*/
pub struct Greedy {
    /// Probabilidade de exploração ε ∈ [0, 1], definindo a frequência com que o agente escolhe ações aleatórias.
    epsilon: f32,
}

impl Greedy {
    /**
    Inicializa a política de seleção de ação ε-greedy com a probabilidade de exploração epsilon.

    Parâmetros:
    - epsilon: Valor real no intervalo [0, 1] que dita a probabilidade de tomar uma ação exploratória.
      - epsilon = 0.0: Política puramente sôfrega (apenas aproveitamento, sem exploração).
      - epsilon = 1.0: Política puramente aleatória (apenas exploração).
    */
    pub fn iniciar(epsilon: f32) -> Self {
        Self { epsilon }
    }

    /**
    Inicializa a política ε-greedy com o valor de exploração predefinido (ε = 0.1).

    O valor ε = 0.1 é amplamente utilizado na literatura de aprendizagem por reforço
    (Sutton & Barto, 2020) por estabelecer um compromisso equilibrado e robusto
    entre a exploração de novas ações e o aproveitamento do conhecimento já adquirido.
    */
    pub fn iniciar_default() -> Self {
        Self::iniciar(0.1)
    }

    /**
    Seleciona a ação a executar no estado s de acordo com a regra ε-greedy.

    Etapas do algoritmo de decisão:
    1. Gera um número aleatório uniforme no intervalo [0, 1].
    2. Se random() > ε:
       - Seleciona a ação sôfrega através de `accao_sofrega(s, accoes, memoria)` (aproveitamento).
    3. Caso contrário (random() <= ε):
       - Seleciona uma ação aleatória através de `accao_exploracao(accoes)` (exploração).

    Retorna a ação selecionada.
    */
    pub fn selecionar_acao(&self, s: &Estado, accoes: &[Acao], memoria: &Memoria) -> Acao {
        if random_range(0.0..=1.0) > self.epsilon {
            Self::accao_sofrega(s, accoes, memoria)
        } else {
            Self::accao_exploracao(accoes)
        }
    }

    /**
    Identifica e retorna a ação sôfrega (greedy) para o estado s: a* = argmax_a Q(s, a).

    Mecanismo de desempate:
    1. Cria uma cópia da lista de ações admissíveis e baralha-a aleatoriamente (shuffle).
    2. Itera sobre as ações baralhadas e encontra a ação que maximiza a estimativa Q(s, a).
       Desta forma, caso existam várias ações com valor idêntico ao máximo, a escolha entre elas
       é estocasticamente uniforme e isenta de enviesamento de ordem.
    */
    pub fn accao_sofrega(s: &Estado, accoes: &[Acao], memoria: &Memoria) -> Acao {
        let mut candidates = accoes.to_vec();
        candidates.shuffle(&mut rand::rng());

        let mut max = f32::NEG_INFINITY;
        let mut argmax: Acao = candidates[0];

        for a in &candidates {
            let q = memoria.q(s, *a);
            if q > max {
                max = q;
                argmax = *a;
            }
        }

        argmax
    }

    /**
    Seleciona uma ação exploratória aleatória a partir do conjunto de ações admissíveis.

    Realiza uma amostragem uniforme sobre o vetor de ações, permitindo ao agente
    obter nova experiência e descobrir transições de estados ainda inexploradas.
    */
    pub fn accao_exploracao(accoes: &[Acao]) -> Acao {
        let mut rng = rand::rng();
        *accoes
            .choose(&mut rng)
            .expect("A lista de acções não pode estar vazia")
    }
}

impl Politica for Greedy {
    fn selecionar_acao(&self, s: &Estado, accoes: &[Acao], memoria: &Memoria) -> Acao {
        self.selecionar_acao(s, accoes, memoria)
    }
}

// ########### TESTES ###########

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_iniciar_default() {
        let politica = Greedy::iniciar_default();
        assert_eq!(politica.epsilon, 0.1);
    }

    #[test]
    fn test_accao_sofrega_escolhe_maior_q() {
        let mut mem = Memoria::iniciar_default();
        let s = Estado::from("s0");
        let a0 = Acao(0);
        let a1 = Acao(1);
        let a2 = Acao(2);

        mem.actualizar(s.clone(), a0, 1.0);
        mem.actualizar(s.clone(), a1, 5.0);
        mem.actualizar(s.clone(), a2, 2.0);

        let accoes = vec![a0, a1, a2];
        let escolhida = Greedy::accao_sofrega(&s, &accoes, &mem);
        assert_eq!(escolhida, a1);
    }

    #[test]
    fn test_accao_sofrega_valores_negativos() {
        let mut mem = Memoria::iniciar_default();
        let s = Estado::from("s0");
        let a0 = Acao(0);
        let a1 = Acao(1);

        mem.actualizar(s.clone(), a0, -10.0);
        mem.actualizar(s.clone(), a1, -2.0);

        let accoes = vec![a0, a1];
        let escolhida = Greedy::accao_sofrega(&s, &accoes, &mem);
        assert_eq!(escolhida, a1);
    }

    #[test]
    fn test_accao_exploracao_retorna_acao_valida() {
        let a0 = Acao(0);
        let a1 = Acao(1);
        let accoes = vec![a0, a1];

        let escolhida = Greedy::accao_exploracao(&accoes);
        assert!(accoes.contains(&escolhida));
    }

    #[test]
    fn test_selecionar_acao_epsilon_zero_sempre_sofrega() {
        let mut mem = Memoria::iniciar_default();
        let s = Estado::from("s0");
        let a0 = Acao(0);
        let a1 = Acao(1);

        mem.actualizar(s.clone(), a0, 0.0);
        mem.actualizar(s.clone(), a1, 10.0);

        let politica = Greedy::iniciar(0.0);
        let accoes = vec![a0, a1];

        for _ in 0..20 {
            assert_eq!(politica.selecionar_acao(&s, &accoes, &mem), a1);
        }
    }
}

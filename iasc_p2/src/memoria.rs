use crate::ambiente::{Acao, Estado};
use std::collections::HashMap;

/**
Representação da memória de aprendizagem tabular da função de valor de ação Q(s, a).

Em problemas de decisão sequencial e aprendizagem por reforço, a função valor de acção
Q(s, a) expressa o retorno esperado obtido pelo agente ao realizar a ação a no estado s
e seguir a política de controlo daí em diante:
    Q(s, a) ≈ E[G(s, a)]

Fundamentação Teórica:
- Retorno Esperado: O retorno cumulativo descontado no tempo é dado por:
  G(s, a) = r + γ * G(s', a')
  sendo a função Q(s, a) a estimativa estatística desse retorno baseada nas amostras
  de experiência observadas pelo agente ao longo do tempo.
- Representação Tabular: A memória mapeia explicitamente pares discretos (estado, acção)
  para valores reais:
  Q : (S × A) → ℝ
- Inicialização e Valor por Omissão: Para pares (s, a) que ainda não foram observados,
  a memória retorna um valor predefinido (valor_omissao). Valores iniciais neutros (0.0)
  representam ausência de conhecimento a priori; valores otimistas (positivos elevados)
  podem incentivar a exploração inicial de ações não visitadas.
- Atualização Incremental: À medida que novas transições e recompensas são observadas,
  os valores de Q(s, a) são atualizados incrementalmente por algoritmos de diferença temporal
  (como o SARSA), convergindo no limite para os valores ótimos caso cada par estado-ação
  seja visitado um número suficiente de vezes.
*/
#[allow(non_snake_case)]
pub struct Memoria {
    /// Valor por omissão retornado para pares (estado, ação) que ainda não foram registados na memória.
    valor_omissao: f32,

    /// Tabela de dispersão associando cada par (Estado, Acao) à sua respetiva estimativa de valor Q(s, a).
    Q: HashMap<(Estado, Acao), f32>,
}

impl Memoria {
    /**
    Inicializa a memória de aprendizagem com um valor por omissão especificado.

    Parâmetros:
    - valor_omissao: Valor numérico devolvido na consulta de pares (s, a) ainda não atualizados.
    */
    pub fn iniciar(valor_omissao: f32) -> Self {
        Self {
            valor_omissao,
            Q: HashMap::new(),
        }
    }

    /**
    Inicializa a memória de aprendizagem com o valor por omissão neutro (0.0).

    Corresponde à inicialização padrão descrita no modelo formal da aprendizagem por reforço:
        valor_omissao = 0.0
        Q = {}
    */
    pub fn iniciar_default() -> Self {
        Self::iniciar(0.0)
    }

    /**
    Consulta a estimativa de valor de ação Q(s, a) para um determinado estado s e ação a.

    Retorna o valor registado na memória se o par (s, a) já tiver sido atualizado;
    caso contrário, retorna o valor por omissão (valor_omissao).
    */
    pub fn q(&self, s: &Estado, a: Acao) -> f32 {
        *self.Q.get(&(s.clone(), a)).unwrap_or(&self.valor_omissao)
    }

    /**
    Atualiza a estimativa de valor de ação Q(s, a) na memória de aprendizagem.

    Atribui o novo valor de retorno estimado q ao par (s, a):
        Q[(s, a)] = q
    */
    pub fn actualizar(&mut self, s: Estado, a: Acao, q: f32) {
        self.Q.insert((s, a), q);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_valor_omissao() {
        let mem = Memoria::iniciar(5.0);
        let s = Estado::from("s0");
        let a = Acao(0);
        assert_eq!(mem.q(&s, a), 5.0);
    }

    #[test]
    fn test_iniciar_default() {
        let mem = Memoria::iniciar_default();
        let s = Estado::from("s0");
        let a = Acao(0);
        assert_eq!(mem.q(&s, a), 0.0);
    }

    #[test]
    fn test_actualizar_e_obter() {
        let mut mem = Memoria::iniciar_default();
        let s = Estado::from("s1");
        let a = Acao(1);

        mem.actualizar(s.clone(), a, 2.5);
        assert_eq!(mem.q(&s, a), 2.5);

        // Outro par deve continuar com o valor por omissão
        let a2 = Acao(2);
        assert_eq!(mem.q(&s, a2), 0.0);
    }

    #[test]
    fn test_sobrescrever_valor() {
        let mut mem = Memoria::iniciar_default();
        let s = Estado::from("s1");
        let a = Acao(0);

        mem.actualizar(s.clone(), a, 1.0);
        assert_eq!(mem.q(&s, a), 1.0);

        mem.actualizar(s.clone(), a, 3.5);
        assert_eq!(mem.q(&s, a), 3.5);
    }
}

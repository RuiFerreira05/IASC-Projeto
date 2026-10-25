/**
Representação do ambiente e dos elementos fundamentais do ciclo de interação em Aprendizagem por Reforço.

No paradigma da aprendizagem por reforço, a resolução de um problema de decisão sequencial
assenta na interação contínua entre um agente decisor e o ambiente externo:
- Estado (s): Representa a configuração ou situação observada do mundo num determinado instante t.
  Num Processo de Decisão de Markov (MDP), o estado sumariza a informação relevante do histórico
  de observações anterior.
- Acção (a): Decisão tomada pelo agente no estado atual, selecionada a partir de um conjunto
  de ações admissíveis.
- Transição de Estado e Incerteza: A realização de uma ação a num estado s conduz a um novo
  estado s'. Em ambientes estocásticos (não deterministas), essa transição está sujeita a incerteza,
  sendo regida por uma distribuição de probabilidades de transição T(s, a, s') = P(s' | s, a),
  a qual é desconhecida a priori na generalidade dos problemas de reforço.
- Recompensa (r): Medida escalar de desempenho fornecida pelo ambiente imediatamente após a
  transição, segundo uma função de recompensa R(s, a, s'). O valor (utilidade) da ação pode
  não ser imediato, surgindo frequentemente de forma diferida no tempo.
- Episódios e Estados Terminais: Os problemas episódicos terminam quando o ambiente atinge um
  estado terminal (s' in S_terminal). Num estado terminal, não existem ações subsequentes a realizar,
  concluindo-se o episódio de experiência.
*/

/**
Representação de uma ação executável pelo agente no ambiente.

No contexto de decisão sequencial, as ações constituem o mecanismo através do qual o agente
influencia o estado do ambiente e desencadeia a geração de recompensas.
*/
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub struct Acao(pub usize);

impl From<usize> for Acao {
    /**
    Cria uma nova ação a partir do seu identificador numérico.
    */
    fn from(id: usize) -> Self {
        Acao(id)
    }
}

/**
Representação de um estado do ambiente.

Identifica de forma unívoca a observação ou configuração do ambiente num determinado instante,
servindo como índice na memória de aprendizagem para associar valores de utilidade às ações.
*/
#[derive(Clone, PartialEq, Eq, Hash, Debug)]
pub struct Estado(pub String);

impl Estado {
    /**
    Cria uma nova instância de estado a partir de um identificador textual.
    */
    pub fn novo(id: impl Into<String>) -> Self {
        Self(id.into())
    }
}

impl From<&str> for Estado {
    /**
    Converte uma referência de string para um estado.
    */
    fn from(s: &str) -> Self {
        Estado(s.to_string())
    }
}

impl From<String> for Estado {
    /**
    Converte uma String para um estado.
    */
    fn from(s: String) -> Self {
        Estado(s)
    }
}

/**
Interface (trait) que define o comportamento do ambiente na aprendizagem por reforço.

Modela a dinâmica do problema de decisão sequencial através do ciclo agente-ambiente:
1. Inicialização do episódio via `iniciar()`, colocando o ambiente no estado inicial s_0.
2. Execução de passos através de `passo(a)`, aplicando a ação a e retornando a transição
   composta pelo novo estado s', pela recompensa instantânea r e pelo indicador de terminação.
*/
pub trait Ambiente {
    /**
    Inicializa ou reinicia o ambiente no início de um novo episódio de aprendizagem.

    Retorna o estado inicial s_0 do ambiente.
    */
    fn iniciar(&mut self) -> Estado;

    /**
    Executa a ação especificada no ambiente, realizando uma transição de estado.

    Etapas do processamento:
    1. Aplicação da ação a sobre o estado atual do ambiente.
    2. Transição para o novo estado s' (sujeita a incerteza caso o ambiente seja estocástico).
    3. Geração da recompensa instantânea observada r.
    4. Verificação da condição de paragem (se s' corresponde a um estado terminal).

    Retorna um tuplo (s', r, final):
    - s': Novo estado observado resultante da transição.
    - r: Recompensa escalar instantânea obtida com a transição.
    - final: Booleano indicando se o novo estado s' é terminal (fim do episódio).
    */
    fn passo(&mut self, a: Acao) -> (Estado, f32, bool);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_acao_criacao_e_from() {
        let a1 = Acao(1);
        let a2 = Acao::from(1);
        assert_eq!(a1, a2);
        assert_eq!(a1.0, 1);
    }

    #[test]
    fn test_estado_criacao_e_from() {
        let e1 = Estado::novo("s0");
        let e2 = Estado::from("s0");
        let e3 = Estado::from(String::from("s0"));
        assert_eq!(e1, e2);
        assert_eq!(e2, e3);
        assert_eq!(e1.0, "s0");
    }

    struct MockAmbiente {
        passos: usize,
    }

    impl Ambiente for MockAmbiente {
        fn iniciar(&mut self) -> Estado {
            self.passos = 0;
            Estado::from("inicio")
        }

        fn passo(&mut self, _a: Acao) -> (Estado, f32, bool) {
            self.passos += 1;
            let terminal = self.passos >= 2;
            (
                Estado::from(format!("passo_{}", self.passos)),
                1.0,
                terminal,
            )
        }
    }

    #[test]
    fn test_ambiente_trait() {
        let mut mock = MockAmbiente { passos: 0 };
        let s0 = mock.iniciar();
        assert_eq!(s0, Estado::from("inicio"));

        let (s1, r1, term1) = mock.passo(Acao(0));
        assert_eq!(s1, Estado::from("passo_1"));
        assert_eq!(r1, 1.0);
        assert!(!term1);

        let (s2, r2, term2) = mock.passo(Acao(0));
        assert_eq!(s2, Estado::from("passo_2"));
        assert_eq!(r2, 1.0);
        assert!(term2);
    }
}

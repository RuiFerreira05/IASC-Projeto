use crate::{
    ambiente::{Acao, Estado},
    memoria::Memoria,
};

/**
Representa a transição de experiência de um passo no ambiente.

Contém os elementos essenciais da experiência (s, a, r, s', a', fim, accoes)
necessários para os algoritmos de aprendizagem por diferença temporal (on-policy e off-policy).
*/
pub struct Transicao<'a> {
    /// Estado de partida s.
    pub s: &'a Estado,
    /// Ação executada a.
    pub a: Acao,
    /// Recompensa instantânea observada r.
    pub r: f32,
    /// Próximo estado resultante da transição s'.
    pub s_linha: &'a Estado,
    /// Próxima ação selecionada a' (obrigatória para SARSA on-policy; opcional para Q-Learning off-policy).
    pub a_linha: Option<Acao>,
    /// Indicador de terminação do episódio (verdadeiro se s' for estado terminal).
    pub fim: bool,
    /// Conjunto de ações admissíveis no ambiente.
    pub accoes: &'a [Acao],
}

/**
Interface (trait) que formaliza um algoritmo de aprendizagem por reforço por diferença temporal.
*/
pub trait Aprendizagem {
    /**
    Executa o passo de atualização da estimativa de valor Q(s, a) na memória com base na transição observada.
    */
    fn aprender(&mut self, transicao: Transicao, memoria: &mut Memoria);
}

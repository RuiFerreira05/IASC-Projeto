use crate::{
    ambiente::{Acao, Estado},
    memoria::Memoria,
};

/**
Interface (trait) que formaliza uma política de seleção de ação.

A política define a estratégia de tomada de decisão do agente, mapeando o estado
observado e o conhecimento acumulado na memória para uma ação a executar.
*/
pub trait Politica {
    /**
    Seleciona uma ação para o estado s a partir do conjunto de ações admissíveis e da memória de valores.
    */
    fn selecionar_acao(&self, s: &Estado, accoes: &[Acao], memoria: &Memoria) -> Acao;
}

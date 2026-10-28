use crate::{
    ambiente::{Acao, Estado},
    memoria::Memoria,
};

pub struct Transicao<'a> {
    pub s: &'a Estado,
    pub a: Acao,
    pub r: f32,
    pub s_linha: &'a Estado,
    pub a_linha: Option<Acao>,
    pub fim: bool,
    pub accoes: &'a [Acao],
}

pub trait Aprendizagem {
    fn aprender(&mut self, transicao: Transicao, memoria: &mut Memoria);
}

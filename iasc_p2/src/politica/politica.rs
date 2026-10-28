use crate::{
    ambiente::{Acao, Estado},
    memoria::Memoria,
};

pub trait Politica {
    fn selecionar_acao(&self, s: &Estado, accoes: &[Acao], memoria: &Memoria) -> Acao;
}

use std::fmt::Debug;

pub trait Problema<E: Estado> {
    fn estado_inicial(&self) -> E;

    fn sucessores(&self, estado: E) -> Vec<E>;

    fn avaliar(&self, estado: E) -> f64;

    fn solucao(&self, estado: E) -> bool;
}

pub trait Estado: Debug + Copy {}

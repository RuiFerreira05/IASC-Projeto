#![cfg(test)]

use projeto::{neuronio::Neuronio, phi::Phi};

#[test]
fn criar_neuronio() {
    Neuronio::iniciar(2, Phi::Degrau);
}

#[test]
fn propagar_neuronio() {
    let n = Neuronio::iniciar(2, Phi::Degrau);

    let v_e = vec![1.0, 1.0];
    n.propagar(&v_e);
}

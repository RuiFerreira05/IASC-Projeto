use projeto::{phi::Phi, rede::RedeNeuronal};

#[test]
fn xor_test() {
    let theta: &[&[(&[f64], f64)]] = &[
        // camada 2
        &[(&[1.0, -1.0], -0.5), (&[-1.0, 1.0], -0.5)],
        // camada 3
        &[(&[1.0, 1.0], -0.5)],
    ];

    let phi = Phi::Degrau;

    let mut rede = RedeNeuronal::iniciar_manual(theta, phi);

    let tests: &[(&[f64], &[f64])] = &[
        (&[0.0, 0.0], &[0.0]),
        (&[0.0, 1.0], &[1.0]),
        (&[1.0, 0.0], &[1.0]),
        (&[1.0, 1.0], &[0.0]),
    ];

    let (x, y): (Vec<_>, Vec<_>) = tests.iter().copied().unzip();

    let y_predict = rede.prever(x.as_slice());

    assert_eq!(y_predict, y);
}

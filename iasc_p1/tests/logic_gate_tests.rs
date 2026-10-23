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

#[test]
fn and_test() {
    let theta: &[&[(&[f64], f64)]] = &[
        // camada 2 (saída)
        &[(&[1.0, 1.0], -1.5)],
    ];

    let phi = Phi::Degrau;

    let mut rede = RedeNeuronal::iniciar_manual(theta, phi);

    let tests: &[(&[f64], &[f64])] = &[
        (&[0.0, 0.0], &[0.0]),
        (&[0.0, 1.0], &[0.0]),
        (&[1.0, 0.0], &[0.0]),
        (&[1.0, 1.0], &[1.0]),
    ];

    let (x, y): (Vec<_>, Vec<_>) = tests.iter().copied().unzip();

    let y_predict = rede.prever(x.as_slice());

    assert_eq!(y_predict, y);
}

#[test]
fn or_test() {
    let theta: &[&[(&[f64], f64)]] = &[
        // camada 2 (saída)
        &[(&[1.0, 1.0], -0.5)],
    ];

    let phi = Phi::Degrau;

    let mut rede = RedeNeuronal::iniciar_manual(theta, phi);

    let tests: &[(&[f64], &[f64])] = &[
        (&[0.0, 0.0], &[0.0]),
        (&[0.0, 1.0], &[1.0]),
        (&[1.0, 0.0], &[1.0]),
        (&[1.0, 1.0], &[1.0]),
    ];

    let (x, y): (Vec<_>, Vec<_>) = tests.iter().copied().unzip();

    let y_predict = rede.prever(x.as_slice());

    assert_eq!(y_predict, y);
}

#[test]
fn nand_test() {
    let theta: &[&[(&[f64], f64)]] = &[
        // camada 2 (saída)
        &[(&[-1.0, -1.0], 1.5)],
    ];

    let phi = Phi::Degrau;

    let mut rede = RedeNeuronal::iniciar_manual(theta, phi);

    let tests: &[(&[f64], &[f64])] = &[
        (&[0.0, 0.0], &[1.0]),
        (&[0.0, 1.0], &[1.0]),
        (&[1.0, 0.0], &[1.0]),
        (&[1.0, 1.0], &[0.0]),
    ];

    let (x, y): (Vec<_>, Vec<_>) = tests.iter().copied().unzip();

    let y_predict = rede.prever(x.as_slice());

    assert_eq!(y_predict, y);
}

#[test]
fn nor_test() {
    let theta: &[&[(&[f64], f64)]] = &[
        // camada 2 (saída)
        &[(&[-1.0, -1.0], 0.5)],
    ];

    let phi = Phi::Degrau;

    let mut rede = RedeNeuronal::iniciar_manual(theta, phi);

    let tests: &[(&[f64], &[f64])] = &[
        (&[0.0, 0.0], &[1.0]),
        (&[0.0, 1.0], &[0.0]),
        (&[1.0, 0.0], &[0.0]),
        (&[1.0, 1.0], &[0.0]),
    ];

    let (x, y): (Vec<_>, Vec<_>) = tests.iter().copied().unzip();

    let y_predict = rede.prever(x.as_slice());

    assert_eq!(y_predict, y);
}

#[test]
fn not_test() {
    let theta: &[&[(&[f64], f64)]] = &[
        // camada 2 (saída)
        &[(&[-1.0], 0.5)],
    ];

    let phi = Phi::Degrau;

    let mut rede = RedeNeuronal::iniciar_manual(theta, phi);

    let tests: &[(&[f64], &[f64])] = &[(&[0.0], &[1.0]), (&[1.0], &[0.0])];

    let (x, y): (Vec<_>, Vec<_>) = tests.iter().copied().unzip();

    let y_predict = rede.prever(x.as_slice());

    assert_eq!(y_predict, y);
}

#[test]
fn xnor_test() {
    let theta: &[&[(&[f64], f64)]] = &[
        // camada 2: [NOR, AND]
        &[(&[-1.0, -1.0], 0.5), (&[1.0, 1.0], -1.5)],
        // camada 3: OR
        &[(&[1.0, 1.0], -0.5)],
    ];

    let phi = Phi::Degrau;

    let mut rede = RedeNeuronal::iniciar_manual(theta, phi);

    let tests: &[(&[f64], &[f64])] = &[
        (&[0.0, 0.0], &[1.0]),
        (&[0.0, 1.0], &[0.0]),
        (&[1.0, 0.0], &[0.0]),
        (&[1.0, 1.0], &[1.0]),
    ];

    let (x, y): (Vec<_>, Vec<_>) = tests.iter().copied().unzip();

    let y_predict = rede.prever(x.as_slice());

    assert_eq!(y_predict, y);
}

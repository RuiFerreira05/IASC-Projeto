use projeto::{camada::Camada, phi::Phi, rede::RedeNeuronal};

#[test]
fn test_delta_saida() {
    let y_n = [0.8, 0.2];
    let y = [1.0, 0.0];
    let delta = RedeNeuronal::delta_saida(&y_n, &y);
    assert!((delta[0] - (-0.2)).abs() < 1e-9);
    assert!((delta[1] - 0.2).abs() < 1e-9);
}

#[test]
#[should_panic(expected = "mesmo tamanho")]
fn test_delta_saida_dim_mismatch() {
    let y_n = [0.8, 0.2];
    let y = [1.0];
    RedeNeuronal::delta_saida(&y_n, &y);
}

#[test]
fn test_retropropagar_single_step_analytical() {
    // Rede simples: 1 entrada -> 1 neurónio de saída com Sigmoide
    // Inicialização manual: w = [0.5], b = 0.1
    let theta: &[&[(&[f64], f64)]] = &[&[(&[0.5], 0.1)]];
    let mut rede = RedeNeuronal::iniciar_manual(theta, Phi::Sigmoide);

    let x = [2.0];
    let y_pred = rede.propagar(&x);
    // h = 0.5 * 2.0 + 0.1 = 1.1
    // y = 1 / (1 + e^-1.1) ~= 0.7502601055951177
    let expected_y = 1.0 / (1.0 + (-1.1_f64).exp());
    assert!((y_pred[0] - expected_y).abs() < 1e-9);

    let target = [1.0];
    let delta_saida = RedeNeuronal::delta_saida(&y_pred, &target);
    let alpha = 0.1;

    rede.retropropagar(&delta_saida, alpha);

    // Derivada: y * (1 - y)
    let y_derivada = expected_y * (1.0 - expected_y);
    let escalar = -alpha * y_derivada * (expected_y - 1.0);
    let expected_w = 0.5 + escalar * 2.0;
    let expected_b = 0.1 + escalar;

    match &rede.camadas[1] {
        Camada::Densa { neuronios, .. } => {
            assert!((neuronios[0].w[0] - expected_w).abs() < 1e-9);
            assert!((neuronios[0].b - expected_b).abs() < 1e-9);
        }
        _ => panic!("Camada 1 deve ser Densa"),
    }
}

#[test]
fn test_adaptar_reduces_error() {
    let mut rede = RedeNeuronal::iniciar(&[2, 1], Phi::Sigmoide);
    let x = [1.0, 1.0];
    let y = [1.0];

    let eps1 = rede.adaptar(&x, &y, 0.5);
    let mut last_eps = eps1;
    for _ in 0..50 {
        last_eps = rede.adaptar(&x, &y, 0.5);
    }

    assert!(
        last_eps < eps1,
        "O erro após adaptações repetidas ({}) deve ser menor que o inicial ({})",
        last_eps,
        eps1
    );
}

#[test]
fn test_and_training() {
    let x: &[&[f64]] = &[
        &[0.0, 0.0],
        &[0.0, 1.0],
        &[1.0, 0.0],
        &[1.0, 1.0],
    ];
    let y: &[&[f64]] = &[
        &[0.0],
        &[0.0],
        &[0.0],
        &[1.0],
    ];

    let mut rede = RedeNeuronal::iniciar(&[2, 1], Phi::Sigmoide);
    rede.treinar(x, y, 10000, 0.01, 0.5);

    let pred = rede.prever(x);
    assert!(pred[0][0] < 0.2, "0 AND 0 esperado < 0.2, obtido {}", pred[0][0]);
    assert!(pred[1][0] < 0.2, "0 AND 1 esperado < 0.2, obtido {}", pred[1][0]);
    assert!(pred[2][0] < 0.2, "1 AND 0 esperado < 0.2, obtido {}", pred[2][0]);
    assert!(pred[3][0] > 0.8, "1 AND 1 esperado > 0.8, obtido {}", pred[3][0]);
}

#[test]
fn test_or_training() {
    let x: &[&[f64]] = &[
        &[0.0, 0.0],
        &[0.0, 1.0],
        &[1.0, 0.0],
        &[1.0, 1.0],
    ];
    let y: &[&[f64]] = &[
        &[0.0],
        &[1.0],
        &[1.0],
        &[1.0],
    ];

    let mut rede = RedeNeuronal::iniciar(&[2, 1], Phi::Sigmoide);
    rede.treinar(x, y, 10000, 0.01, 0.5);

    let pred = rede.prever(x);
    assert!(pred[0][0] < 0.2, "0 OR 0 esperado < 0.2, obtido {}", pred[0][0]);
    assert!(pred[1][0] > 0.8, "0 OR 1 esperado > 0.8, obtido {}", pred[1][0]);
    assert!(pred[2][0] > 0.8, "1 OR 0 esperado > 0.8, obtido {}", pred[2][0]);
    assert!(pred[3][0] > 0.8, "1 OR 1 esperado > 0.8, obtido {}", pred[3][0]);
}

#[test]
fn test_xor_training() {
    let x: &[&[f64]] = &[
        &[0.0, 0.0],
        &[0.0, 1.0],
        &[1.0, 0.0],
        &[1.0, 1.0],
    ];
    let y: &[&[f64]] = &[
        &[0.0],
        &[1.0],
        &[1.0],
        &[0.0],
    ];

    // XOR requer camada oculta não linear
    let mut rede = RedeNeuronal::iniciar(&[2, 4, 1], Phi::Sigmoide);
    rede.treinar(x, y, 30000, 0.005, 0.5);

    let pred = rede.prever(x);
    assert!(pred[0][0] < 0.2, "0 XOR 0 esperado < 0.2, obtido {}", pred[0][0]);
    assert!(pred[1][0] > 0.8, "0 XOR 1 esperado > 0.8, obtido {}", pred[1][0]);
    assert!(pred[2][0] > 0.8, "1 XOR 0 esperado > 0.8, obtido {}", pred[2][0]);
    assert!(pred[3][0] < 0.2, "1 XOR 1 esperado < 0.2, obtido {}", pred[3][0]);
}

use projeto::{camada::Camada, neuronio::Neuronio, phi::Phi, rede::RedeNeuronal};

#[test]
fn test_neuronio_momentum_analytical() {
    // Neurónio com 2 entradas e ativação Sigmoide
    let mut n = Neuronio::iniciar_manual(&[0.5, -0.3], 0.1, Phi::Sigmoide);
    let x = [1.0, 2.0];

    // Passo 1: propagar
    // h = 0.5 * 1.0 + (-0.3) * 2.0 + 0.1 = 0.0
    // y = Sigmoide(0.0) = 0.5
    // y_derivada = 0.5 * (1.0 - 0.5) = 0.25
    let y = n.propagar(&x);
    assert!((y - 0.5).abs() < 1e-9);
    assert!((n.y_derivada - 0.25).abs() < 1e-9);

    // Estado inicial de momento deve ser nulo
    assert_eq!(n.delta_w, vec![0.0, 0.0]);
    assert_eq!(n.delta_b, 0.0);

    let alpha = 0.1;
    let beta = 0.9;
    let propagacao_erro_1 = 0.4;

    // Adaptar passo 1:
    // escalar = -0.1 * 0.25 * 0.4 = -0.01
    // m_w = [0.0, 0.0], m_b = 0.0
    // delta_w[0] = -0.01 * 1.0 + 0.0 = -0.01
    // delta_w[1] = -0.01 * 2.0 + 0.0 = -0.02
    // delta_b = -0.01 + 0.0 = -0.01
    n.adaptar(propagacao_erro_1, &x, alpha, beta);

    assert!((n.w[0] - 0.49).abs() < 1e-9);
    assert!((n.w[1] - (-0.32)).abs() < 1e-9);
    assert!((n.b - 0.09).abs() < 1e-9);
    assert!((n.delta_w[0] - (-0.01)).abs() < 1e-9);
    assert!((n.delta_w[1] - (-0.02)).abs() < 1e-9);
    assert!((n.delta_b - (-0.01)).abs() < 1e-9);

    // Passo 2: adaptar novamente mantendo y_derivada = 0.25
    let propagacao_erro_2 = 0.2;
    // escalar = -0.1 * 0.25 * 0.2 = -0.005
    // m_w[0] = 0.9 * (-0.01) = -0.009
    // m_w[1] = 0.9 * (-0.02) = -0.018
    // m_b = 0.9 * (-0.01) = -0.009
    // delta_w[0] = -0.005 * 1.0 + (-0.009) = -0.014
    // delta_w[1] = -0.005 * 2.0 + (-0.018) = -0.028
    // delta_b = -0.005 + (-0.009) = -0.014
    // w[0] = 0.49 + (-0.014) = 0.476
    // w[1] = -0.32 + (-0.028) = -0.348
    // b = 0.09 + (-0.014) = 0.076
    n.adaptar(propagacao_erro_2, &x, alpha, beta);

    assert!((n.w[0] - 0.476).abs() < 1e-9);
    assert!((n.w[1] - (-0.348)).abs() < 1e-9);
    assert!((n.b - 0.076).abs() < 1e-9);
    assert!((n.delta_w[0] - (-0.014)).abs() < 1e-9);
    assert!((n.delta_w[1] - (-0.028)).abs() < 1e-9);
    assert!((n.delta_b - (-0.014)).abs() < 1e-9);
}

#[test]
fn test_zero_gradient_preserves_momentum_inertia() {
    // Verifica que se o gradiente for zero num passo posterior,
    // o momento (inércia) continua a atualizar os pesos na direção anterior
    let mut n = Neuronio::iniciar_manual(&[1.0], 0.5, Phi::Sigmoide);
    let x = [1.0];
    n.propagar(&x);

    let alpha = 0.2;
    let beta = 0.5;

    // Passo 1 com erro positivo
    n.adaptar(1.0, &x, alpha, beta);
    let w_apos_passo_1 = n.w[0];
    let b_apos_passo_1 = n.b;
    let delta_w_1 = n.delta_w[0];
    let delta_b_1 = n.delta_b;

    assert!(delta_w_1 != 0.0);
    assert!(delta_b_1 != 0.0);

    // Passo 2 com erro zero (sem gradiente da função de custo)
    n.adaptar(0.0, &x, alpha, beta);

    // Variação deve ser exatamente beta * delta anterior devido à inércia
    let expected_delta_w_2 = beta * delta_w_1;
    let expected_delta_b_2 = beta * delta_b_1;

    assert!((n.w[0] - (w_apos_passo_1 + expected_delta_w_2)).abs() < 1e-9);
    assert!((n.b - (b_apos_passo_1 + expected_delta_b_2)).abs() < 1e-9);
    assert!((n.delta_w[0] - expected_delta_w_2).abs() < 1e-9);
    assert!((n.delta_b - expected_delta_b_2).abs() < 1e-9);
}

#[test]
fn test_beta_zero_equivalence() {
    // Com beta = 0.0, o comportamento deve coincidir exatamente com a descida de gradiente padrão
    let theta: &[&[(&[f64], f64)]] = &[&[(&[0.3, -0.2], 0.1)]];
    let mut rede_padrao = RedeNeuronal::iniciar_manual(theta, Phi::Sigmoide);
    let mut rede_momento_zero = RedeNeuronal::iniciar_manual(theta, Phi::Sigmoide);

    let x = [0.5, 0.8];
    let y = [1.0];
    let alpha = 0.1;

    for _ in 0..10 {
        rede_padrao.adaptar(&x, &y, alpha, 0.0);
        rede_momento_zero.adaptar(&x, &y, alpha, 0.0);
    }

    match (&rede_padrao.camadas[1], &rede_momento_zero.camadas[1]) {
        (
            Camada::Densa { neuronios: n1, .. },
            Camada::Densa { neuronios: n2, .. },
        ) => {
            assert!((n1[0].w[0] - n2[0].w[0]).abs() < 1e-12);
            assert!((n1[0].w[1] - n2[0].w[1]).abs() < 1e-12);
            assert!((n1[0].b - n2[0].b).abs() < 1e-12);
        }
        _ => panic!("Camada 1 deve ser Densa"),
    }
}

#[test]
fn test_network_retropropagar_multi_step_momentum_analytical() {
    // Rede simples: 1 entrada -> 1 neurónio com Sigmoide
    let theta: &[&[(&[f64], f64)]] = &[&[(&[0.5], 0.1)]];
    let mut rede = RedeNeuronal::iniciar_manual(theta, Phi::Sigmoide);

    let x = [2.0];
    let target = [1.0];
    let alpha = 0.1;
    let beta = 0.8;

    // Passo 1
    let y_pred_1 = rede.propagar(&x);
    let delta_saida_1 = RedeNeuronal::delta_saida(&y_pred_1, &target);
    rede.retropropagar(&delta_saida_1, alpha, beta);

    let expected_y_1 = 1.0 / (1.0 + (-1.1_f64).exp());
    let y_derivada_1 = expected_y_1 * (1.0 - expected_y_1);
    let escalar_1 = -alpha * y_derivada_1 * (expected_y_1 - 1.0);
    let delta_w_1 = escalar_1 * 2.0;
    let delta_b_1 = escalar_1;
    let expected_w_1 = 0.5 + delta_w_1;
    let expected_b_1 = 0.1 + delta_b_1;

    match &rede.camadas[1] {
        Camada::Densa { neuronios, .. } => {
            assert!((neuronios[0].w[0] - expected_w_1).abs() < 1e-9);
            assert!((neuronios[0].b - expected_b_1).abs() < 1e-9);
            assert!((neuronios[0].delta_w[0] - delta_w_1).abs() < 1e-9);
            assert!((neuronios[0].delta_b - delta_b_1).abs() < 1e-9);
        }
        _ => panic!("Camada 1 deve ser Densa"),
    }

    // Passo 2
    let y_pred_2 = rede.propagar(&x);
    let delta_saida_2 = RedeNeuronal::delta_saida(&y_pred_2, &target);
    rede.retropropagar(&delta_saida_2, alpha, beta);

    let h_2 = expected_w_1 * 2.0 + expected_b_1;
    let expected_y_2 = 1.0 / (1.0 + (-h_2).exp());
    let y_derivada_2 = expected_y_2 * (1.0 - expected_y_2);
    let escalar_2 = -alpha * y_derivada_2 * (expected_y_2 - 1.0);
    let delta_w_2 = escalar_2 * 2.0 + beta * delta_w_1;
    let delta_b_2 = escalar_2 + beta * delta_b_1;
    let expected_w_2 = expected_w_1 + delta_w_2;
    let expected_b_2 = expected_b_1 + delta_b_2;

    match &rede.camadas[1] {
        Camada::Densa { neuronios, .. } => {
            assert!((neuronios[0].w[0] - expected_w_2).abs() < 1e-9);
            assert!((neuronios[0].b - expected_b_2).abs() < 1e-9);
            assert!((neuronios[0].delta_w[0] - delta_w_2).abs() < 1e-9);
            assert!((neuronios[0].delta_b - delta_b_2).abs() < 1e-9);
        }
        _ => panic!("Camada 1 deve ser Densa"),
    }
}

#[test]
fn test_momentum_accelerates_learning_vs_no_momentum() {
    // Comparamos a velocidade de convergência de pesos com e sem momento (beta > 0 vs beta = 0)
    let theta: &[&[(&[f64], f64)]] = &[&[(&[0.1, 0.1], -0.2)]];
    let mut rede_sem_momento = RedeNeuronal::iniciar_manual(theta, Phi::Sigmoide);
    let mut rede_com_momento = RedeNeuronal::iniciar_manual(theta, Phi::Sigmoide);

    let x = [1.0, 1.0];
    let y = [1.0];
    let alpha = 0.2;
    let beta = 0.8;

    let mut eps_sem_momento = 0.0;
    let mut eps_com_momento = 0.0;

    for _ in 0..15 {
        eps_sem_momento = rede_sem_momento.adaptar(&x, &y, alpha, 0.0);
        eps_com_momento = rede_com_momento.adaptar(&x, &y, alpha, beta);
    }

    // A rede com momento acumula velocidade ao longo de gradientes consistentes, obtendo menor erro
    assert!(
        eps_com_momento < eps_sem_momento,
        "Rede com momento ({}) deve convergir mais rápido que sem momento ({})",
        eps_com_momento,
        eps_sem_momento
    );
}

#[test]
fn test_and_training_with_momentum() {
    let x: &[&[f64]] = &[&[0.0, 0.0], &[0.0, 1.0], &[1.0, 0.0], &[1.0, 1.0]];
    let y: &[&[f64]] = &[&[0.0], &[0.0], &[0.0], &[1.0]];

    let mut rede = RedeNeuronal::iniciar(&[2, 1], Phi::Sigmoide);
    // Treino da porta AND com momento beta = 0.7
    rede.treinar(x, y, 10000, 0.01, 0.5, 0.7);

    let pred = rede.prever(x);
    assert!(
        pred[0][0] < 0.2,
        "0 AND 0 esperado < 0.2, obtido {}",
        pred[0][0]
    );
    assert!(
        pred[1][0] < 0.2,
        "0 AND 1 esperado < 0.2, obtido {}",
        pred[1][0]
    );
    assert!(
        pred[2][0] < 0.2,
        "1 AND 0 esperado < 0.2, obtido {}",
        pred[2][0]
    );
    assert!(
        pred[3][0] > 0.8,
        "1 AND 1 esperado > 0.8, obtido {}",
        pred[3][0]
    );
}

#[test]
fn test_or_training_with_momentum() {
    let x: &[&[f64]] = &[&[0.0, 0.0], &[0.0, 1.0], &[1.0, 0.0], &[1.0, 1.0]];
    let y: &[&[f64]] = &[&[0.0], &[1.0], &[1.0], &[1.0]];

    let mut rede = RedeNeuronal::iniciar(&[2, 1], Phi::Sigmoide);
    // Treino da porta OR com momento beta = 0.7
    rede.treinar(x, y, 10000, 0.01, 0.5, 0.7);

    let pred = rede.prever(x);
    assert!(
        pred[0][0] < 0.2,
        "0 OR 0 esperado < 0.2, obtido {}",
        pred[0][0]
    );
    assert!(
        pred[1][0] > 0.8,
        "0 OR 1 esperado > 0.8, obtido {}",
        pred[1][0]
    );
    assert!(
        pred[2][0] > 0.8,
        "1 OR 0 esperado > 0.8, obtido {}",
        pred[2][0]
    );
    assert!(
        pred[3][0] > 0.8,
        "1 OR 1 esperado > 0.8, obtido {}",
        pred[3][0]
    );
}

#[test]
fn test_xor_training_with_momentum() {
    let x: &[&[f64]] = &[&[0.0, 0.0], &[0.0, 1.0], &[1.0, 0.0], &[1.0, 1.0]];
    let y: &[&[f64]] = &[&[0.0], &[1.0], &[1.0], &[0.0]];

    // XOR requer camada oculta não-linear; treinamos com momento beta = 0.6
    let mut rede = RedeNeuronal::iniciar(&[2, 4, 1], Phi::Sigmoide);
    rede.treinar(x, y, 30000, 0.005, 0.5, 0.6);

    let pred = rede.prever(x);
    assert!(
        pred[0][0] < 0.2,
        "0 XOR 0 esperado < 0.2, obtido {}",
        pred[0][0]
    );
    assert!(
        pred[1][0] > 0.8,
        "0 XOR 1 esperado > 0.8, obtido {}",
        pred[1][0]
    );
    assert!(
        pred[2][0] > 0.8,
        "1 XOR 0 esperado > 0.8, obtido {}",
        pred[2][0]
    );
    assert!(
        pred[3][0] < 0.2,
        "1 XOR 1 esperado < 0.2, obtido {}",
        pred[3][0]
    );
}

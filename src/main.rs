use projeto::{phi::Phi, rede::RedeNeuronal};

/**
 * A seguinte função pretende reproduzir o exemplo "Tarefa de reconhecimento de barras verticais" dado
 * no ppw "Redes neuronais artificiais - Parte 3".
 *
 * Resultados:
 * vbar test 1 -> predicted output: 0.80
 * vbar test 2 -> predicted output: 0.85
 * vbar test 3 -> predicted output: 0.87
 * vbar test 4 -> predicted output: -0.00
 * vbar test 5 -> predicted output: -0.02
 * vbar generalization test 1 -> predicted output: 0.56
 * vbar generalization test 2 -> predicted output: 0.95
 * vbar generalization test 3 -> predicted output: 0.73
 * vbar generalization test 4 -> predicted output: 0.43
 * vbar generalization test 5 -> predicted output: -0.57
 * vbar generalization test 6 -> predicted output: -0.24
 */
fn reconhecimento_de_barras_verticais() {
    let forma = [9, 3, 1];
    let alpha = 0.2;
    let n_epocas = 1000;
    let epsilon_max = 0.05;
    let phi = Phi::Tanh;

    let x_train: &[&[f64]] = &[
        &[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        &[1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0],
        &[0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0],
        &[0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0],
        &[1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        &[0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0],
        &[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0],
    ];

    let y_train: &[&[f64]] = &[&[0.0], &[1.0], &[1.0], &[1.0], &[0.0], &[0.0], &[0.0]];

    let x_test: &[&[f64]] = &[
        &[1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0],
        &[0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0],
        &[0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0],
        // two horizontal bar tests
        &[1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        &[0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0],
    ];

    let mut rede = RedeNeuronal::iniciar(&forma, phi);
    rede.treinar(x_train, y_train, n_epocas, epsilon_max, alpha);

    let pred = rede.prever(x_test);

    println!("vbar test 1 -> predicted output: {:.2}", pred[0][0]);
    println!("vbar test 2 -> predicted output: {:.2}", pred[1][0]);
    println!("vbar test 3 -> predicted output: {:.2}", pred[2][0]);
    println!("vbar test 4 -> predicted output: {:.2}", pred[3][0]);
    println!("vbar test 5 -> predicted output: {:.2}", pred[4][0]);

    let x_test_generalized: &[&[f64]] = &[
        &[0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0],
        &[0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
        &[0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
        &[0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        &[0.0, 0.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 0.0],
        &[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 0.0],
    ];

    let pred = rede.prever(x_test_generalized);

    println!(
        "vbar generalization test 1 -> predicted output: {:.2}",
        pred[0][0]
    );
    println!(
        "vbar generalization test 2 -> predicted output: {:.2}",
        pred[1][0]
    );
    println!(
        "vbar generalization test 3 -> predicted output: {:.2}",
        pred[2][0]
    );
    println!(
        "vbar generalization test 4 -> predicted output: {:.2}",
        pred[3][0]
    );
    println!(
        "vbar generalization test 5 -> predicted output: {:.2}",
        pred[4][0]
    );
    println!(
        "vbar generalization test 6 -> predicted output: {:.2}",
        pred[5][0]
    );
}

fn main() {
    reconhecimento_de_barras_verticais();
}

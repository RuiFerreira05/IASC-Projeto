use projeto::{phi::Phi, rede::RedeNeuronal};
use rand::random_range;

fn gen_vbar_array(
    x_len: usize,
    y_len: usize,
    vbars_index: &[usize],
    noise: f64,
) -> Vec<Vec<usize>> {
    assert!(
        vbars_index.iter().all(|&i| i < x_len),
        "Nenhum valor de vbar_index deve ser maior que x_len"
    );
    let mut array = Vec::with_capacity(y_len);
    for _ in 0..y_len {
        let mut row = vec![0; x_len];
        for &i in vbars_index {
            row[i] = 1;
        }
        if noise > 0.0 {
            for x_i in 0..row.len() {
                let rand = random_range(0.0..=1.0);
                if rand <= noise {
                    if row[x_i] == 0 {
                        row[x_i] = 1
                    } else {
                        row[x_i] = 0
                    }
                }
            }
        }
        array.push(row);
    }
    array
}

#[test]
#[should_panic]
fn vbar_gen_fn_index_overflow() {
    gen_vbar_array(3, 3, &[10], 0.0);
}

#[test]
fn vbar_gen_fn_test() {
    let gen_vbars = gen_vbar_array(4, 4, &[1, 3], 0.0);
    let test_vbars = vec![
        vec![0, 1, 0, 1],
        vec![0, 1, 0, 1],
        vec![0, 1, 0, 1],
        vec![0, 1, 0, 1],
    ];
    assert_eq!(test_vbars, gen_vbars)
}

#[test]
fn vbar_learning_test() {
    let forma = [9, (9 + 1) / 2, 1];
    let phi = Phi::Sigmoide;
    let n_epocas = 1000;
    let epsilon_max = 0.005;
    let alpha = 0.5;
    let beta = 0.0;

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
    rede.treinar(x_train, y_train, n_epocas, epsilon_max, alpha, beta);

    let pred = rede.prever(x_test);

    assert!(pred[0][0] > 0.8);
    assert!(pred[1][0] > 0.8);
    assert!(pred[2][0] > 0.8);
    assert!(pred[3][0] < 0.2);
    assert!(pred[4][0] < 0.2);
}

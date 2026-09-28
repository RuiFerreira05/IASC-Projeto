use projeto::{neuronio, phi::Phi, rede::RedeNeuronal};
use rand::random_range;

fn usize_to_binary_array(val: usize) -> Vec<usize> {
    (0..usize::BITS).rev().map(|i| (val >> i) & 1).collect()
}

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

fn gen_random_vbar_array(
    x_len: usize,
    y_len: usize,
    vbar_num: usize,
    noise: f64,
) -> Vec<Vec<usize>> {
    assert!(
        vbar_num <= x_len,
        "Número de barras verticais deve ser menor que x_len"
    );

    let mut vbar_indices: Vec<usize> = Vec::with_capacity(vbar_num);

    while vbar_indices.len() < vbar_num {
        let rand = random_range(0..x_len);
        if vbar_indices.contains(&rand) {
            continue;
        }
        vbar_indices.push(rand);
    }

    gen_vbar_array(x_len, y_len, &vbar_indices, noise)
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

    assert!(pred[0][0] > 0.8);
    assert!(pred[1][0] > 0.8);
    assert!(pred[2][0] > 0.8);
    assert!(pred[3][0] < 0.2);
    assert!(pred[4][0] < 0.2);
}

// #[test]
// fn vbar_learning_test_with_counting() {
//     let x_len = 100;
//     let y_len = 100;

//     let phi = Phi::Sigmoide;
//     let num_neuronios_camada_escondida = 100;
//     let num_neuronios_camada_saida = 8;

//     let train_array_size = 100;
//     let train_noise = 0.0;

//     let test_array_size = 10;
//     let test_noise = train_noise;

//     // Generate train x and y values
//     let xy_train: Vec<(Vec<Vec<usize>>, usize)> = (0..train_array_size)
//         .map(|_| {
//             let vbar_num = random_range(0..y_len);

//             // return the tuple
//             (
//                 gen_random_vbar_array(x_len, y_len, vbar_num, train_noise),
//                 vbar_num,
//             )
//         })
//         .collect();

//     // Generate test x and y values
//     let xy_test: Vec<(Vec<Vec<usize>>, usize)> = (0..test_array_size)
//         .map(|_| {
//             let vbar_num = random_range(0..y_len);

//             // return the tuple
//             (
//                 gen_random_vbar_array(x_len, y_len, vbar_num, test_noise),
//                 vbar_num,
//             )
//         })
//         .collect();

//     let num_neuronios_entrada = x_len*y_len;

//     let rede = RedeNeuronal::iniciar(, Phi::Sigmoide);
// }

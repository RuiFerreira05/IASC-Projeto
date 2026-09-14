use rand::random_range;

use crate::phi::Phi;

/**
 Este Struct representa a unidade singular de uma rede neuronal, o neurónio.

 Este elemento é responsável por aplicar a um vetor de entrada uma função de transferência que produz um vetor de saida.
 Esta funcção de transfência consiste de aplicar uma função de ativação (aqui representada pelo Struct "Phi") sob a
 soma de ativação (produto escalar das entradas pelos pesos, aqui representados pelo vetor "w", mais o valor do pendor,
 representado pela propriedade "b").
 */
pub struct Neuronio {
    pub w: Vec<f64>,
    pub b: f64,
    pub phi: Phi,
}

impl Neuronio {
    /**
     O neuronio é iniciado com dois parametros:
     * A dimensão do vetor de entrada que estabelece a dimensão do vetor de pesos;
     * A função de ativação aplicada sob a soma de ativação
     */
    pub fn iniciar(d: usize, phi: Phi) -> Self {
        Neuronio {
            // Para cada entrada, gerar um peso aleatório entre -1 e 1, inclusive
            w: (0..d).map(|_| random_range(-1.0..=1.0)).collect(),

            // Gerar um pendor aleatório entre -1 e 1
            b: random_range(-1.0..=1.0),
            phi,
        }
    }

    /**
     O seguinte método pretende representar a função de tranfêrencia do neurónio aplicada a um vetor de entrada, x.
     */
    pub fn propagar(&self, x: &[f64]) -> f64 {
        // Certificar que o número de pesos é igual ao número de entradas
        assert_eq!(
            x.len(),
            self.w.len(),
            "Tamanho de entrada deve ser igual ao numero de pesos (Entradas esperadas: {}. Entradas dadas: {})",
            self.w.len(),
            x.len()
        );

        // Calcular o produto escalar (dot product)
        let dot: f64 = self.w.iter().zip(x).map(|(w_i, x_i)| w_i * x_i).sum();
        // calcular a soma de ativação
        let h: f64 = dot + self.b;

        // aplicar a função de ativação sob a soma de ativação
        self.phi.aplicar(h)
    }
}

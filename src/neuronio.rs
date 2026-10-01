use rand::random_range;

use crate::phi::Phi;

/**
Este Struct representa a unidade singular de uma rede neuronal, o neurónio.

Este elemento é responsável por aplicar a um vetor de entrada uma função de transferência que produz um vetor de saida.
Esta funcção de transfência consiste de aplicar uma função de ativação (aqui representada pelo Struct "Phi") sob a
soma de ativação (produto escalar das entradas pelos pesos, aqui representados pelo vetor "w", mais o valor do pendor,
representado pela propriedade "b").
*/
#[derive(Debug, Clone)]
pub struct Neuronio {
    /// Tamanho do vetor de entrada
    pub d: usize,

    /// Função de ativação
    pub phi: Phi,

    /// Vetor de pesos (len(w) == d)
    pub w: Vec<f64>,

    /// pendor
    pub b: f64,

    /// soma de ativação
    pub h: f64,

    /// saida do neurónio (y = f(x))
    pub y: f64,

    /// derivada da saida do neurónio
    pub y_derivada: f64,
}

impl Neuronio {
    /**
    O neuronio é iniciado com dois parametros:
    * A dimensão do vetor de entrada que estabelece a dimensão do vetor de pesos;
    * A função de ativação aplicada sob a soma de ativação
    */
    pub fn iniciar(d: usize, phi: Phi) -> Self {
        let w: Vec<f64> = (0..d).map(|_| random_range(-1.0..=1.0)).collect();
        let b: f64 = random_range(-1.0..=1.0);

        Neuronio::iniciar_manual(&w, b, phi)
    }

    /**
    Este função permite iniciar os neurónios com pesos especificos
    */
    pub fn iniciar_manual(w: &[f64], b: f64, phi: Phi) -> Self {
        Neuronio {
            d: w.len(),
            phi,
            w: w.to_vec(),
            b,
            h: 0.0,
            y: 0.0,
            y_derivada: 0.0,
        }
    }

    /**
    O seguinte método pretende representar a função de tranfêrencia do neurónio aplicada a um vetor de entrada, x.
    */
    pub fn propagar(&mut self, x: &[f64]) -> f64 {
        // Certificar que o número de pesos é igual ao número de entradas
        assert_eq!(
            x.len(),
            self.d,
            "Tamanho de entrada deve ser igual ao numero de pesos (Entradas esperadas: {}. Entradas dadas: {})",
            self.d,
            x.len()
        );

        // Calcular o produto escalar (dot product)
        let dot: f64 = self.w.iter().zip(x).map(|(w_i, x_i)| w_i * x_i).sum();

        // calcular a soma de ativação
        self.h = dot + self.b;

        // aplicar a função de ativação sob a soma de ativação
        self.y = self.phi.aplicar(self.h);

        self.y_derivada = self.phi.derivar(self.h);

        // Retornar a saida do neurónio
        self.y
    }

    /**
     * A seguinte função adapta os pesos do neurónio com base em:
     * * Um componente de propagação do erro na saida do neurónio, delta
     * * O vetor de saida da camada anterior
     * * a taxa de aprendizagem
     */
    pub fn adaptar(&mut self, propagacao_erro: f64, y_anterior: &[f64], alpha: f64) {
        let escalar = -alpha * self.y_derivada * propagacao_erro;

        // Atualizar pesos
        let weight_var: Vec<f64> = y_anterior.iter().map(|&py| escalar * py).collect();
        self.w
            .iter_mut()
            .zip(weight_var)
            .for_each(|(w, var_w)| *w += var_w);

        // Atualizar pendor
        let b_var: f64 = escalar;
        self.b += b_var;
    }
}

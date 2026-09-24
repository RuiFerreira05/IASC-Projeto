use std::assert_eq;

use crate::{neuronio::Neuronio, phi::Phi};

/**
Representa uma camada da rede neuronal.
*/
pub enum Camada {
    /// Esta variante representa uma camada de entrada da rede neuronal, que se limita a transferir os valores de entrada para a sua saida
    Entrada { ds: usize, y: Vec<f64> },

    /// Esta variante representa uma camada densa da rede neuronal, onde todos os neurónios da camada anterior se encontram ligados a todos os neurónios desta camada
    Densa {
        de: usize,
        ds: usize,
        phi: Phi,
        neuronios: Vec<Neuronio>,
    },
}

impl Camada {
    /**
    Inicia uma camada de entrada a partir da sua dimensão de saída.
    */
    pub fn iniciar_entrada(ds: usize) -> Self {
        Camada::Entrada {
            ds,
            y: vec![0.0; ds],
        }
    }

    /**
    Inicia uma camada densa especificando as dimensões de entrada, saída e a função de ativação.
    */
    pub fn iniciar_densa(de: usize, ds: usize, phi: Phi) -> Self {
        Camada::Densa {
            de,
            ds,
            phi,
            neuronios: (0..ds).map(|_| Neuronio::iniciar(de, phi)).collect(),
        }
    }

    /**
     * Esta função permite iniciar uma camada da rede neuronal com neurónios cujos pesos e pendors são determinados manualmente
     */
    pub fn iniciar_densa_manual(w: &[&[f64]], b: &[f64], ds: usize, phi: Phi) -> Self {
        assert_eq!(
            w.len(),
            ds,
            "A primeira dimensão do vetor de pesos não tem o tamanho correto (Esperado: {}, Recebido: {})",
            ds,
            w.len()
        );

        assert_eq!(
            b.len(),
            w.len(),
            "Numero de pendores não corresponde com número de neurónios (Esperado: {}, Recebido: {})",
            w.len(),
            b.len(),
        );

        let mut neuronios = Vec::with_capacity(ds);
        for i in 0..ds {
            neuronios.push(Neuronio::iniciar_manual(w[i], b[i], phi));
        }

        let de = if ds > 0 { w[0].len() } else { 0 };
        Camada::Densa {
            de,
            ds,
            phi,
            neuronios,
        }
    }

    /**
    Propaga o vetor de entrada através da camada.
    - Se for de entrada, valida a dimensão e replica o vetor de entrada.
    - Se for densa, propaga cada um dos seus neurónios.
    */
    pub fn propagar(&mut self, x: &[f64]) -> Vec<f64> {
        match self {
            Camada::Entrada { ds, y } => {
                assert_eq!(
                    x.len(),
                    *ds,
                    "Entrada com tamanho incorreto (Esperado: {}, Recebido: {})",
                    ds,
                    x.len()
                );
                *y = x.to_vec();
                y.clone()
            }
            Camada::Densa { neuronios, .. } => neuronios
                .iter_mut()
                .map(|neuronio| neuronio.propagar(x))
                .collect(),
        }
    }

    /**
    Retorna o vetor de saídas da camada da última propagação realizada.
    */
    pub fn y(&self) -> Vec<f64> {
        match self {
            Camada::Entrada { y, .. } => y.clone(),
            Camada::Densa { neuronios, .. } => {
                neuronios.iter().map(|neuronio| neuronio.y).collect()
            }
        }
    }

    pub fn ds(&self) -> usize {
        match self {
            Camada::Entrada { ds, .. } => *ds,
            Camada::Densa { ds, .. } => *ds,
        }
    }

    pub fn neuronios(&self) -> &[Neuronio] {
        match self {
            Camada::Entrada { .. } => &[],
            Camada::Densa { neuronios, .. } => neuronios.as_slice(),
        }
    }

    pub fn adaptar(&mut self, delta: &[f64], y_anterior: &[f64], alpha: f64) {
        match self {
            Camada::Entrada { .. } => {}
            Camada::Densa { neuronios, .. } => neuronios
                .iter_mut()
                .zip(delta)
                .for_each(|(neuronio_j, delta_j)| neuronio_j.adaptar(*delta_j, y_anterior, alpha)),
        }
    }
}

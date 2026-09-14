use crate::{neuronio::Neuronio, phi::{Phi}};

pub trait Camada {
    fn propagar(&self, x: &[f64]) -> Vec<f64>;
}

/**
 O seguinte struct pretende representar uma camada de entrada de uma rede neuronal.

 A camada de entrada de uma rede neuronal tem como único objetivo disponibilizar os dados de entrada para as restantes camadas da rede neuronal.
 Por isto, esta encontra-se representada em código, como um Struct sem propriedades, cuja unica função consiste de propagar o vetor de entrada para a sua saida.
 */
pub struct CamadaEntrada {}

impl CamadaEntrada {
    /**
     Camada de entrada é iniciada a partir da dimensão de saida, que dita o número de neuronios que constituem esta.
     No entanto, visto que o único proposito desta é propagar os dados de entrada para a saida, em código esta não contem neuronios e é iniciada vazia.
     */
    pub fn iniciar(_ds: usize) -> Self {
        CamadaEntrada {}
    }
}

impl Camada for CamadaEntrada {
    /**
     Sendo o único propósito da camada de entrada propagar o vetor de entrada para a saida, este método acaba por apenas retornar o próprio vetor de entrada, para simplificar.
     */
    fn propagar(&self, x: &[f64]) -> Vec<f64> {
        x.to_vec()
    }
}

/**
 O seguinte struct representa uma camada densa da rede neuronal.

 Uma camada densa consiste de um agrupamento de neurónios onde cada um consome todo o vetor de entradas e produz uma saida atravéz da sua função de transferência.
 */
pub struct CamadaDensa {
    neuronios: Vec<Neuronio>
}

impl CamadaDensa {
    /**
     Uma camada densa é iniciada especificando:
     * A dimensão do vetor de entrada, que irá ser consumido por todos os neurónios;
     * A dimensão do vetor de saida, que consiste por si de todas as saidas de todos os neurónios (daí indicando o número de neurónios);
     * A função de ativação que cada neurónio implementará;
     */
    pub fn iniciar(de: usize, ds: usize, phi: Phi) -> Self {
        CamadaDensa {
            // Gerar "ds" neurónios, cada um iniciado com o vetor de entrada da camada, e a função de ativação.
            neuronios: (0..ds).map(|_| Neuronio::iniciar(de, phi)).collect()
        }
    }
}

impl Camada for CamadaDensa {
    /**
     Propagar uma camada densa de uma rede neuronal consiste em propagar cada neurónio dessa camada, reunindo os resultados num vetor de saida.
     */
    fn propagar(&self, x: &[f64]) -> Vec<f64> {
        self.neuronios.iter().map(|neuronio| neuronio.propagar(x)).collect()
    }
}

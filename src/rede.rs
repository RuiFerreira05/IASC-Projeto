use crate::{
    camada::{Camada, CamadaDensa, CamadaEntrada},
    phi::Phi,
};


/**
 Uma rede neuronal (neste caso, multicamada) consiste da junção de multiplas camadas de neurónios interligados entre si.
 */
pub struct RedeNeuronal {
    pub camadas: Vec<Box<dyn Camada>>,
}

impl RedeNeuronal {
    /**
     Iniciar uma rede neural requer especificar:
     * A sua forma, aqui representada por um vetor, onde cada entrada representa o número de neurónios de uma camada
     * A função de ativação que cada neurónio da rede neuronal vai implementar.
     */
    pub fn iniciar(forma: &[usize], phi: Phi) -> Self {
        // Número de camadas da rede
        let n = forma.len();
        assert!(
            n >= 2,
            "Vetor de forma deve conter pelo menos 2 camadas (tamanho dado: {})",
            n
        );

        // Gerar o vetor vazio com capacidade n
        let mut camadas: Vec<Box<dyn Camada>> = Vec::with_capacity(n);

        // Gerar a camada de entrada independentemente
        let de_1 = forma[0];
        let camada_1 = CamadaEntrada::iniciar(de_1);
        camadas.push(Box::new(camada_1));

        // Iterar sob o vetor de forma, reunir as dimensões de entrada e saida de cada camada, e gerar a camada
        for i in 1..n {
            let de_n = forma[i - 1];
            let ds_n = forma[i];
            let camada_n = CamadaDensa::iniciar(de_n, ds_n, phi);
            camadas.push(Box::new(camada_n));
        }

        // Possivel implementação com "janelas" do rust. No entanto, comentado para manter aproximação com o pseudo-codigo
        // for dim in forma.windows(2) {
        //     let de_n = dim[0];
        //     let ds_n = dim[1];
        //     let camada_n = CamadaDensa::iniciar(de_n, ds_n, phi);
        //     camadas.push(Box::new(camada_n));
        // }

        RedeNeuronal { camadas }
    }

    /**
     Propagar uma rede neuronal consiste em propagar cada camada da mesma, que por si propagam os seus neurónios.
     A propagação de uma rede neuronal produz um vetor de saida com dimensão igual ao número de neurónios da sua última camada (camada de saida)
     */
    pub fn propagar(&self, x: &[f64]) -> Vec<f64> {
        let mut y = x.to_vec();

        for camada in &self.camadas {
            y = camada.propagar(&y)
        }

        y
    }

    /**
     O seguinte método permite prever os vetores de saida resultantes da propagação da rede neuronal, quando aplicada sob vários vetores de entrada.
     */
    pub fn prever(&self, x: Vec<Vec<f64>>) -> Vec<Vec<f64>> {
        x.iter().map(|v| self.propagar(v)).collect()
    }
}

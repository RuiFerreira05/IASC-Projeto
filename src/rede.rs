use crate::{camada::Camada, phi::Phi};

/**
Uma rede neuronal (neste caso, multicamada) consiste da junção de multiplas camadas de neurónios interligados entre si.
*/
pub struct RedeNeuronal {
    pub camadas: Vec<Camada>,
}

impl RedeNeuronal {
    pub fn y(&self) -> Vec<f64> {
        self.camadas.last().unwrap().y()
    }

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
        let mut camadas: Vec<Camada> = Vec::with_capacity(n);

        // Gerar a camada de entrada
        let de_1 = forma[0];
        camadas.push(Camada::iniciar_entrada(de_1));

        // Iterar sob o vetor de forma, reunir as dimensões de entrada e saida de cada camada, e gerar a camada
        for i in 1..n {
            let de_n = forma[i - 1];
            let ds_n = forma[i];
            camadas.push(Camada::iniciar_densa(de_n, ds_n, phi));
        }

        RedeNeuronal { camadas }
    }

    /**
    Propagar uma rede neuronal consiste em propagar cada camada da mesma, que por si propagam os seus neurónios.
    A propagação de uma rede neuronal produz um vetor de saida com dimensão igual ao número de neurónios da sua última camada (camada de saida)
    */
    pub fn propagar(&mut self, x: &[f64]) -> Vec<f64> {
        let mut y = x.to_vec();

        for camada in &mut self.camadas {
            y = camada.propagar(&y)
        }

        // Retornar y diretamente já que já o calculámos
        y
    }

    /**
    O seguinte método permite prever os vetores de saida resultantes da propagação da rede neuronal, quando aplicada sob vários vetores de entrada.
    */
    pub fn prever(&mut self, x: &[&[f64]]) -> Vec<Vec<f64>> {
        x.iter().map(|v| self.propagar(v)).collect()
    }
}

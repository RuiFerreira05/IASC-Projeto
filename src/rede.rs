use crate::{camada::Camada, phi::Phi};

/**
 * Uma rede neuronal (neste caso, multicamada) consiste da junção de multiplas camadas de neurónios interligados entre si.
 */
pub struct RedeNeuronal {
    pub camadas: Vec<Camada>,
}

impl RedeNeuronal {
    pub fn y(&self) -> Vec<f64> {
        self.camadas.last().unwrap().y()
    }

    /**
     * Iniciar uma rede neural requer especificar:
     * * A sua forma, aqui representada por um vetor, onde cada entrada representa o número de neurónios de uma camada
     * * A função de ativação que cada neurónio da rede neuronal vai implementar.
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
     * A seguinte função permite a criação de uma rede neuronal com pârametros
     * prédefinidos através do parâmetro theta
     */
    pub fn iniciar_manual(theta: &[&[(&[f64], f64)]], phi: Phi) -> Self {
        let mut camadas: Vec<Camada> = Vec::with_capacity(theta.len() + 1);

        // Dimensão de entrada da segunda camada
        let de_2 = theta[0][0].0.len();
        camadas.push(Camada::iniciar_entrada(de_2));

        for camada in theta {
            let ds = camada.len();
            let (w, b): (Vec<&[f64]>, Vec<f64>) = camada.iter().copied().unzip();
            camadas.push(Camada::iniciar_densa_manual(
                w.as_slice(),
                b.as_slice(),
                ds,
                phi,
            ));
        }

        RedeNeuronal { camadas }
    }

    /**
     * Propagar uma rede neuronal consiste em propagar cada camada da mesma, que por si propagam os seus neurónios.
     * A propagação de uma rede neuronal produz um vetor de saida com dimensão igual ao número de
     * neurónios da sua última camada (camada de saida)
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

    pub fn delta_saida(y_n: &[f64], y: &[f64]) -> Vec<f64> {
        assert_eq!(
            y_n.len(),
            y.len(),
            "O vetor de saída da rede e o vetor esperado devem ter o mesmo tamanho (Saída: {}, Esperado: {})",
            y_n.len(),
            y.len()
        );
        y_n.iter().zip(y).map(|(yk_n, yk)| yk_n - yk).collect()
    }

    pub fn retropropagar(&mut self, delta_saida: &[f64], alpha: f64) {
        let mut delta_n = delta_saida.to_vec();
        for n in (1..self.camadas.len()).rev() {
            let y_anterior = self.camadas[n - 1].y();
            let dimensao_anterior = self.camadas[n - 1].ds();
            let dimensao = self.camadas[n].ds();
            let neuronios_n = self.camadas[n].neuronios();

            let mut delta_anterior = vec![0.0; dimensao_anterior];

            for i in 0..dimensao_anterior {
                let mut soma = 0.0;
                for j in 0..dimensao {
                    // neur^n[j].w_i * delta^n_j * neur^n[j].y'
                    soma += neuronios_n[j].w[i] * delta_n[j] * neuronios_n[j].y_derivada;
                }
                delta_anterior[i] = soma;
            }

            self.camadas[n].adaptar(delta_n.as_slice(), &y_anterior, alpha);

            delta_n = delta_anterior;
        }
    }

    pub fn adaptar(&mut self, x: &[f64], y: &[f64], alpha: f64) -> f64 {
        let y_n = self.propagar(x);
        let delta_n = RedeNeuronal::delta_saida(&y_n, y);

        self.retropropagar(&delta_n, alpha);
        let k = delta_n.len();
        let epsilon: f64 = delta_n
            .iter()
            .map(|delta_n_k| delta_n_k.powi(2))
            .sum::<f64>()
            / (k as f64);

        epsilon
    }

    pub fn treinar(
        &mut self,
        x: &[&[f64]],
        y: &[&[f64]],
        n_epocas: usize,
        epsilon_max: f64,
        alpha: f64,
    ) {
        assert_eq!(
            x.len(),
            y.len(),
            "O número de entradas deve ser igual ao número de saídas esperadas"
        );

        for _ in 0..n_epocas {
            let mut epsilon = 0.0;

            for (&x_set, &y_set) in x.iter().zip(y) {
                let epsilon_x = self.adaptar(x_set, y_set, alpha);
                epsilon = f64::max(epsilon, epsilon_x);
            }

            if epsilon < epsilon_max {
                break;
            }
        }
    }
}

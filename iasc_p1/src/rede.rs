use crate::{camada::Camada, phi::Phi};

/**
Representação de uma Rede Neuronal Artificial Multicamada (Multilayer Perceptron - MLP).

As redes neuronais artificiais são um modelo computacional derivado do estudo de sistemas
nervosos biológicos, em particular do cérebro, caracterizadas por processamento distribuído
massivo e com capacidade de aprendizagem e generalização.

Conceitos teóricos fundamentais:
- Separação não linear: Redes de uma única camada (como o perceptrão) não têm capacidade
  de resolver problemas que não são linearmente separáveis, como a disjunção exclusiva (XOR).
  As redes neuronais multicamada ultrapassam essa limitação através da composição de sucessivas
  fronteiras lineares nas camadas escondidas, projetando os dados em espaços de maior dimensionalidade
  onde esses dados se tornam linearmente separáveis.
- Teorema da Aproximação Universal: Uma rede neuronal com uma única camada escondida, contendo
  um número finito de neurónios, pode aproximar qualquer função contínua num intervalo compacto
  (limitado e fechado) com precisão arbitrária (epsilon > 0), desde que a função de ativação
  dos neurónios escondidos seja não linear: |g(x) - f(x)| < epsilon.
- Parâmetros da rede (theta): Conjunto de matrizes de pesos W^n e vetores de pendores b^n
  theta = { W^2, b^2, ..., W^N, b^N } para as camadas n = 2 até N.
- Retropropagação do erro: Formulação da aprendizagem como um problema de otimização, cujo
  objetivo é minimizar uma função de perda L(y, y^N) por descida de gradiente, atualizando os
  parâmetros no sentido oposto ao gradiente: theta <- theta - alpha * gradiente(L(theta)).
- Termo de momento: Introduz inércia de deslocação baseada na variação anterior dos parâmetros
  (beta * delta theta anterior), permitindo acelerar a convergência, amortecer oscilações e
  ultrapassar regiões menos favoráveis do espaço de pesos (como regiões planas ou de baixo gradiente).
*/
pub struct RedeNeuronal {
    /// Conjunto de camadas da rede (camada 1 de entrada, camadas escondidas e camada de saída).
    pub camadas: Vec<Camada>,
}

impl RedeNeuronal {
    /**
    Retorna o vetor de saídas y^N da última camada da rede na propagação mais recente.
    */
    pub fn y(&self) -> Vec<f64> {
        self.camadas.last().unwrap().y()
    }

    /**
    Inicializa uma rede neuronal multicamada a partir da lista de neurónios por camada (forma) e da função de ativação phi.

    Parâmetros:
    - forma: Lista com o número de neurónios por camada [ds^1, ds^2, ..., ds^N], onde:
      - N = comprimento de forma (profundidade da rede);
      - forma[0] define a dimensão da camada de entrada;
      - forma[n] define a largura da camada n;
      - forma[N-1] define a dimensão da camada de saída.
    - phi: Função de ativação dos neurónios das camadas escondidas e de saída.

    Valores iniciais das camadas:
    - Camada 1: Camada de entrada com dimensão ds^1 = forma[0].
    - Camadas n = 2 até N: Camadas densas com de^n = forma[n-1] e ds^n = forma[n].
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
    Inicializa a rede neuronal com parâmetros pré-definidos theta = { W^2, b^2, ..., W^N, b^N }.
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
    Ativação direta da rede: produz a saída y da rede com base na entrada x
    por ativação sequencial encadeada das várias camadas da rede.

    Relações de ativação:
    - y^1 = x (com x^1 = x).
    - h^n = x^n * W^n + b^n, com x^n = y^(n-1) para n > 1.
    - y^n = phi^n(h^n).
    - Saída final da rede: y^N.
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
    Prevê o conjunto de vetores de saída Y para um conjunto de vetores de entrada X:
    Y = [propagar(x) para x em X].
    */
    pub fn prever(&mut self, x: &[&[f64]]) -> Vec<Vec<f64>> {
        x.iter().map(|v| self.propagar(v)).collect()
    }

    /**
    Calcula a componente de variação do erro na camada de saída delta^N.

    Considerando a função de perda definida pelo Erro Quadrático Médio com fator de normalização:
    L(y, y^N) = (1/2) * somatório((y_k^N - y_k)^2) para k = 1 até K.

    A derivada da perda em relação a cada saída estimada y_k^N resulta em:
    delta_k^N = dL / dy_k^N = y_k^N - y_k
    onde:
    - y_k^N: valor estimado da saída k gerado pela rede.
    - y_k: valor alvo a aprender para a saída k.
    */
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

    /**
    Retropropaga o erro ao longo das sucessivas camadas da rede no sentido inverso (da saída para a entrada).

    Para cada camada n de N até 2 (passo -1):
    1. Calcula a componente de propagação de erro para a camada anterior n - 1 através da regra da cadeia:
       delta_i^(n-1) = somatório_j(neur^n[j].w_i * delta_j^n * neur^n[j].y') para i = 1 até d^(n-1).
    2. Adapta os parâmetros da camada n (pesos e pendores) com base em delta^n, y^(n-1), taxa alpha e momento beta.
    3. Atualiza o vetor de erro: delta^n = delta^(n-1).
    */
    pub fn retropropagar(&mut self, delta_saida: &[f64], alpha: f64, beta: f64) {
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

            self.camadas[n].adaptar(delta_n.as_slice(), &y_anterior, alpha, beta);

            delta_n = delta_anterior;
        }
    }

    /**
    Adapta os parâmetros da rede neuronal para uma única amostra de treino (x, y).

    Passos executados:
    1. Propagação direta: y^N = propagar(x).
    2. Perda de saída da rede: delta^N = delta_saida(y^N, y).
    3. Retropropagação do erro: retropropagar(delta^N, alpha, beta).
    4. Cálculo do erro médio de saída da rede:
       epsilon = (1/K) * somatório((delta_k^N)^2) para k = 1 até K.
    */
    pub fn adaptar(&mut self, x: &[f64], y: &[f64], alpha: f64, beta: f64) -> f64 {
        let y_n = self.propagar(x);
        let delta_n = RedeNeuronal::delta_saida(&y_n, y);

        self.retropropagar(&delta_n, alpha, beta);
        let k = delta_n.len();
        let epsilon: f64 = delta_n
            .iter()
            .map(|delta_n_k| delta_n_k.powi(2))
            .sum::<f64>()
            / (k as f64);

        epsilon
    }

    /**
    Treina a rede neuronal para um conjunto de entradas X e saídas Y.

    Organização geral do treino:
    - Repetir para várias épocas de treino:
      - Inicializar o erro da época: epsilon = 0.
      - Para cada par (x, y) no conjunto de treino:
        - epsilon_x = adaptar(x, y, alpha, beta).
        - epsilon = max(epsilon, epsilon_x).
      - Critério de paragem: se epsilon <= epsilon_max, interromper o treino.
    */
    pub fn treinar(
        &mut self,
        x: &[&[f64]],
        y: &[&[f64]],
        n_epocas: usize,
        epsilon_max: f64,
        alpha: f64,
        beta: f64,
    ) {
        assert_eq!(
            x.len(),
            y.len(),
            "O número de entradas deve ser igual ao número de saídas esperadas"
        );

        for _ in 0..n_epocas {
            let mut epsilon = 0.0;

            for (&x_set, &y_set) in x.iter().zip(y) {
                let epsilon_x = self.adaptar(x_set, y_set, alpha, beta);
                epsilon = f64::max(epsilon, epsilon_x);
            }

            if epsilon < epsilon_max {
                break;
            }
        }
    }
}

use rand::random_range;

use crate::phi::Phi;

/**
Representação da unidade elementar de processamento: o neurónio artificial.

O modelo computacional é inspirado no funcionamento do neurónio biológico:
- Dendrites / Entradas (x): Sinais de entrada que chegam ao neurónio através de
  ligações com outros neurónios ou com as entradas da rede.
- Sinapses / Pesos (w): Permeabilidade sináptica que determina a influência
  de cada ligação para a resposta do neurónio. O produto escalar w . x representa
  uma medida de semelhança entre as entradas e os pesos, funcionando
  o neurónio como um protótipo de dados que a rede tem capacidade de reconhecer.
- Soma / Pendor (b): Parâmetro que possibilita o ajuste do limiar de ativação
  da rede. Geometricamente, determina a deslocação da fronteira de discriminação
  definida por h = w . x + b, enquanto os pesos w determinam o seu declive ou orientação.
- Axónio / Saída (y): Sinal transmitido após aplicação da função de transferência
  y = phi(h), onde h = w . x + b.

Geometricamente, um neurónio define uma fronteira de discriminação no espaço das entradas
(reta, plano ou hiperplano, em função do número de entradas) entre a região de resposta
excitadora e a região de resposta inibitória.
*/
#[derive(Debug, Clone)]
pub struct Neuronio {
    /// Dimensão de entrada d (número de ligações de entrada).
    pub d: usize,

    /// Função de ativação phi.
    pub phi: Phi,

    /// Vetor de pesos das ligações de entrada w = (w_1, w_2, ..., w_d).
    pub w: Vec<f64>,

    /// Pendor interno do neurónio b, responsável pela deslocação do limiar de ativação.
    pub b: f64,

    /// Vetor de variação dos pesos delta_w, utilizado pelo termo de momento.
    pub delta_w: Vec<f64>,

    /// Variação do pendor delta_b, utilizada pelo termo de momento.
    pub delta_b: f64,

    /// Soma de ativação das entradas: h = w . x + b.
    pub h: f64,

    /// Saída do neurónio: y = phi(h).
    pub y: f64,

    /// Derivada da saída em ordem à soma de ativação: y' = d phi(h) / dh.
    pub y_derivada: f64,
}

impl Neuronio {
    /**
    Inicializa um neurónio com valores aleatórios para os pesos e pendor.

    Valores iniciais:
    - w: vetor aleatório no intervalo [-1, 1] com dimensão d.
    - b: valor aleatório no intervalo [-1, 1].
    - delta_w: vetor de zeros com dimensão d.
    - delta_b: 0.0.
    - h: 0.0.
    - y: 0.0.
    - y': 0.0.
    */
    pub fn iniciar(d: usize, phi: Phi) -> Self {
        let w: Vec<f64> = (0..d).map(|_| random_range(-1.0..=1.0)).collect();
        let b: f64 = random_range(-1.0..=1.0);

        Neuronio::iniciar_manual(&w, b, phi)
    }

    /**
    Inicializa um neurónio com valores específicos para os pesos w e pendor b.
    */
    pub fn iniciar_manual(w: &[f64], b: f64, phi: Phi) -> Self {
        Neuronio {
            d: w.len(),
            phi,
            w: w.to_vec(),
            b,
            delta_w: vec![0.0; w.len()],
            delta_b: 0.0,
            h: 0.0,
            y: 0.0,
            y_derivada: 0.0,
        }
    }

    /**
    Propaga o vetor de entradas x gerando a saída y do neurónio e a respetiva derivada y'.

    Etapas do processamento:
    1. Cálculo da soma de ativação h: h = w . x + b = somatório(x_i * w_i) + b.
    2. Aplicação da função de ativação: y = phi(h).
    3. Cálculo da derivada da saída: y' = d phi(h) / dh.
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
    Adapta os pesos das ligações de entrada e o pendor com suporte a termo de momento.

    Parâmetros:
    - propagacao_erro (delta): Componente de propagação do erro de saída do neurónio delta_j^n.
    - y_anterior (y^(n-1)): Vetor de saída da camada anterior n - 1.
    - alpha (a): Taxa de aprendizagem que determina a dimensão dos passos do algoritmo.
    - beta (b): Fator de momento que introduz inércia para acelerar convergência e amortecer oscilações.

    Fórmulas de atualização:
    - Momento dos pesos: M_w = beta * delta_w
    - Variação dos pesos: delta_w = -alpha * y' * delta * y^(n-1) + M_w
    - Atualização dos pesos: w = w + delta_w
    - Momento do pendor: M_b = beta * delta_b
    - Variação do pendor: delta_b = -alpha * y' * delta + M_b
    - Atualização do pendor: b = b + delta_b
    */
    pub fn adaptar(&mut self, propagacao_erro: f64, y_anterior: &[f64], alpha: f64, beta: f64) {
        // Calcular momento dos pesos
        let m_w: Vec<f64> = self.delta_w.iter().map(|w| beta * w).collect();

        // Calcular variação dos pesos
        let escalar = -alpha * self.y_derivada * propagacao_erro;
        let delta_w: Vec<f64> = y_anterior
            .iter()
            .zip(m_w)
            .map(|(&py, momento)| escalar * py + momento)
            .collect();

        // Atualizar pesos
        self.w
            .iter_mut()
            .zip(&delta_w)
            .for_each(|(w, var_w)| *w += var_w);
        self.delta_w = delta_w;

        // Calcular momento do pendor
        let m_b = beta * self.delta_b;

        // Atualizar pendor
        let b_var: f64 = escalar + m_b;
        self.b += b_var;
        self.delta_b = b_var;
    }
}

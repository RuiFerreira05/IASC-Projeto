use std::assert_eq;

use crate::{neuronio::Neuronio, phi::Phi};

/**
Representação de uma camada de uma rede neuronal multicamada.

Uma rede neuronal artificial multicamada é composta por três tipos de camadas:
- Camada de entrada: Regista e disponibiliza os dados de entrada, implementando
  a função identidade f(x_i) = x_i com vetor de saída y = x.
- Camada escondida: Processa os dados recebidos da camada anterior através de neurónios
  interligados. Aplica transformações não lineares aos dados projetando-os em espaços de maior
  dimensionalidade, onde dados não separáveis linearmente se tornam separáveis.
- Camada de saída: Produz as saídas da rede com base no processamento das camadas anteriores.

Em termos matriciais, cada camada n (escondida ou de saída) é representada por:
- Matriz W^n de pesos das ligações dos respetivos neurónios com os neurónios da camada anterior.
- Vetor b^n de pendores dos neurónios.
- Função de ativação phi^n.

O processamento da camada compreende:
1. Receção das entradas: x^n = y^(n-1).
2. Cálculo da soma de ativação: h^n = x^n * W^n + b^n, onde h_k^n = somatório(x_j^n * w_jk^n) + b_k^n.
3. Aplicação da função de ativação: y^n = phi^n(h^n).
4. Disponibilização das saídas y^n para a próxima camada da rede.
*/
pub enum Camada {
    /// Camada de entrada da rede: apenas regista e disponibiliza os dados de entrada (f(x_i) = x_i).
    Entrada {
        /// Dimensão de saída ds (número de saídas da camada, equivalente à dimensão de entrada x).
        ds: usize,
        /// Vetor de saída y, onde y = x.
        y: Vec<f64>,
    },

    /// Camada densa (completamente ligada): todos os neurónios recebem todas as saídas da camada anterior.
    Densa {
        /// Dimensão de entrada de (número de ligações de entrada por neurónio).
        de: usize,
        /// Dimensão de saída ds (número de neurónios da camada).
        ds: usize,
        /// Função de ativação phi partilhada pelos neurónios da camada.
        phi: Phi,
        /// Vetor de neurónios da camada.
        neuronios: Vec<Neuronio>,
    },
}

impl Camada {
    /**
    Inicia uma camada de entrada com dimensão de saída ds.

    Valores iniciais: y = [0 para ds].
    */
    pub fn iniciar_entrada(ds: usize) -> Self {
        Camada::Entrada {
            ds,
            y: vec![0.0; ds],
        }
    }

    /**
    Inicia uma camada densa especificando a dimensão de entrada de, a dimensão de saída ds e a função de ativação phi.

    Cria ds neurónios inicializados aleatoriamente com de entradas cada:
    neuronios = [Neuronio(de, phi) para ds].
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
    Inicia uma camada densa com configurações específicas para a matriz de pesos W e vetor de pendores b.
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
    Propaga a entrada x gerando a saída da camada y.

    - Na camada de entrada: armazena e propaga diretamente a entrada (y = x).
    - Na camada densa: propaga a entrada através de cada um dos seus neurónios:
      y = [neuronio.propagar(x) para neuronio em neuronios].
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
    Retorna o vetor de saídas y da camada relativo à última propagação.
    */
    pub fn y(&self) -> Vec<f64> {
        match self {
            Camada::Entrada { y, .. } => y.clone(),
            Camada::Densa { neuronios, .. } => {
                neuronios.iter().map(|neuronio| neuronio.y).collect()
            }
        }
    }

    /**
    Retorna a dimensão de saída ds (número de saídas da camada).
    */
    pub fn ds(&self) -> usize {
        match self {
            Camada::Entrada { ds, .. } => *ds,
            Camada::Densa { ds, .. } => *ds,
        }
    }

    /**
    Retorna uma referência para o vetor de neurónios da camada.
    */
    pub fn neuronios(&self) -> &[Neuronio] {
        match self {
            Camada::Entrada { .. } => &[],
            Camada::Densa { neuronios, .. } => neuronios.as_slice(),
        }
    }

    /**
    Adapta os pesos e pendores dos neurónios da camada com base no vetor de erro delta^n.

    Itera sobre os neurónios j = 1 até ds da camada executando:
    neuronio[j].adaptar(delta_j^n, y^(n-1), alpha, beta)
    */
    pub fn adaptar(&mut self, delta: &[f64], y_anterior: &[f64], alpha: f64, beta: f64) {
        match self {
            Camada::Entrada { .. } => {}
            Camada::Densa { neuronios, .. } => {
                neuronios
                    .iter_mut()
                    .zip(delta)
                    .for_each(|(neuronio_j, delta_j)| {
                        neuronio_j.adaptar(*delta_j, y_anterior, alpha, beta)
                    })
            }
        }
    }
}

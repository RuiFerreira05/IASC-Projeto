/**
Funções de ativação de um neurónio artificial.

A função de ativação y = phi(h) é responsável por determinar a saída do neurónio
com base na sua soma de ativação h, resultante do produto escalar das entradas e dos
pesos sinápticos somado com o pendor b (h = w . x + b).

Objetivos da função de ativação:
- Introduzir não linearidade na resposta do neurónio, possibilitando a discriminação
  entre padrões de dados e a definição de fronteiras de separação não lineares.
- Limitar a saída do neurónio a um intervalo de valores determinado, evitando que a
  saída cresça indefinidamente e prevenindo problemas de estabilidade na rede.

A função de ativação determina o tipo de fronteira de separação no espaço das entradas:
transição abrupta (classificação booleana de verdadeiro ou falso) ou transição progressiva
(grau de pertença a uma classe de dados).
A sua derivada phi'(h) é relevante no contexto da aprendizagem, influenciando a convergência
do treino e a capacidade de generalização da rede.
*/
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Phi {
    /**
    Função Degrau (Step).

    Produz uma transição abrupta: o valor 1 se a soma de ativação h for maior ou igual a 0,
    e 0 no caso contrário (h < 0).
    Determina uma classificação booleana (verdadeiro ou falso).
    É uma função binária e não diferenciável.
    */
    Degrau,

    /**
    Função Linear por troços (Rectified Linear Unit - ReLU).

    Produz o valor h se este for positivo (h > 0), e 0 caso contrário (h <= 0).
    É uma função linear por partes que possibilita um cálculo eficiente, sendo diferenciável
    exceto na origem.
    */
    Relu,

    /**
    Função Sigmoide.

    Produz uma transição progressiva com valores no intervalo [0, 1], dada por:
    phi(h) = 1 / (1 + exp(-h))
    É uma função contínua e diferenciável, adequada para codificação binária onde a saída
    expressa uma probabilidade ou grau de pertença. Não admite valores de saída negativos.
    */
    Sigmoide,

    /**
    Função Tangente Hiperbólica (tanh).

    Produz uma transição progressiva com valores no intervalo [-1, 1], dada por:
    phi(h) = tanh(h)
    É uma função diferenciável que admite valores de saída negativos, sendo adequada
    para codificação bipolar (onde as saídas assumem valores como -1 e +1).
    */
    Tanh,
}

impl Phi {
    /**
    Aplica a função de ativação phi(h) à soma de ativação h, produzindo a saída y.
    */
    pub fn aplicar(&self, h: f64) -> f64 {
        match self {
            Phi::Degrau => {
                if h >= 0.0 {
                    1.0
                } else {
                    0.0
                }
            }

            Phi::Relu => {
                if h >= 0.0 {
                    h
                } else {
                    0.0
                }
            }

            Phi::Sigmoide => 1.0 / (1.0 + (-h).exp()),

            Phi::Tanh => h.tanh(),
        }
    }

    /**
    Calcula o valor da derivada da função de ativação em ordem à soma de ativação:
    phi'(h) = d phi(h) / dh = dy / dh.

    A derivada é relevante no contexto do algoritmo de retropropagação para calcular
    a variação do erro em relação aos parâmetros:
    - Degrau: por não ser diferenciável, assume-se derivada nula.
    - ReLU: derivada 1 para h > 0, e 0 para h <= 0.
    - Sigmoide: phi'(h) = phi(h) * (1 - phi(h)) = y * (1 - y).
    - Tangente Hiperbólica: phi'(h) = 1 - tanh(h)^2 = 1 - y^2.
    */
    pub(crate) fn derivar(&self, h: f64) -> f64 {
        match self {
            Phi::Degrau => 0.0,
            Phi::Relu => {
                if h <= 0.0 {
                    0.0
                } else {
                    1.0
                }
            }
            Phi::Sigmoide => {
                let phi_h = self.aplicar(h);

                phi_h * (1.0 - phi_h)
            }
            Phi::Tanh => 1.0 - h.tanh().powi(2),
        }
    }
}

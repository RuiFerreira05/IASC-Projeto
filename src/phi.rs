/**
O seguinte enum pretende representar todos as funções de ativação possiveis de um neurónio, que presentemente consistem da função de degrau e de tangente
*/
#[derive(Debug, Clone, Copy)]
pub enum Phi {
    /// A função de ativação degrau (step) produz o valor 1 para todas as somas de ativação positivas (ou zero), e 0 para todos as somas de ativação negativas
    Degrau,

    /// A função de ativação Relu (Rectified Linear Unit) reproduz a soma de ativação quando esta é positiva, e 0 quando esta é negativa
    Relu,

    /// A função de ativação sigmoide calcula a sigmoide da soma de ativação
    Sigmoide,

    /// A função de ativação Tangente Hiperbólica produz a tangente hiperbólica da soma de ativação
    Tanh,
}

impl Phi {
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

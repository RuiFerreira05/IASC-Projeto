/**
O seguinte enum pretende representar todos as funções de ativação possiveis de um neurónio, que presentemente consistem da função de degrau e de tangente
*/
#[derive(Debug, Clone, Copy)]
pub enum Phi {
    Degrau,
    Tan,
}

impl Phi {
    pub fn aplicar(&self, h: f64) -> f64 {
        match self {
            // A função de ativação degrau (step) produz o valor 1 para todas as somas de ativação positivas (ou zero), e 0 para todos as somas de ativação negativas
            Phi::Degrau => {
                if h >= 0.0 {
                    1.0
                } else {
                    0.0
                }
            }
            // A função de ativação Tan produz a tangente da soma de ativação
            Phi::Tan => h.tanh(),
        }
    }
}

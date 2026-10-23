// Biblioteca de Redes Neuronais Artificiais
//
// Implementação de uma arquitetura de redes neuronais artificiais multicamada (Multilayer Perceptron - MLP),
// com suporte para retropropagação do erro (backpropagation) e método de descida de gradiente com termo de momento.
//
// Fundamentação Teórica:
//
// As redes neuronais artificiais (RNA) constituem um modelo computacional inspirado na estrutura e funcionamento
// dos sistemas nervosos biológicos (em particular o cérebro), caracterizadas por processamento distribuído massivo
// e capacidade de aprendizagem a partir de exemplos.
//
// Principais características:
// - Aprendizagem com base em exemplos: Capacidade de melhorar o desempenho com a experiência através de dados de treino.
// - Capacidade de generalização: Aplicação do conhecimento adquirido a novos dados não observados durante o treino.
// - Processamento distribuído: Processamento paralelo da informação em múltiplas unidades interligadas (neurónios).
// - Tolerância a falhas e robustez: Resiliência a perturbações e dados incompletos devido à distribuição do conhecimento na rede.
//
// Módulos da Biblioteca:
// - neuronio: Unidade elementar de processamento inspirada no modelo biológico (dendrites, soma, axónio e sinapses).
// - phi: Funções de ativação não lineares (phi) e respetivas derivadas analíticas (phi').
// - camada: Camadas de entrada e camadas densas (completamente ligadas) interligadas.
// - rede: Topologia multicamada com suporte para propagação direta (feedforward), retropropagação e treino supervisionado.

pub mod camada;
pub mod neuronio;
pub mod phi;
pub mod rede;

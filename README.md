# Projeto - Rede Neuronal (IASC)

Implementação de uma biblioteca de redes neuronais artificiais multicamada (*Multi-Layer Perceptron*) em Rust, desenvolvida no âmbito da unidade curricular de Inteligência Artificial e Sistemas Cognitivos (Mestrado).

## Módulos Expostos

A biblioteca expõe quatro módulos principais através de `src/lib.rs`:

- **`phi`**: Modela as funções de ativação dos neurónios através do enum `Phi` (`Degrau`, `Relu`, `Sigmoide` e `Tanh`), disponibilizando o método `aplicar(h)` para calcular o valor de ativação a partir da soma ponderada (soma de ativação).
- **`neuronio`**: Modela a unidade básica da rede através do struct `Neuronio`. Gere os pesos (`w`), o pendor (`b`), a soma de ativação (`h`) e a saída (`y`), suportando inicialização aleatória (`iniciar`), inicialização manual de pesos e pendores (`iniciar_manual`) e a função de transferência via `propagar`.
- **`camada`**: Modela a arquitetura das camadas através do enum `Camada`, com variantes para camada de entrada (`Entrada`) e camada densa totalmente ligada (`Densa`). É responsável pela orquestração dos neurónios e pela propagação de vetores a nível de camada.
- **`rede`**: Modela a estrutura global da rede neuronal multicamada através do struct `RedeNeuronal`, encadeando uma sequência de camadas (`Vec<Camada>`). Suporta inicialização automática por topologia (`iniciar`), inicialização manual com parâmetros pré-definidos (`iniciar_manual`), propagação direta (`propagar`) e previsões em lote (`prever`).

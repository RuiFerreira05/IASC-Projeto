# Projeto - Rede Neuronal (IASC)

Implementação de uma biblioteca de redes neuronais artificiais multicamada (*Multi-Layer Perceptron*) em Rust, desenvolvida no âmbito da unidade curricular de Inteligência Artificial e Sistemas Cognitivos (Mestrado).

## Módulos Expostos

A biblioteca expõe quatro módulos principais através de `src/lib.rs`:

- **`phi`**: Modela as funções de ativação dos neurónios através do enum `Phi` (`Degrau`, `Relu`, `Sigmoide` e `Tanh`), disponibilizando o método `aplicar(h)` para calcular o valor de ativação a partir da soma ponderada (soma de ativação).
- **`neuronio`**: Modela a unidade básica da rede através do struct `Neuronio`. Gere os pesos (`w`), o pendor (`b`), a soma de ativação (`h`) e a saída (`y`), suportando inicialização aleatória (`iniciar`), inicialização manual de pesos e pendores (`iniciar_manual`) e a função de transferência via `propagar`.
- **`camada`**: Modela a arquitetura das camadas através do enum `Camada`, com variantes para camada de entrada (`Entrada`) e camada densa totalmente ligada (`Densa`). É responsável pela orquestração dos neurónios e pela propagação de vetores a nível de camada.
- **`rede`**: Modela a estrutura global da rede neuronal multicamada através do struct `RedeNeuronal`, encadeando uma sequência de camadas (`Vec<Camada>`). Suporta inicialização automática por topologia (`iniciar`), inicialização manual com parâmetros pré-definidos (`iniciar_manual`), propagação direta (`propagar`) e previsões em lote (`prever`).

## Como verificar o funcionamento da aplicação

A seguinte biblioteca contém um ficheiro `main.rs` com a aplicação da biblioteca sob o conjunto de dados MNIST (banco de dados com digitos manuscritos frequentemente usada para o treino de redes neuronais). A execução desta aplicação dá-se de duas formas possiveis:

### Com o toolchain rust instalado (ou só o cargo)

Executar o comando:

```bash
cargo run
```

(ou opcionalmente, `cargo run --release`, para um execução x10 mais rápida com otimizações do compilador)

### Sem o toolchain rust instalado

Executar diretamente o executável pré-compilado presente da diretoria `target`.

- Para linux: `target\release\projeto`
- Para Windows: `target\release\projeto.exe`

## Como configurar o exemplo

Os parametros do exemplo são completamente configuraveis e podem ser alterados comentando a linha 24 do ficheiro `main.rs`, e descomentando a linha 7 a 22, editandos os campos do struct com os parametros desejados (Os campos já presentes representam os valores _default_ do metodo `padrao()`).

Após serem configurados os campos é necessário recompilar o projeto, que pode ser feito através do comando:

```bash
cargo build

# Ou diretamente
cargo run
```

(para ambos os comandos pode ser adicionada a flag `--release` para uma versão x10 mais rápida com otimizações do compilador)

## Como correr os modulos de teste

Para realizar build da biblioteca é necessário o toolchain rust disponibilizado pela package "rustup", ou instalado separadamente.
Com o toolchain presente no computador, basta correr o comando abaixo para executar os módulos de testes presentes em `tests/`:

```bash
cargo test
```

## Como aceder à documentação

A documentação do projeto encontra-se disponivel em formato html na diretoria `target/doc/projeto`, através da geração automatica de documentação do rust.
Em caso da documentação não se encontrar presente, esta pode ser facilmente gerada com o comando (assumindo a presença do toolchain rust):

```bash
cargo doc
```

(opcionalmente, pode ser adicionada a flag `--open` para abrir a documentação do projeto automaticamente no browser pre-definido, após compilação do html)

## Uso de IA

IA foi utilizada para a geração da suites de testes presente em `tests/`.
Adicionalmente, foi utilizado IA para a documentação da aplicação em `main.rs`.
A implementação da biblioteca não teve qualquer suporte de Inteligência Artificial generativa e o código que compõe a lógica da biblioteca é 100% humano.

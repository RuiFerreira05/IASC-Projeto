# Projeto - Aprendizagem por Reforço (IASC)

Implementação de uma biblioteca de aprendizagem por reforço (*Reinforcement Learning*) baseada em métodos de Diferença Temporal (*Temporal Difference* - TD) e controlo *on-policy* (algoritmo SARSA) em Rust, desenvolvida no âmbito da unidade curricular de Inteligência Artificial e Sistemas Cognitivos (Mestrado).

## Módulos Expostos

A biblioteca expõe cinco módulos principais através de `src/lib.rs`:

- **`ambiente`**: Modela os elementos fundamentais do ciclo de interação agente-ambiente. Define os tipos fundamentais `Estado` e `Acao`, bem como a trait `Ambiente` que formaliza o ciclo de vida episódico através dos métodos `iniciar()` (reinicialização do episódio) e `passo(a)` (aplicação da ação, retornando o próximo estado `s'`, a recompensa instantânea `r` e o indicador de terminação `final`).
- **`memoria`**: Modela a memória de aprendizagem tabular da função de valor de ação Q(s, a) ≈ E[G(s, a)] através do struct `Memoria`. Suporta inicialização padrão neutra (`iniciar_default`), valor por omissão configurável para estados não observados (`iniciar`), consulta da estimativa de utilidade (`q(&s, a)`) e atualização pontual (`actualizar(s, a, q)`).
- **`politica`**: Modela a estratégia de seleção de ações epsilon-greedy através do struct `Greedy`, gerindo o compromisso entre exploração (*exploration*) e aproveitamento (*exploitation*). Inclui seleção estocástica de ação exploratória uniforme (`accao_exploracao`), identificação de ação sôfrega/gulosa com desempate aleatório não-tendencioso (`accao_sofrega`), seleção segundo a regra epsilon-greedy (`selecionar_acao`) e inicialização com valor de referência epsilon = 0.1 (`iniciar_default`).
- **`aprendizagem`**: Modela o algoritmo de aprendizagem por reforço por diferença temporal SARSA (*State-Action-Reward-State-Action*) através do struct `Sarsa`. Calcula o retorno estimado alvo J distinguindo passos com continuidade (J = r + gama * Q(s', a')) de passos terminais (J = r), atualizando incrementalmente o valor Q(s, a) com base na taxa de aprendizagem alfa (`aprender`). Disponibiliza inicialização com valores canónicos alfa = 0.1 e gama = 0.9 (`iniciar_default`).
- **`agente`**: Modela a entidade autónoma decisora através do struct `Agente`, integrando a memória tabular, a política de seleção de ações, o algoritmo de aprendizagem e o espaço de ações admissíveis. Suporta inicialização manual com componentes configurados (`iniciar`), inicialização rápida com configurações padrão (`iniciar_default`), seleção de ação (`selecionar_acao`) e orquestração do ciclo completo de treino por episódios (`treinar_episodio`).

## Como verificar o funcionamento da aplicação

A biblioteca inclui um ficheiro `main.rs` com uma aplicação prática de demonstração da biblioteca num mundo de grelha 2D (*Grid World*) de dimensões 6x6, contendo múltiplos obstáculos internos e uma meta terminal.

A execução pode ser efetuada das seguintes formas:

### Com o toolchain Rust instalado (via Cargo)

Executar o comando na raiz do projeto:

```bash
cargo run
```

(opcionalmente, pode ser utilizada a flag `--release` para compilação com otimizações: `cargo run --release`)

### Sem o toolchain Rust instalado

Executar diretamente o binário pré-compilado presente na pasta `target`:

- Para Linux: `target/release/iasc-p2`
- Para Windows: `target\release\iasc-p2.exe`

## Exemplo da aplicação

Um exemplo detalhado com o output visual da grelha inicial e as tabelas estatísticas de progresso e convergência do agente encontra-se disponível em `OUTPUT_EXEMPLO.md`.

## Como configurar o exemplo

Os parâmetros e hiperparâmetros da simulação são centralizados no struct `Configuracao` em `src/main.rs`. Os campos configuráveis incluem:

- `largura` e `altura`: Dimensões da grelha 2D;
- `inicio`: Coordenadas (x, y) da posição de partida do agente;
- `meta`: Coordenadas (x, y) da célula objetivo / terminal;
- `obstaculos`: Vetor de pares (x, y) com as barreiras intransponíveis do labirinto;
- `n_episodios`: Número total de episódios de treino;
- `max_passos_episodio`: Limite máximo de passos admitidos por episódio;
- `alfa`: Taxa de aprendizagem do algoritmo SARSA (alfa);
- `gama`: Fator de desconto temporal das recompensas futuras (gama);
- `epsilon`: Probabilidade de exploração da política epsilon-greedy (epsilon);
- `tamanho_bloco`: Frequência de agregação das métricas estatísticas em tabela.

Após editar os valores pretendidos em `Configuracao::padrao()`, recompilar e executar com:

```bash
cargo run
```

## Como correr os módulos de teste

Para verificar a correção funcional da biblioteca e dos seus componentes, executar a suite de testes unitários e de integração através de:

```bash
cargo test
```

## Como aceder à documentação

A documentação da biblioteca (com fundamentação teórica detalhada, equações e deduções dos algoritmos) pode ser gerada em formato HTML via `rustdoc`:

```bash
cargo doc
```

Para compilar e abrir automaticamente a documentação no browser padrão:

```bash
cargo doc --open
```

## Uso de IA

IA foi utilizada para o apoio na elaboração da documentação técnica e formatação da apresentação dos resultados tabulares em `main.rs`, bem como na geração complementar da suite de testes presente em `tests/`.

A implementação da biblioteca e a estruturação lógica dos seus algoritmos foi desenvolvida inteiramente pelo estudante.

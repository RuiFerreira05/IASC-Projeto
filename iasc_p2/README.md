# Projeto - Aprendizagem por Reforço (IASC)

Implementação de uma biblioteca de aprendizagem por reforço (*Reinforcement Learning*) baseada em métodos de Diferença Temporal (*Temporal Difference* - TD), suportando controlo *on-policy* (algoritmo SARSA) e *off-policy* (algoritmo Q-Learning) em Rust, desenvolvida no âmbito da unidade curricular de Inteligência Artificial e Sistemas Cognitivos (Mestrado).

*Repositório Github:* https://github.com/RuiFerreira05/IASC-Projeto

## Módulos Expostos

A biblioteca expõe cinco módulos principais através de `src/lib.rs`:

- **`ambiente`**: Modela os elementos fundamentais do ciclo de interação agente-ambiente. Define os tipos fundamentais `Estado` e `Acao`, bem como a trait `Ambiente` que formaliza o ciclo de vida episódico através dos métodos `iniciar()` (reinicialização do episódio) e `passo(a)` (aplicação da ação, retornando o próximo estado `s'`, a recompensa instantânea `r` e o indicador de terminação `fim`).
- **`memoria`**: Modela a memória de aprendizagem tabular da função de valor de ação Q(s, a) ≈ E[G(s, a)] através do struct `Memoria`. Suporta inicialização padrão neutra (`iniciar_default`), valor por omissão configurável para estados não observados (`iniciar`), consulta da estimativa de utilidade (`q(&s, a)`) e atualização pontual (`actualizar(s, a, q)`).
- **`politica`**: Define a abstração de estratégias de seleção de ações através da trait `Politica` (`selecionar_acao`). Disponibiliza a implementação da política epsilon-greedy através do struct `Greedy`, gerindo o compromisso entre exploração (*exploration*) e aproveitamento (*exploitation*). Inclui seleção estocástica de ação exploratória uniforme (`accao_exploracao`), identificação de ação sôfrega/gulosa com desempate aleatório não-tendencioso (`accao_sofrega`), seleção segundo a regra epsilon-greedy (`selecionar_acao`), inicialização parametrizada (`iniciar`) e valor de referência epsilon = 0.1 (`iniciar_default`).
- **`aprendizagem`**: Define a abstração de algoritmos de aprendizagem temporal através da trait `Aprendizagem` (`aprender`) com base na estrutura de transição `Transicao` (s, a, r, s', a', fim, accoes). Disponibiliza duas abordagens canónicas de TD control:
  - **`Sarsa`** (*On-Policy*): Calcula o retorno estimado alvo J com base na ação a' efetivamente selecionada pela política atual (J = r + gama * Q(s', a') em passos contínuos e J = r em estados terminais).
  - **`QLearning`** (*Off-Policy*): Calcula o retorno estimado alvo J com base no valor máximo entre as ações admissíveis no estado seguinte (J = r + gama * max Q(s', a*) em passos contínuos e J = r em estados terminais).
- **`agente`**: Modela a entidade autónoma decisora através do struct genérico `Agente<P: Politica, A: Aprendizagem>`, integrando a memória tabular, a política de seleção de ações, o algoritmo de aprendizagem e o espaço de ações admissíveis. Suporta inicialização com componentes configurados (`iniciar`), seleção de ação (`selecionar_acao`) e orquestração do ciclo completo de treino por episódios (`treinar_episodio`).

## Exemplo de Utilização

A arquitetura modular da biblioteca permite instanciar e combinar livremente diferentes políticas e algoritmos de aprendizagem:

```rust
use iasc_p2::{
    agente::Agente,
    ambiente::{Acao, Ambiente},
    aprendizagem::{q_learning::QLearning, sarsa::Sarsa},
    memoria::Memoria,
    politica::greedy::Greedy,
};

// 1. Espaço de ações admissíveis no ambiente
let accoes = vec![Acao(0), Acao(1), Acao(2), Acao(3)];

// 2. Componentes modulares do agente
let memoria = Memoria::iniciar_default();
let politica = Greedy::iniciar_default(); // epsilon = 0.1
let algoritmo = QLearning::iniciar(0.1, 0.9); // alfa = 0.1, gama = 0.9 (ou Sarsa::iniciar(0.1, 0.9))

// 3. Inicialização e treino do agente
let mut agente = Agente::iniciar(memoria, politica, algoritmo, accoes);
// agente.treinar_episodio(&mut ambiente);
```

## Como correr os testes

A biblioteca possui uma suite de testes unitários que cobre todos os módulos e componentes (incluindo transições terminais e não terminais de SARSA e Q-Learning, seleção de ações da política epsilon-greedy e ciclo de treino do agente).

Para executar os testes a partir da pasta `iasc_p2`:

```bash
cargo test
```

Ou a partir da raiz do repositório:

```bash
cargo test --manifest-path iasc_p2/Cargo.toml
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

IA foi utilizada para apoio na elaboração da documentação técnica e na geração complementar da suite de testes unitários dos módulos.

A implementação da biblioteca e a estruturação lógica dos seus algoritmos foi desenvolvida inteiramente pelo estudante.

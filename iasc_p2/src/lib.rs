// Biblioteca de Aprendizagem por Reforço (Reinforcement Learning)
//
// Implementação de uma arquitetura de aprendizagem por reforço baseada em métodos de Diferença Temporal (Temporal Difference - TD),
// com controlo on-policy (algoritmo SARSA) e off-policy (algoritmo Q-Learning), e estratégias de seleção de ações com compromisso de exploração e aproveitamento (epsilon-greedy).
//
// Fundamentação Teórica:
//
// A aprendizagem por reforço constitui um paradigma de aprendizagem computacional focado na aprendizagem de comportamentos ("como fazer"),
// onde um agente aprende a tomar decisões sequenciais através da sua interação direta com o ambiente por tentativa e erro, recebendo
// uma medida de desempenho sob a forma de sinal de recompensa (reforço).
//
// Principais conceitos e características:
// - Problema da Decisão Sequencial: Para cada estado s, cada decisão de ação a pode desencadear múltiplas ramificações
//   de estados e decisões futuras. O valor (utilidade) das ações pode não ser imediato, revelando-se diferido no tempo.
// - Incerteza e Ambientes Não-Deterministas: A evolução dos estados pode estar sujeita a incerteza, sendo os modelos de
//   transição T(s, a, s') e de recompensa R(s, a, s') dos Processos de Decisão de Markov (MDP) frequentemente desconhecidos a priori.
// - Recompensa e Retorno: A recompensa r representa o reforço instantâneo, enquanto o retorno G_t expressa o efeito cumulativo
//   das recompensas ao longo do tempo. Com desconto temporal (fator gamma in [0, 1]), o retorno futuro é ponderado:
//   G_t = r_t + gamma * r_{t+1} + gamma^2 * r_{t+2} + ... = r_t + gamma * G_{t+1}.
// - Princípio da Decisão Óptima e Equação de Bellman: A decisão ótima maximiza a utilidade esperada em cada estado.
//   A política ótima pi*(s) seleciona a ação de maior utilidade: pi*(s) = argmax_a U(s, a).
// - Estimativa de Valor de Ação Q(s, a): Aproximação do retorno esperado E[G(s, a)] através da experiência incremental.
// - Aprendizagem por Diferença Temporal (TD): Atualização iterativa da estimativa de valor baseada no erro de estimativa
//   (diferença temporal entre instantes sucessivos): Q(s, a) <- Q(s, a) + alpha * [J - Q(s, a)], onde J é o valor alvo.
// - Dilema Exploração vs. Aproveitamento (Exploration vs. Exploitation): Necessidade de equilibrar a exploração de novas ações
//   para aquisição de experiência e o aproveitamento do conhecimento já adquirido para maximizar a recompensa (política epsilon-greedy).
// - Controlo On-Policy (SARSA): O algoritmo SARSA (State-Action-Reward-State-Action) utiliza a mesma política tanto para selecionar
//   a ação a executar no ambiente como para estimar o retorno futuro (J = r + gamma * Q(s', a')).
// - Controlo Off-Policy (Q-Learning): O algoritmo Q-Learning utiliza a política atual para selecionar a ação no ambiente,
//   mas estima o retorno futuro considerando a ação ótima que maximiza o valor no próximo estado (J = r + gamma * max_a Q(s', a)).
//
// Módulos da Biblioteca:
// - ambiente: Abstração do ambiente, estados (s), ações (a) e ciclo de interação do agente.
// - memoria: Memória de aprendizagem tabular para representação da função de valor de ação Q(s, a).
// - politica: Estratégias de seleção de ação (trait Politica), incluindo a política epsilon-greedy (struct Greedy) com desempate estocástico.
// - aprendizagem: Abstração de algoritmos de aprendizagem por diferença temporal (trait Aprendizagem), incluindo SARSA (on-policy) e Q-Learning (off-policy) com tratamento de estados terminais.
// - agente: Agente autónomo genérico (Agente<P, A>) que integra memória, política e aprendizagem para treino e interação por episódios.

pub mod agente;
pub mod ambiente;
pub mod aprendizagem;
pub mod memoria;
pub mod politica;

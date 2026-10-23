# Execução da biblioteca sobre os dados do MNIST

## Configuração

O seguinte output representa a execução do ficheiro main.rs, configurado da seguinte forma:

```rust
tam_treino: 10_000,
tam_teste: 1_000,
forma: vec![784, 64, 10],
phi: Phi::Sigmoide,
n_epocas: 10,
alpha: 0.1,
beta: 0.5,
epsilon_max: 1e-5,
avaliar_a_cada_n_epocas: 1,
```

## Output

```txt
========================================================================
          CLASSIFICAÇÃO DE DÍGITOS MNIST COM A BIBLIOTECA
========================================================================
Configuração ativa:
  - Topologia da Rede: [784, 64, 10]
  - Função de Ativação: Sigmoide
  - Subconjunto de Treino: 10000 amostras
  - Subconjunto de Teste: 1000 amostras
  - Hiperparâmetros: alpha = 0.1, beta = 0.5, épocas = 10
------------------------------------------------------------------------

A carregar o dataset MNIST a partir de 'data'...
Dados carregados com sucesso: 10000 amostras de treino, 1000 amostras de teste.

A inicializar a rede neuronal via 'RedeNeuronal::iniciar'...

A iniciar o treino da rede...
+--------+---------------+-------------------+----------------------+
| Época  |  Tempo Gasto  |     MSE Médio     | Exatidão Teste (%)   |
+--------+---------------+-------------------+----------------------+
|   1    |      1.26s    |        0.043440   |         81.00%       |
|   2    |      1.26s    |        0.026157   |         82.50%       |
|   3    |      1.25s    |        0.017848   |         92.30%       |
|   4    |      1.25s    |        0.011188   |         92.80%       |
|   5    |      1.24s    |        0.009342   |         93.60%       |
|   6    |      1.24s    |        0.008126   |         93.90%       |
|   7    |      1.25s    |        0.007227   |         93.50%       |
|   8    |      1.28s    |        0.006547   |         94.00%       |
|   9    |      1.30s    |        0.005986   |         93.90%       |
|   10   |      1.29s    |        0.005500   |         94.10%       |
+--------+---------------+-------------------+----------------------+

A calcular relatório de desempenho detalhado no conjunto de teste...

========================================================================
                    RELATÓRIO DE DESEMPENHO NO TESTE
========================================================================
Exatidão Global: 94.10% (941/1000 amostras classificadas corretamente)

Taxa de Acerto por Classe:
+--------+-----------+---------------+--------------------+
| Dígito |   Total   | Classificados | Exatidão da Classe |
+--------+-----------+---------------+--------------------+
|    0   |    97     |       93      |       95.88%       |
|    1   |    112    |       106     |       94.64%       |
|    2   |    97     |       97      |      100.00%       |
|    3   |    103    |       97      |       94.17%       |
|    4   |    106    |       96      |       90.57%       |
|    5   |    95     |       86      |       90.53%       |
|    6   |    103    |       96      |       93.20%       |
|    7   |    97     |       92      |       94.85%       |
|    8   |    86     |       79      |       91.86%       |
|    9   |    104    |       99      |       95.19%       |
+--------+-----------+---------------+--------------------+

Matriz de Confusão (Linhas: Rótulo Real | Colunas: Previsão da Rede):
 Real \ Prev |   0     1     2     3     4     5     6     7     8     9  |  Total
-------------+------------------------------------------------------------+-------
        0    |   93     .     1     .     .     .     1     .     1     1 |     97
        1    |    .   106     2     .     .     1     .     .     3     . |    112
        2    |    .     .    97     .     .     .     .     .     .     . |     97
        3    |    .     .     2    97     .     1     1     1     1     . |    103
        4    |    1     .     2     .    96     .     1     .     5     1 |    106
        5    |    2     .     .     4     .    86     .     .     3     . |     95
        6    |    2     .     2     1     .     .    96     .     2     . |    103
        7    |    1     .     .     .     1     .     .    92     2     1 |     97
        8    |    .     2     .     3     .     .     1     .    79     1 |     86
        9    |    1     .     .     .     1     1     .     .     2    99 |    104
-------------+------------------------------------------------------------+-------

Processo concluído com sucesso.
```

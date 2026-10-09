use std::{path::Path, time::Instant};

use mnist::{Mnist, MnistBuilder};
use projeto::{phi::Phi, rede::RedeNeuronal};

fn main() {
    // let config = Configuracao {
    //     caminho_dados: "data".to_string(),
    //     ficheiro_treino_imgs: "train-images.idx3-ubyte".to_string(),
    //     ficheiro_treino_lbls: "train-labels.idx1-ubyte".to_string(),
    //     ficheiro_teste_imgs: "t10k-images.idx3-ubyte".to_string(),
    //     ficheiro_teste_lbls: "t10k-labels.idx1-ubyte".to_string(),
    //     tam_treino: 10_000,
    //     tam_teste: 1_000,
    //     forma: vec![784, 64, 10],
    //     phi: Phi::Sigmoide,
    //     n_epocas: 10,
    //     alpha: 0.1,
    //     beta: 0.5,
    //     epsilon_max: 1e-5,
    //     avaliar_a_cada_n_epocas: 1,
    // };

    let config = Configuracao::padrao();

    exemplo_mnist(config);
}

/// Estrutura que centraliza todos os parâmetros e hiperparâmetros da experiência.
/// Todos os valores são configuráveis para permitir testes de afinação e experimentação contínua.
pub struct Configuracao {
    /// Diretório onde residem os ficheiros do MNIST descompactados
    pub caminho_dados: String,

    /// Nome do ficheiro das imagens de treino
    pub ficheiro_treino_imgs: String,

    /// Nome do ficheiro dos rótulos de treino
    pub ficheiro_treino_lbls: String,

    /// Nome do ficheiro das imagens de teste
    pub ficheiro_teste_imgs: String,

    /// Nome do ficheiro dos rótulos de teste
    pub ficheiro_teste_lbls: String,

    /// Dimensão do subconjunto de imagens de treino a utilizar (1 até 60.000)
    pub tam_treino: usize,

    /// Dimensão do subconjunto de imagens de teste a utilizar (1 até 10.000)
    pub tam_teste: usize,

    /// Topologia da rede neuronal [entradas, camadas ocultas..., saídas]
    /// Para o MNIST, a entrada é sempre 784 (28x28) e a saída é 10 (dígitos 0 a 9)
    pub forma: Vec<usize>,

    /// Função de ativação a aplicar em todas as camadas da rede
    pub phi: Phi,

    /// Número total de épocas de treino
    pub n_epocas: usize,

    /// Taxa de aprendizagem (learning rate) aplicada na retropropagação
    pub alpha: f64,

    /// Coeficiente de momento (momentum) para aceleração de gradiente
    pub beta: f64,

    /// Erro quadrático máximo admissível para paragem antecipada
    pub epsilon_max: f64,

    /// Frequência de épocas entre avaliações de desempenho no conjunto de teste
    pub avaliar_a_cada_n_epocas: usize,
}

impl Configuracao {
    /// Constrói uma configuração padrão com valores sensatos para experimentação ágil,
    /// detetando automaticamente se os ficheiros em disco usam ponto ou hífen.
    pub fn padrao() -> Self {
        let base = "data";

        // Deteção flexível dos nomes dos ficheiros presentes na diretoria
        let ficheiro_treino_imgs = if Path::new(base).join("train-images.idx3-ubyte").exists() {
            "train-images.idx3-ubyte".to_string()
        } else {
            "train-images-idx3-ubyte".to_string()
        };

        let ficheiro_treino_lbls = if Path::new(base).join("train-labels.idx1-ubyte").exists() {
            "train-labels.idx1-ubyte".to_string()
        } else {
            "train-labels-idx1-ubyte".to_string()
        };

        let ficheiro_teste_imgs = if Path::new(base).join("t10k-images.idx3-ubyte").exists() {
            "t10k-images.idx3-ubyte".to_string()
        } else {
            "t10k-images-idx3-ubyte".to_string()
        };

        let ficheiro_teste_lbls = if Path::new(base).join("t10k-labels.idx1-ubyte").exists() {
            "t10k-labels.idx1-ubyte".to_string()
        } else {
            "t10k-labels-idx1-ubyte".to_string()
        };

        Configuracao {
            caminho_dados: base.to_string(),
            ficheiro_treino_imgs,
            ficheiro_treino_lbls,
            ficheiro_teste_imgs,
            ficheiro_teste_lbls,
            tam_treino: 10_000,
            tam_teste: 1_000,
            forma: vec![784, 64, 10],
            phi: Phi::Sigmoide,
            n_epocas: 10,
            alpha: 0.1,
            beta: 0.5,
            epsilon_max: 1e-5,
            avaliar_a_cada_n_epocas: 1,
        }
    }
}

/// Representa um conjunto estruturado de dados pré-processados prontos a alimentar a rede.
pub struct DadosConjunto {
    /// Vetor de entradas, onde cada elemento é uma imagem achatada com 784 píxeis normalizados em [0.0, 1.0]
    pub entradas: Vec<Vec<f64>>,

    /// Vetor de saídas esperadas, onde cada elemento é um vetor one-hot de dimensão 10
    pub alvos_one_hot: Vec<Vec<f64>>,

    /// Rótulos inteiros reais (0 a 9) para avaliação direta de métricas
    pub rotulos: Vec<usize>,
}

/// Carrega e converte o dataset MNIST a partir dos ficheiros binários em disco.
fn carregar_dados(config: &Configuracao) -> (DadosConjunto, DadosConjunto) {
    println!(
        "A carregar o dataset MNIST a partir de '{}'...",
        config.caminho_dados
    );

    let Mnist {
        trn_img,
        trn_lbl,
        tst_img,
        tst_lbl,
        ..
    } = MnistBuilder::new()
        .base_path(&config.caminho_dados)
        .training_images_filename(&config.ficheiro_treino_imgs)
        .training_labels_filename(&config.ficheiro_treino_lbls)
        .test_images_filename(&config.ficheiro_teste_imgs)
        .test_labels_filename(&config.ficheiro_teste_lbls)
        .training_set_length(config.tam_treino as u32)
        .validation_set_length(0)
        .test_set_length(config.tam_teste as u32)
        .label_format_digit()
        .finalize();

    let treino = processar_conjunto(trn_img, trn_lbl, config.tam_treino);
    let teste = processar_conjunto(tst_img, tst_lbl, config.tam_teste);

    println!(
        "Dados carregados com sucesso: {} amostras de treino, {} amostras de teste.",
        treino.entradas.len(),
        teste.entradas.len()
    );

    (treino, teste)
}

/// Normaliza píxeis [0, 255] -> [0.0, 1.0] e gera a codificação one-hot dos rótulos.
fn processar_conjunto(
    imagens_brutas: Vec<u8>,
    rotulos_brutos: Vec<u8>,
    quantidade: usize,
) -> DadosConjunto {
    let mut entradas = Vec::with_capacity(quantidade);
    let mut alvos_one_hot = Vec::with_capacity(quantidade);
    let mut rotulos = Vec::with_capacity(quantidade);

    for i in 0..quantidade {
        // Fatiamento dos 784 píxeis correspondentes à i-ésima imagem
        let inicio = i * 784;
        let fim = inicio + 784;
        let imagem_normalizada: Vec<f64> = imagens_brutas[inicio..fim]
            .iter()
            .map(|&p| p as f64 / 255.0)
            .collect();

        // Rótulo inteiro e respetiva representação one-hot de 10 posições
        let digito = rotulos_brutos[i] as usize;
        let mut one_hot = vec![0.0; 10];
        if digito < 10 {
            one_hot[digito] = 1.0;
        }

        entradas.push(imagem_normalizada);
        alvos_one_hot.push(one_hot);
        rotulos.push(digito);
    }

    DadosConjunto {
        entradas,
        alvos_one_hot,
        rotulos,
    }
}

/// Estrutura com os resultados estatísticos da avaliação do modelo.
pub struct RelatorioAvaliacao {
    pub total: usize,
    pub acertos: usize,
    pub exatidao_global: f64,
    pub acertos_por_digito: [usize; 10],
    pub total_por_digito: [usize; 10],
    pub matriz_confusao: [[usize; 10]; 10],
}

/// Retorna o índice do maior valor num vetor (predição categórica).
fn argmax(v: &[f64]) -> usize {
    v.iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
        .map(|(indice, _)| indice)
        .unwrap_or(0)
}

/// Avalia o desempenho da rede neuronal sobre um conjunto de teste.
fn avaliar_rede(rede: &mut RedeNeuronal, dados: &DadosConjunto) -> RelatorioAvaliacao {
    let total = dados.entradas.len();
    let mut acertos = 0;
    let mut acertos_por_digito = [0; 10];
    let mut total_por_digito = [0; 10];
    let mut matriz_confusao = [[0; 10]; 10];

    for i in 0..total {
        let x = &dados.entradas[i];
        let digito_real = dados.rotulos[i];

        let saida = rede.propagar(x);
        let digito_previsto = argmax(&saida);

        if digito_real < 10 {
            total_por_digito[digito_real] += 1;
            if digito_previsto < 10 {
                matriz_confusao[digito_real][digito_previsto] += 1;
            }
        }

        if digito_previsto == digito_real {
            acertos += 1;
            if digito_real < 10 {
                acertos_por_digito[digito_real] += 1;
            }
        }
    }

    let exatidao_global = if total > 0 {
        (acertos as f64 / total as f64) * 100.0
    } else {
        0.0
    };

    RelatorioAvaliacao {
        total,
        acertos,
        exatidao_global,
        acertos_por_digito,
        total_por_digito,
        matriz_confusao,
    }
}

/// Apresenta o relatório de desempenho formatado no terminal.
fn imprimir_relatorio(relatorio: &RelatorioAvaliacao) {
    println!("\n========================================================================");
    println!("                    RELATÓRIO DE DESEMPENHO NO TESTE                    ");
    println!("========================================================================");
    println!(
        "Exatidão Global: {:.2}% ({}/{} amostras classificadas corretamente)\n",
        relatorio.exatidao_global, relatorio.acertos, relatorio.total
    );

    println!("Taxa de Acerto por Classe:");
    println!("+--------+-----------+---------------+--------------------+");
    println!("| Dígito |   Total   | Classificados | Exatidão da Classe |");
    println!("+--------+-----------+---------------+--------------------+");
    for d in 0..10 {
        let n_total = relatorio.total_por_digito[d];
        let n_acertos = relatorio.acertos_por_digito[d];
        let taxa = if n_total > 0 {
            (n_acertos as f64 / n_total as f64) * 100.0
        } else {
            0.0
        };
        println!(
            "|   {:^4} | {:^9} |   {:^11} |      {:>6.2}%       |",
            d, n_total, n_acertos, taxa
        );
    }
    println!("+--------+-----------+---------------+--------------------+");

    println!("\nMatriz de Confusão (Linhas: Rótulo Real | Colunas: Previsão da Rede):");
    print!(" Real \\ Prev |");
    for c in 0..10 {
        print!("  {:^3} ", c);
    }
    println!("|  Total");
    println!("-------------+---------------------------------------------------+-------");

    for r in 0..10 {
        print!("      {:^6} |", r);
        for c in 0..10 {
            let contagem = relatorio.matriz_confusao[r][c];
            if contagem > 0 {
                print!(" {:>4} ", contagem);
            } else {
                print!("    . ");
            }
        }
        println!("| {:>6}", relatorio.total_por_digito[r]);
    }
    println!("-------------+---------------------------------------------------+-------");
}

fn exemplo_mnist(config: Configuracao) {
    assert!(
        Path::new(&config.caminho_dados).is_dir(),
        "ERRO: Caminho de dados não é uma diretoria"
    );
    assert!(
        Path::new(&config.caminho_dados)
            .join(&config.ficheiro_teste_imgs)
            .exists(),
        "ERRO: O caminho para as imagens de teste não existe"
    );
    assert!(
        Path::new(&config.caminho_dados)
            .join(&config.ficheiro_teste_lbls)
            .exists(),
        "ERRO: O caminho para as labels de teste não existe"
    );
    assert!(
        Path::new(&config.caminho_dados)
            .join(&config.ficheiro_treino_imgs)
            .exists(),
        "ERRO: O caminho para as imagens de treino não existe"
    );
    assert!(
        Path::new(&config.caminho_dados)
            .join(&config.ficheiro_treino_lbls)
            .exists(),
        "ERRO: O caminho para as labels de treino não existe"
    );

    println!("========================================================================");
    println!("          CLASSIFICAÇÃO DE DÍGITOS MNIST COM A BIBLIOTECA               ");
    println!("========================================================================");

    // 1. Definição da configuração (totalmente parametrizável para afinação)
    let config = config;

    println!("Configuração ativa:");
    println!("  - Topologia da Rede: {:?}", config.forma);
    println!("  - Função de Ativação: {:?}", config.phi);
    println!("  - Subconjunto de Treino: {} amostras", config.tam_treino);
    println!("  - Subconjunto de Teste: {} amostras", config.tam_teste);
    println!(
        "  - Hiperparâmetros: alpha = {}, beta = {}, épocas = {}",
        config.alpha, config.beta, config.n_epocas
    );
    println!("------------------------------------------------------------------------\n");

    // 2. Carregamento e pré-processamento do dataset
    let (dados_treino, dados_teste) = carregar_dados(&config);

    // 3. Inicialização estrita através da biblioteca existente
    println!("\nA inicializar a rede neuronal via 'RedeNeuronal::iniciar'...");
    let mut rede = RedeNeuronal::iniciar(&config.forma, config.phi);

    // 4. Ciclo de treino época a época com telemetria
    println!("\nA iniciar o treino da rede...");
    println!("+--------+---------------+-------------------+----------------------+");
    println!("| Época  |  Tempo Gasto  |     MSE Médio     | Exatidão Teste (%)   |");
    println!("+--------+---------------+-------------------+----------------------+");

    for epoca in 1..=config.n_epocas {
        let inicio_epoca = Instant::now();
        let mut soma_erro = 0.0;

        // Adaptação sequencial de cada amostra de treino
        for i in 0..dados_treino.entradas.len() {
            let x = &dados_treino.entradas[i];
            let y = &dados_treino.alvos_one_hot[i];

            let erro_amostra = rede.adaptar(x, y, config.alpha, config.beta);
            soma_erro += erro_amostra;
        }

        let tempo_epoca = inicio_epoca.elapsed();
        let mse_medio = soma_erro / (dados_treino.entradas.len() as f64);

        // Avaliação no conjunto de teste segundo a frequência configurada
        let str_exatidao =
            if epoca % config.avaliar_a_cada_n_epocas == 0 || epoca == config.n_epocas {
                let relatorio = avaliar_rede(&mut rede, &dados_teste);
                format!("{:>6.2}%", relatorio.exatidao_global)
            } else {
                "   ---   ".to_string()
            };

        println!(
            "| {:^6} |   {:>7.2}s    |    {:>12.6}   |      {:^11}     |",
            epoca,
            tempo_epoca.as_secs_f64(),
            mse_medio,
            str_exatidao
        );

        if mse_medio < config.epsilon_max {
            println!(
                "\nCritério de paragem antecipada atingido (MSE < {}).",
                config.epsilon_max
            );
            break;
        }
    }
    println!("+--------+---------------+-------------------+----------------------+");

    // 5. Avaliação final detalhada
    println!("\nA calcular relatório de desempenho detalhado no conjunto de teste...");
    let relatorio_final = avaliar_rede(&mut rede, &dados_teste);
    imprimir_relatorio(&relatorio_final);

    println!("\nProcesso concluído com sucesso.");
}

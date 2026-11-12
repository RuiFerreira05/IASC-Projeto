use std::{collections::HashSet, time::Instant};

use iasc_p2::{
    agente::Agente,
    ambiente::{Acao, Ambiente, Estado},
    aprendizagem::q_learning::QLearning,
    memoria::Memoria,
    politica::greedy::Greedy,
};

// AÇÕES

pub const ACAO_CIMA: Acao = Acao(0);
pub const ACAO_BAIXO: Acao = Acao(1);
pub const ACAO_ESQUERDA: Acao = Acao(2);
pub const ACAO_DIREITA: Acao = Acao(3);

pub const ACTIONS: [Acao; 4] = [ACAO_CIMA, ACAO_BAIXO, ACAO_ESQUERDA, ACAO_DIREITA];

pub fn nome_acao(a: Acao) -> &'static str {
    match a {
        ACAO_CIMA => "Cima",
        ACAO_BAIXO => "Baixo",
        ACAO_ESQUERDA => "Esquerda",
        ACAO_DIREITA => "Direita",
        _ => "Desconhecida",
    }
}

// CONFIGURAÇÃO

pub struct Configuracao {
    /// Número de colunas da grelha (eixo X)
    pub largura: i32,
    /// Número de linhas da grelha (eixo Y)
    pub altura: i32,
    /// Posição inicial do agente (x, y)
    pub inicio: (i32, i32),
    /// Posição da célula meta / objetivo (x, y)
    pub meta: (i32, i32),
    /// Conjunto de coordenadas contendo obstáculos intransponíveis
    pub obstaculos: Vec<(i32, i32)>,
    /// Número total de episódios a treinar
    pub n_episodios: usize,
    /// Limite máximo de passos permitidos por episódio (evita ciclos infinitos)
    pub max_passos_episodio: usize,
    /// Fator de aprendizagem do Q-Learning: alfa in [0, 1]
    pub alfa: f32,
    /// Fator de desconto temporal: gama in [0, 1]
    pub gama: f32,
    /// Taxa de exploração da política epsilon-greedy: epsilon in [0, 1]
    pub epsilon: f32,
    /// Recompensa instantânea concedida ao alcançar a meta
    pub recompensa_meta: f32,
    /// Penalização instantânea por cada passo regular efetuado
    pub penalizacao_passo: f32,
    /// Dimensão de cada bloco de episódios para agregação e exibição estatística
    pub tamanho_bloco: usize,
}

impl Configuracao {
    /// Retorna os valores padrão da experiência de Grid World (grelha 6x6 com labirinto).
    pub fn padrao() -> Self {
        Self {
            largura: 6,
            altura: 6,
            inicio: (0, 0),
            meta: (5, 5),
            obstaculos: vec![
                (1, 1),
                (1, 2),
                (1, 3),
                (1, 4),
                (3, 5),
                (3, 4),
                (3, 3),
                (3, 2),
                (3, 0),
                (4, 2),
                (5, 4),
            ],
            n_episodios: 600,
            max_passos_episodio: 200,
            alfa: 0.2,
            gama: 0.95,
            epsilon: 0.15,
            recompensa_meta: 100.0,
            penalizacao_passo: -1.0,
            tamanho_bloco: 60,
        }
    }
}

// AMBIENTE

/// Ambiente discreto de navegação 2D em grelha com barreiras intransponíveis.
pub struct MundoGrelha2D {
    largura: i32,
    altura: i32,
    inicio: (i32, i32),
    meta: (i32, i32),
    obstaculos: HashSet<(i32, i32)>,
    pos: (i32, i32),
    passos_atuais: usize,
    max_passos: usize,
    recompensa_meta: f32,
    penalizacao_passo: f32,

    // Métricas do episódio mais recente
    pub ultimo_episodio_passos: usize,
    pub ultimo_episodio_recompensa: f32,
    pub ultimo_episodio_meta_atingida: bool,
}

impl MundoGrelha2D {
    pub fn novo(config: &Configuracao) -> Self {
        let obstaculos = config.obstaculos.iter().copied().collect();

        Self {
            largura: config.largura,
            altura: config.altura,
            inicio: config.inicio,
            meta: config.meta,
            obstaculos,
            pos: config.inicio,
            passos_atuais: 0,
            max_passos: config.max_passos_episodio,
            recompensa_meta: config.recompensa_meta,
            penalizacao_passo: config.penalizacao_passo,
            ultimo_episodio_passos: 0,
            ultimo_episodio_recompensa: 0.0,
            ultimo_episodio_meta_atingida: false,
        }
    }

    pub fn coords_para_estado(pos: (i32, i32)) -> Estado {
        Estado::novo(format!("{},{}", pos.0, pos.1))
    }

    pub fn e_obstaculo(&self, x: i32, y: i32) -> bool {
        self.obstaculos.contains(&(x, y))
    }
}

impl Ambiente for MundoGrelha2D {
    fn iniciar(&mut self) -> Estado {
        self.pos = self.inicio;
        self.passos_atuais = 0;
        self.ultimo_episodio_passos = 0;
        self.ultimo_episodio_recompensa = 0.0;
        self.ultimo_episodio_meta_atingida = false;
        Self::coords_para_estado(self.pos)
    }

    fn passo(&mut self, a: Acao) -> (Estado, f32, bool) {
        self.passos_atuais += 1;

        let (dx, dy) = match a {
            ACAO_CIMA => (0, -1),
            ACAO_BAIXO => (0, 1),
            ACAO_ESQUERDA => (-1, 0),
            ACAO_DIREITA => (1, 0),
            _ => (0, 0),
        };

        let novo_x = self.pos.0 + dx;
        let novo_y = self.pos.1 + dy;

        if novo_x >= 0
            && novo_x < self.largura
            && novo_y >= 0
            && novo_y < self.altura
            && !self.e_obstaculo(novo_x, novo_y)
        {
            self.pos = (novo_x, novo_y);
        }

        let atingiu_meta = self.pos == self.meta;
        let esgotou_passos = self.passos_atuais >= self.max_passos;
        let fim = atingiu_meta || esgotou_passos;

        let recompensa = if atingiu_meta {
            self.recompensa_meta
        } else {
            self.penalizacao_passo
        };

        self.ultimo_episodio_passos = self.passos_atuais;
        self.ultimo_episodio_recompensa += recompensa;
        self.ultimo_episodio_meta_atingida = atingiu_meta;

        (Self::coords_para_estado(self.pos), recompensa, fim)
    }
}

// MÉTRICAS

pub struct MetricasBloco {
    pub bloco_inicio: usize,
    pub bloco_fim: usize,
    pub passos_medios: f32,
    pub recompensa_media: f32,
    pub taxa_sucesso: f32,
    pub tempo_segundos: f64,
}

pub struct PassoTrajetoria {
    pub passo: usize,
    pub pos_atual: (i32, i32),
    pub acao: Acao,
    pub recompensa: f32,
    pub pos_seguinte: (i32, i32),
    pub valor_q: f32,
}

// VISUALIZAÇÃO

fn imprimir_cabecalho(config: &Configuracao) {
    println!("========================================================================");
    println!("        APRENDIZAGEM POR REFORÇO (Q-LEARNING) EM MUNDO DE GRELHA        ");
    println!("========================================================================");
    println!("Configuração da Experiência:");
    println!(
        "  - Dimensões da Grelha : {}x{} células",
        config.largura, config.altura
    );
    println!("  - Célula de Início    : {:?}", config.inicio);
    println!("  - Célula Meta         : {:?}", config.meta);
    println!(
        "  - Número de Barreiras : {} obstáculos intransponíveis",
        config.obstaculos.len()
    );
    println!("  - Espaço de Ações     : [Cima, Baixo, Esquerda, Direita]");
    println!(
        "  - Hiperparâmetros     : alfa = {}, gama = {}, epsilon = {}",
        config.alfa, config.gama, config.epsilon
    );
    println!(
        "  - Recompensas         : meta = +{:.1}, passo = {:.1}",
        config.recompensa_meta, config.penalizacao_passo
    );
    println!(
        "  - Treino              : {} episódios (máx. {} passos/episódio)",
        config.n_episodios, config.max_passos_episodio
    );
    println!("------------------------------------------------------------------------\n");
}

fn imprimir_grelha(config: &Configuracao, caminho: Option<&HashSet<(i32, i32)>>) {
    print!("   ");
    for x in 0..config.largura {
        print!("  {:^3} ", x);
    }
    println!(" (X)");

    print!("   +");
    for _ in 0..config.largura {
        print!("-----+");
    }
    println!();

    for y in 0..config.altura {
        print!("{:>2} |", y);
        for x in 0..config.largura {
            let coord = (x, y);
            let celula = if coord == config.inicio {
                "  S  "
            } else if coord == config.meta {
                "  G  "
            } else if config.obstaculos.contains(&coord) {
                " ### "
            } else if let Some(pontos) = caminho {
                if pontos.contains(&coord) {
                    "  *  "
                } else {
                    "     "
                }
            } else {
                "     "
            };
            print!("{}|", celula);
        }
        println!();

        print!("   +");
        for _ in 0..config.largura {
            print!("-----+");
        }
        println!();
    }
    println!("(Y)\n");
}

fn imprimir_mapa_inicial(config: &Configuracao) {
    println!("Representação do Ambiente Inicial:");
    imprimir_grelha(config, None);
    println!("Legenda:");
    println!("  S   : Início / Agente em {:?}", config.inicio);
    println!("  G   : Meta / Objetivo Terminal em {:?}", config.meta);
    println!(
        " ###  : Parede / Obstáculo intransponível ({} células)",
        config.obstaculos.len()
    );
    println!("      : Célula Livre transitável\n");
}

fn imprimir_tabela_evolucao(metricas: &[MetricasBloco]) {
    println!("========================================================================");
    println!("       TABELA 1: EVOLUÇÃO ESTATÍSTICA DO TREINO (POR BLOCOS)            ");
    println!("========================================================================");
    println!(
        "+-------------------+---------------+-------------------+------------------+---------------+"
    );
    println!(
        "| Bloco Episódios   | Passos Médios | Recompensa Média  | Taxa Sucesso (%) | Tempo Gasto   |"
    );
    println!(
        "+-------------------+---------------+-------------------+------------------+---------------+"
    );
    for m in metricas {
        println!(
            "| {:>6} .. {:<6} | {:>13.1} | {:>17.1} | {:>15.1}% | {:>11.3}s |",
            m.bloco_inicio,
            m.bloco_fim,
            m.passos_medios,
            m.recompensa_media,
            m.taxa_sucesso,
            m.tempo_segundos
        );
    }
    println!(
        "+-------------------+---------------+-------------------+------------------+---------------+\n"
    );
}

fn imprimir_tabela_valores_q(
    agente: &Agente<Greedy, QLearning>,
    config: &Configuracao,
    env: &MundoGrelha2D,
) {
    println!("========================================================================");
    println!("   TABELA 2: ESTIMATIVAS DE VALOR Q(s, a) E POLÍTICA EM ESTADOS-CHAVE   ");
    println!("========================================================================");
    println!(
        "+--------------+-----------+-----------+-----------+-----------+-----------+----------------+"
    );
    println!(
        "| Estado (x,y) |  Q(Cima)  |  Q(Baixo) |  Q(Esq)   |  Q(Dir)   | V(s)=maxQ | Ação Ótima π*  |"
    );
    println!(
        "+--------------+-----------+-----------+-----------+-----------+-----------+----------------+"
    );

    let mut estados_para_exibir = Vec::new();
    for y in 0..config.altura {
        for x in 0..config.largura {
            if !env.e_obstaculo(x, y) {
                estados_para_exibir.push((x, y));
            }
        }
    }

    let selecionados: Vec<(i32, i32)> = if estados_para_exibir.len() > 18 {
        estados_para_exibir.into_iter().step_by(2).collect()
    } else {
        estados_para_exibir
    };

    for (x, y) in selecionados {
        let s = MundoGrelha2D::coords_para_estado((x, y));
        let q_cima = agente.memoria.q(&s, ACAO_CIMA);
        let q_baixo = agente.memoria.q(&s, ACAO_BAIXO);
        let q_esq = agente.memoria.q(&s, ACAO_ESQUERDA);
        let q_dir = agente.memoria.q(&s, ACAO_DIREITA);

        let v_s = q_cima.max(q_baixo).max(q_esq).max(q_dir);
        let a_opt = Greedy::accao_sofrega(&s, &agente.accoes, &agente.memoria);

        println!(
            "|    ({:>1}, {:<1})    | {:>9.2} | {:>9.2} | {:>9.2} | {:>9.2} | {:>9.2} | {:>14} |",
            x,
            y,
            q_cima,
            q_baixo,
            q_esq,
            q_dir,
            v_s,
            nome_acao(a_opt)
        );
    }
    println!(
        "+--------------+-----------+-----------+-----------+-----------+-----------+----------------+\n"
    );
}

// Simula greedy
fn obter_trajetoria_otima(
    agente: &Agente<Greedy, QLearning>,
    config: &Configuracao,
) -> Vec<PassoTrajetoria> {
    let mut env = MundoGrelha2D::novo(config);
    let mut s = env.iniciar();
    let mut passos = 0;
    let mut trajetoria = Vec::new();

    while passos < config.max_passos_episodio {
        let pos_antes = env.pos;
        let a = Greedy::accao_sofrega(&s, &agente.accoes, &agente.memoria);
        let q_val = agente.memoria.q(&s, a);

        let (s_prox, r, fim) = env.passo(a);
        passos += 1;

        trajetoria.push(PassoTrajetoria {
            passo: passos,
            pos_atual: pos_antes,
            acao: a,
            recompensa: r,
            pos_seguinte: env.pos,
            valor_q: q_val,
        });

        s = s_prox;
        if fim {
            break;
        }
    }

    trajetoria
}

fn imprimir_trajetoria_otima(agente: &Agente<Greedy, QLearning>, config: &Configuracao) {
    let trajetoria = obter_trajetoria_otima(agente, config);

    println!("========================================================================");
    println!("         TABELA 3: TRAJETÓRIA ÓTIMA FINAL CONVERGIDA PELO AGENTE        ");
    println!("========================================================================");
    println!(
        "+-------+-------------------+----------------+------------+--------------------+-----------+"
    );
    println!(
        "| Passo |    Estado Atual   | Ação Escolhida | Recompensa |     Novo Estado    |  Q(s, a)  |"
    );
    println!(
        "+-------+-------------------+----------------+------------+--------------------+-----------+"
    );
    for p in &trajetoria {
        println!(
            "| {:>5} |      ({:>1}, {:<1})       | {:>14} | {:>10.1} |       ({:>1}, {:<1})       | {:>9.2} |",
            p.passo,
            p.pos_atual.0,
            p.pos_atual.1,
            nome_acao(p.acao),
            p.recompensa,
            p.pos_seguinte.0,
            p.pos_seguinte.1,
            p.valor_q
        );
    }
    println!(
        "+-------+-------------------+----------------+------------+--------------------+-----------+"
    );

    let recompensa_total: f32 = trajetoria.iter().map(|p| p.recompensa).sum();
    let sucesso = trajetoria
        .last()
        .map_or(false, |p| p.pos_seguinte == config.meta);

    if sucesso {
        println!(
            "Trajetória final concluída com SUCESSO em {} passos (recompensa acumulada: {:.1}).\n",
            trajetoria.len(),
            recompensa_total
        );
    } else {
        println!(
            "Trajetória final interrompida após {} passos sem atingir a meta.\n",
            trajetoria.len()
        );
    }

    let mut visitadas = HashSet::new();
    for p in &trajetoria {
        visitadas.insert(p.pos_atual);
        visitadas.insert(p.pos_seguinte);
    }

    println!("Visualização da Trajetória Ótima (marcadores '*'):");
    imprimir_grelha(config, Some(&visitadas));
    println!("Legenda da Trajetória:");
    println!("  S   : Início em {:?}", config.inicio);
    println!("  G   : Meta em {:?}", config.meta);
    println!("  *   : Célula percorrida pela política ótima");
    println!(" ###  : Parede intransponível");
    println!("      : Célula livre não visitada\n");
}

// EXECUÇÃO

pub fn executar_experiencia(config: Configuracao) {
    imprimir_cabecalho(&config);
    imprimir_mapa_inicial(&config);

    // 1. Inicialização dos componentes de Aprendizagem por Reforço
    let accoes = ACTIONS.to_vec();
    let memoria = Memoria::iniciar_default();
    let politica = Greedy::iniciar(config.epsilon);
    let algoritmo = QLearning::iniciar(config.alfa, config.gama);

    let mut agente = Agente::iniciar(memoria, politica, algoritmo, accoes);
    let mut ambiente = MundoGrelha2D::novo(&config);

    // 2. Treino por blocos de episódios
    println!("A iniciar o treino do agente com Q-Learning...");
    let inicio_tempo_total = Instant::now();

    let n_blocos = config.n_episodios / config.tamanho_bloco;
    let mut historico_blocos = Vec::with_capacity(n_blocos);

    for bloco in 0..n_blocos {
        let ep_inicio = bloco * config.tamanho_bloco + 1;
        let ep_fim = (bloco + 1) * config.tamanho_bloco;
        let inicio_bloco = Instant::now();

        let mut soma_passos = 0;
        let mut soma_recompensa = 0.0;
        let mut sucessos = 0;

        for _ in ep_inicio..=ep_fim {
            agente.treinar_episodio(&mut ambiente);

            soma_passos += ambiente.ultimo_episodio_passos;
            soma_recompensa += ambiente.ultimo_episodio_recompensa;
            if ambiente.ultimo_episodio_meta_atingida {
                sucessos += 1;
            }
        }

        let duracao_bloco = inicio_bloco.elapsed().as_secs_f64();
        let tam_f32 = config.tamanho_bloco as f32;

        historico_blocos.push(MetricasBloco {
            bloco_inicio: ep_inicio,
            bloco_fim: ep_fim,
            passos_medios: soma_passos as f32 / tam_f32,
            recompensa_media: soma_recompensa / tam_f32,
            taxa_sucesso: (sucessos as f32 / tam_f32) * 100.0,
            tempo_segundos: duracao_bloco,
        });
    }

    let tempo_total = inicio_tempo_total.elapsed().as_secs_f64();
    println!("Treino concluído em {:.3}s.\n", tempo_total);

    // 3. Exibição das tabelas de resultados
    imprimir_tabela_evolucao(&historico_blocos);
    imprimir_tabela_valores_q(&agente, &config, &ambiente);
    imprimir_trajetoria_otima(&agente, &config);
}

fn main() {
    // let config = Configuracao {
    //     largura: 6,
    //     altura: 6,
    //     inicio: (0, 0),
    //     meta: (5, 5),
    //     obstaculos: vec![
    //         (1, 1), (1, 2), (1, 3), (1, 4),
    //         (3, 5), (3, 4), (3, 3), (3, 2), (3, 0),
    //         (4, 2),
    //         (5, 4),
    //     ],
    //     n_episodios: 600,
    //     max_passos_episodio: 200,
    //     alfa: 0.2,
    //     gama: 0.95,
    //     epsilon: 0.15,
    //     recompensa_meta: 100.0,
    //     penalizacao_passo: -1.0,
    //     tamanho_bloco: 60,
    // };

    let config = Configuracao::padrao();

    executar_experiencia(config);
}

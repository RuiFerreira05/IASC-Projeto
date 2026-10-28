use std::{collections::HashSet, time::Instant};

use iasc_p2::{
    agente::Agente,
    ambiente::{Acao, Ambiente, Estado},
    aprendizagem::{
        q_learning::{self, QLearning},
        sarsa::Sarsa,
    },
    memoria::Memoria,
    politica::greedy::Greedy,
};

// Definição das ações discretas no mundo de grelha 2D
pub const ACAO_CIMA: Acao = Acao(0);
pub const ACAO_BAIXO: Acao = Acao(1);
pub const ACAO_ESQUERDA: Acao = Acao(2);
pub const ACAO_DIREITA: Acao = Acao(3);

// Mapeamento dos identificadores de ação para nomes legíveis
pub fn nome_acao(a: Acao) -> &'static str {
    match a {
        ACAO_CIMA => "Cima",
        ACAO_BAIXO => "Baixo",
        ACAO_ESQUERDA => "Esquerda",
        ACAO_DIREITA => "Direita",
        _ => "Desconhecida",
    }
}

// Configuração dos parâmetros do ambiente e hiperparâmetros de treino
pub struct Configuracao {
    pub largura: i32,
    pub altura: i32,
    pub inicio: (i32, i32),
    pub meta: (i32, i32),
    pub obstaculos: Vec<(i32, i32)>,
    pub n_episodios: usize,
    pub max_passos_episodio: usize,
    pub alfa: f32,
    pub gama: f32,
    pub epsilon: f32,
    pub tamanho_bloco: usize,
}

impl Configuracao {
    // Configuração com grelha 6x6 e barreiras internas
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
            tamanho_bloco: 60,
        }
    }
}

// Ambiente que simula a grelha 2D com obstáculos e meta
pub struct MundoGrelha2D {
    largura: i32,
    altura: i32,
    inicio: (i32, i32),
    meta: (i32, i32),
    obstaculos: HashSet<(i32, i32)>,
    pos: (i32, i32),
    passos_atuais: usize,
    max_passos: usize,
    pub ultimo_episodio_passos: usize,
    pub ultimo_episodio_recompensa: f32,
    pub ultimo_episodio_meta_atingida: bool,
}

impl MundoGrelha2D {
    pub fn novo(config: &Configuracao) -> Self {
        let mut obstaculos = HashSet::new();
        for &obs in &config.obstaculos {
            obstaculos.insert(obs);
        }

        Self {
            largura: config.largura,
            altura: config.altura,
            inicio: config.inicio,
            meta: config.meta,
            obstaculos,
            pos: config.inicio,
            passos_atuais: 0,
            max_passos: config.max_passos_episodio,
            ultimo_episodio_passos: 0,
            ultimo_episodio_recompensa: 0.0,
            ultimo_episodio_meta_atingida: false,
        }
    }

    fn coords_para_estado(pos: (i32, i32)) -> Estado {
        Estado::novo(format!("{},{}", pos.0, pos.1))
    }

    pub fn eh_obstaculo(&self, x: i32, y: i32) -> bool {
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

        // Mantém a posição caso tente sair dos limites ou colidir com um obstáculo
        if novo_x >= 0
            && novo_x < self.largura
            && novo_y >= 0
            && novo_y < self.altura
            && !self.eh_obstaculo(novo_x, novo_y)
        {
            self.pos = (novo_x, novo_y);
        }

        let atingiu_meta = self.pos == self.meta;
        let esgotou_passos = self.passos_atuais >= self.max_passos;
        let fim = atingiu_meta || esgotou_passos;

        // Penalização por passo (-1.0) para incentivar o caminho mais curto; +100.0 ao alcançar a meta
        let recompensa = if atingiu_meta { 100.0 } else { -1.0 };

        self.ultimo_episodio_passos = self.passos_atuais;
        self.ultimo_episodio_recompensa += recompensa;
        self.ultimo_episodio_meta_atingida = atingiu_meta;

        (Self::coords_para_estado(self.pos), recompensa, fim)
    }
}

// Estrutura para acumular métricas por bloco de episódios de treino
struct MetricasBloco {
    bloco_inicio: usize,
    bloco_fim: usize,
    passos_medios: f32,
    recompensa_media: f32,
    taxa_sucesso: f32,
    tempo_segundos: f64,
}

// Registo de cada passo individual da trajetória ótima final
struct PassoTrajetoria {
    passo: usize,
    pos_atual: (i32, i32),
    acao: Acao,
    recompensa: f32,
    pos_seguinte: (i32, i32),
    valor_q: f32,
}

// Recolhe a sequência completa de passos da trajetória ótima
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

// Apresenta uma representação visual em grelha do ambiente inicial (labirinto)
fn imprimir_mapa(config: &Configuracao) {
    println!("Representação do Ambiente Inicial (Mundo de Grelha 2D):");
    println!();
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
            let celula = if (x, y) == config.inicio {
                "  A  "
            } else if (x, y) == config.meta {
                "  G  "
            } else if config.obstaculos.contains(&(x, y)) {
                " ### "
            } else {
                "  .  "
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
    println!("(Y)");
    println!();
    println!("Legenda:");
    println!("  A   : Agente (Posição Inicial em {:?})", config.inicio);
    println!(
        "  G   : Meta / Objetivo (Célula Terminal em {:?})",
        config.meta
    );
    println!(
        " ###  : Parede / Obstáculo intransponível ({} células)",
        config.obstaculos.len()
    );
    println!("  .   : Célula Livre transitável\n");
}

fn main() {
    let config = Configuracao::padrao();

    println!("========================================================================");
    println!("      APRENDIZAGEM POR REFORÇO (Q_LEARNING) EM MUNDO DE GRELHA 2D       ");
    println!("========================================================================");
    println!("Configuração da Experiência:");
    println!(
        "  - Dimensões da Grelha: {}x{} células",
        config.largura, config.altura
    );
    println!(
        "  - Posição Inicial: {:?} | Célula Meta: {:?}",
        config.inicio, config.meta
    );
    println!(
        "  - Número de Obstáculos: {} barreiras intransponíveis",
        config.obstaculos.len()
    );
    println!("  - Espaço de Ações: [Cima, Baixo, Esquerda, Direita]");
    println!(
        "  - Hiperparâmetros: alfa = {}, gama = {}, epsilon = {}",
        config.alfa, config.gama, config.epsilon
    );
    println!(
        "  - Total de Episódios: {} (limite de {} passos/episódio)",
        config.n_episodios, config.max_passos_episodio
    );
    println!("------------------------------------------------------------------------\n");

    imprimir_mapa(&config);

    let accoes = vec![ACAO_CIMA, ACAO_BAIXO, ACAO_ESQUERDA, ACAO_DIREITA];
    let memoria = Memoria::iniciar_default();
    let politica = Greedy::iniciar(config.epsilon);
    // let sarsa = Sarsa::iniciar(config.alfa, config.gama);
    let q_learning = QLearning::iniciar(config.alfa, config.gama);
    let mut agente = Agente::iniciar(memoria, politica, q_learning, accoes);

    let mut ambiente = MundoGrelha2D::novo(&config);

    let mut historico_blocos: Vec<MetricasBloco> = Vec::new();

    let inicio_tempo_total = Instant::now();

    println!("A iniciar o treino do agente...");

    let n_blocos = config.n_episodios / config.tamanho_bloco;

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
        let tam = config.tamanho_bloco as f32;

        historico_blocos.push(MetricasBloco {
            bloco_inicio: ep_inicio,
            bloco_fim: ep_fim,
            passos_medios: soma_passos as f32 / tam,
            recompensa_media: soma_recompensa / tam,
            taxa_sucesso: (sucessos as f32 / tam) * 100.0,
            tempo_segundos: duracao_bloco,
        });
    }

    let tempo_total = inicio_tempo_total.elapsed().as_secs_f64();
    println!("Treino concluído em {:.3}s.\n", tempo_total);

    // TABELA 1: Evolução estatística do treino por blocos
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
    for m in &historico_blocos {
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

    // TABELA 2: Tabela de valores Q(s, a) e ação ótima para estados-chave
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

    let estados_chave = vec![
        (0, 0),
        (0, 1),
        (0, 2),
        (0, 3),
        (0, 4),
        (0, 5),
        (2, 0),
        (2, 1),
        (2, 2),
        (2, 3),
        (2, 4),
        (2, 5),
        (4, 3),
        (4, 4),
        (4, 5),
        (5, 4),
        (5, 5),
    ];

    for (x, y) in estados_chave {
        if ambiente.eh_obstaculo(x, y) {
            continue;
        }
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

    // TABELA 3: Trajetória ótima final
    let trajetoria = obter_trajetoria_otima(&agente, &config);
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
    println!(
        "Trajetória final concluída com sucesso em {} passos (recompensa total: {:.1}).",
        trajetoria.len(),
        trajetoria.iter().map(|p| p.recompensa).sum::<f32>()
    );
    println!("========================================================================\n");
}

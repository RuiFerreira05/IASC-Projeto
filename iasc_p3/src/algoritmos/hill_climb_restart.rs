use crate::{
    algoritmos::hill_climb::hill_climb,
    ambiente::{Estado, Problema},
};

pub fn hill_climb_restart<E: Estado>(problema: &impl Problema<E>, max_reinicios: u64) -> E {
    // Rust precisa do estado inicializado dado que a descrição do algoritmo nos slides contem um path onde melhor_estado pode nunca ser inicializado
    let mut melhor_estado = hill_climb(problema);
    let mut valor_melhor_estado = f64::NEG_INFINITY;

    for _ in 0..max_reinicios {
        let estado = hill_climb(problema);
        if problema.solucao(estado) {
            melhor_estado = estado;
            break;
        } else {
            let valor_estado = problema.avaliar(estado);
            if valor_estado > valor_melhor_estado {
                melhor_estado = estado;
                valor_melhor_estado = valor_estado;
            }
        }
    }

    return melhor_estado;
}

#[cfg(test)]
mod hill_climb_restart_tests {
    use super::*;

    // Este teste não descreve bem a utilidade do hill climb com restart aleatório, apenas certifica que funciona.
    #[test]
    fn grelha_2d() {
        #[derive(Debug, PartialEq, Clone, Copy)]
        struct Grelha2dEstado {
            pub x: u64,
            pub y: u64,
        }
        impl Estado for Grelha2dEstado {}

        struct Grelha2d {
            pub max_x: u64,
            pub max_y: u64,
        }

        struct Grelha2dProb {
            ambiente: Grelha2d,
            meta: Grelha2dEstado,
            meta_valor: f64,
        }

        impl Problema<Grelha2dEstado> for Grelha2dProb {
            fn estado_inicial(&self) -> Grelha2dEstado {
                Grelha2dEstado {
                    x: rand::random_range(1..=self.ambiente.max_x),
                    y: rand::random_range(1..=self.ambiente.max_y),
                }
            }

            fn sucessores(&self, estado: Grelha2dEstado) -> Vec<Grelha2dEstado> {
                let mut sucessores: Vec<Grelha2dEstado> = Vec::new();

                for x in (estado.x as i64 - 1)..=(estado.x as i64 + 1) {
                    if !(1..self.ambiente.max_x as i64).contains(&x) {
                        continue; // x out of bouds
                    }

                    for y in (estado.y as i64 - 1)..=(estado.y as i64 + 1) {
                        if !(1..self.ambiente.max_y as i64).contains(&x) {
                            continue; // y out of bounds
                        }

                        if x == estado.x as i64 && y == estado.y as i64 {
                            continue; // own state
                        }

                        sucessores.push(Grelha2dEstado {
                            x: x as u64,
                            y: y as u64,
                        });
                    }
                }

                sucessores
            }

            fn avaliar(&self, estado: Grelha2dEstado) -> f64 {
                let dist_x = self.meta.x.abs_diff(estado.x) as f64;
                let dist_y = self.meta.y.abs_diff(estado.y) as f64;
                let dist = (dist_x.powi(2) + dist_y.powi(2)).sqrt();

                self.meta_valor - dist
            }

            fn solucao(&self, estado: Grelha2dEstado) -> bool {
                self.meta == estado
            }
        }

        let grelha = Grelha2d {
            max_x: 10,
            max_y: 10,
        };

        let prob = Grelha2dProb {
            ambiente: grelha,
            meta: Grelha2dEstado { x: 5, y: 5 },
            meta_valor: 10.0,
        };

        let estado_final = hill_climb_restart(&prob, 5000);
        assert!(prob.solucao(estado_final))
    }
}

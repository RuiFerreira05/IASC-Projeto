use rand::seq::{IndexedRandom, SliceRandom};

use crate::problema::{Estado, Problema};

pub fn hce_unico<E: Estado>(problema: &impl Problema<E>) -> E {
    let mut estado = problema.estado_inicial();
    let mut valor_estado = f64::NEG_INFINITY;
    println!("estado inicial: {estado:?}");

    loop {
        let mut sucessores = problema.sucessores(estado);

        // shuffle para garantir escolha aleatoria
        sucessores.shuffle(&mut rand::rng());

        let novo_estado = sucessores
            .into_iter()
            .find(|e| problema.avaliar(*e) > valor_estado);

        match novo_estado {
            Some(e) => {
                estado = e;
                valor_estado = problema.avaliar(e);
                println!("novo estado: {estado:?} - valor: {valor_estado:?}");
            }
            _ => break,
        };
    }

    return estado;
}

#[cfg(test)]
mod hce_unico_tests {
    use super::*;

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

        let estado_final = hce_unico(&prob);
        assert_eq!(estado_final, prob.meta)
    }
}

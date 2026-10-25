#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub struct Acao(pub usize);

impl From<usize> for Acao {
    fn from(id: usize) -> Self {
        Acao(id)
    }
}

#[derive(Clone, PartialEq, Eq, Hash, Debug)]
pub struct Estado(pub String);

impl Estado {
    pub fn novo(id: impl Into<String>) -> Self {
        Self(id.into())
    }
}

impl From<&str> for Estado {
    fn from(s: &str) -> Self {
        Estado(s.to_string())
    }
}

impl From<String> for Estado {
    fn from(s: String) -> Self {
        Estado(s)
    }
}

pub trait Ambiente {
    fn iniciar(&mut self) -> Estado;
    fn passo(&mut self, a: Acao) -> (Estado, f32, bool);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_acao_criacao_e_from() {
        let a1 = Acao(1);
        let a2 = Acao::from(1);
        assert_eq!(a1, a2);
        assert_eq!(a1.0, 1);
    }

    #[test]
    fn test_estado_criacao_e_from() {
        let e1 = Estado::novo("s0");
        let e2 = Estado::from("s0");
        let e3 = Estado::from(String::from("s0"));
        assert_eq!(e1, e2);
        assert_eq!(e2, e3);
        assert_eq!(e1.0, "s0");
    }

    struct MockAmbiente {
        passos: usize,
    }

    impl Ambiente for MockAmbiente {
        fn iniciar(&mut self) -> Estado {
            self.passos = 0;
            Estado::from("inicio")
        }

        fn passo(&mut self, _a: Acao) -> (Estado, f32, bool) {
            self.passos += 1;
            let terminal = self.passos >= 2;
            (
                Estado::from(format!("passo_{}", self.passos)),
                1.0,
                terminal,
            )
        }
    }

    #[test]
    fn test_ambiente_trait() {
        let mut mock = MockAmbiente { passos: 0 };
        let s0 = mock.iniciar();
        assert_eq!(s0, Estado::from("inicio"));

        let (s1, r1, term1) = mock.passo(Acao(0));
        assert_eq!(s1, Estado::from("passo_1"));
        assert_eq!(r1, 1.0);
        assert!(!term1);

        let (s2, r2, term2) = mock.passo(Acao(0));
        assert_eq!(s2, Estado::from("passo_2"));
        assert_eq!(r2, 1.0);
        assert!(term2);
    }
}

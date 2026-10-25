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

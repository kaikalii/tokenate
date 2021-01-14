use crate::*;

pub trait Pattern<R>
where
    R: Read,
{
    type Token;
    fn matches(&self, chars: &mut Chars<R>) -> TokenResult<Self::Token>;
}

impl<R, F, T> Pattern<R> for F
where
    R: Read,
    F: Fn(&mut Chars<R>) -> TokenResult<T>,
{
    type Token = T;
    fn matches(&self, chars: &mut Chars<R>) -> TokenResult<Self::Token> {
        self(chars)
    }
}

pub fn if_char<R, F>(f: F) -> impl Fn(&mut Chars<R>) -> TokenResult<String>
where
    R: Read,
    F: Fn(char) -> bool,
{
    move |chars: &mut Chars<R>| {
        let mut s = String::new();
        while let Some(c) = chars.take_if(&f)? {
            s.push(c)
        }
        if s.is_empty() {
            Err(TokenError::Unmatched)
        } else {
            Ok(s)
        }
    }
}

pub fn charset<R>(set: &str) -> impl Fn(&mut Chars<R>) -> TokenResult<String> + '_
where
    R: Read,
{
    move |chars| if_char(move |c| set.contains(c))(chars)
}

pub fn whitespace<R>(chars: &mut Chars<R>) -> TokenResult<String>
where
    R: Read,
{
    if_char(char::is_whitespace)(chars)
}

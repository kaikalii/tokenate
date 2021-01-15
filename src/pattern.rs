use std::{marker::PhantomData, str::FromStr};

use crate::*;

pub fn not_whitespace(c: char) -> bool {
    !c.is_whitespace()
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Default)]
pub struct Patterns<A, B> {
    pub first: A,
    pub second: B,
}

pub trait Pattern<R>
where
    R: Read,
{
    type Token;
    fn matching(&self, chars: &mut Chars<R>) -> TokenResult<Sp<Self::Token>>;
    fn or<B>(self, other: B) -> Patterns<Self, B>
    where
        Self: Sized,
        B: Pattern<R>,
    {
        Patterns {
            first: self,
            second: other,
        }
    }
    fn or_chars<B>(self, other: B) -> Patterns<Self, CharPatternWrapper<B>>
    where
        Self: Sized,
        B: CharPattern,
    {
        Patterns {
            first: self,
            second: other.pattern(),
        }
    }
    fn map<F>(self, f: F) -> MappedPattern<Self, F>
    where
        Self: Sized,
    {
        MappedPattern { pattern: self, f }
    }
    fn parse<T>(self) -> Parse<R, Self, T>
    where
        Self: Sized,
        Self::Token: AsRef<str>,
    {
        Parse {
            pattern: self,
            pd: PhantomData,
        }
    }
    fn is<T>(self, val: T) -> Is<R, Self, T>
    where
        Self: Sized,
        T: Clone,
    {
        Is {
            pattern: self,
            val,
            pd: PhantomData,
        }
    }
}

impl<R, F, T> Pattern<R> for F
where
    R: Read,
    F: Fn(&mut Chars<R>) -> TokenResult<T>,
{
    type Token = T;
    fn matching(&self, chars: &mut Chars<R>) -> TokenResult<Sp<Self::Token>> {
        let start_size = chars.history.len();
        let start_loc = chars.loc;
        match self(chars) {
            Ok(Some(token)) => Ok(Some(start_loc.to(chars.loc).sp(token))),
            Ok(None) => {
                chars.revert(start_size, start_loc);
                Ok(None)
            }
            Err(e) => Err(e),
        }
    }
}

impl<R, A, B> Pattern<R> for Patterns<A, B>
where
    R: Read,
    A: Pattern<R>,
    B: Pattern<R, Token = A::Token>,
{
    type Token = A::Token;
    fn matching(&self, chars: &mut Chars<R>) -> TokenResult<Sp<Self::Token>> {
        match self.first.matching(chars) {
            Ok(None) => self.second.matching(chars),
            res => res,
        }
    }
}

pub trait CharPattern {
    fn matches(&self, c: char) -> bool;
    fn pattern(self) -> CharPatternWrapper<Self>
    where
        Self: Sized,
    {
        CharPatternWrapper(self)
    }
}

impl CharPattern for char {
    fn matches(&self, c: char) -> bool {
        self == &c
    }
}

impl<'a> CharPattern for &'a str {
    fn matches(&self, c: char) -> bool {
        self.contains(c)
    }
}

impl<F> CharPattern for F
where
    F: Fn(char) -> bool,
{
    fn matches(&self, c: char) -> bool {
        self(c)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Default)]
pub struct CharPatternWrapper<P>(pub P);

impl<R, P> Pattern<R> for CharPatternWrapper<P>
where
    R: Read,
    P: CharPattern,
{
    type Token = String;
    fn matching(&self, chars: &mut Chars<R>) -> TokenResult<Sp<Self::Token>> {
        let mut token = String::new();
        let start_loc = chars.loc;
        while let Some(c) = chars.take_if(|c| self.0.matches(c))? {
            token.push(c);
        }
        Ok(if token.is_empty() {
            None
        } else {
            Some(start_loc.to(chars.loc).sp(token))
        })
    }
}

pub struct MappedPattern<P, F> {
    pattern: P,
    f: F,
}

impl<R, P, F, U> Pattern<R> for MappedPattern<P, F>
where
    R: Read,
    P: Pattern<R>,
    F: Fn(P::Token) -> U,
{
    type Token = U;
    fn matching(&self, chars: &mut Chars<R>) -> TokenResult<Sp<Self::Token>> {
        Ok(self
            .pattern
            .matching(chars)?
            .map(|token| token.map(&self.f)))
    }
}

pub struct Parse<R, P, T>
where
    R: Read,
    P: Pattern<R>,
{
    pattern: P,
    pd: PhantomData<(T, R)>,
}

impl<R, P, T> Pattern<R> for Parse<R, P, T>
where
    R: Read,
    P: Pattern<R>,
    P::Token: AsRef<str>,
    T: FromStr,
{
    type Token = T;
    fn matching(&self, chars: &mut Chars<R>) -> TokenResult<Sp<Self::Token>> {
        Ok(self.pattern.matching(chars)?.and_then(|token| {
            token
                .data
                .as_ref()
                .parse::<T>()
                .ok()
                .map(|parsed| token.span.sp(parsed))
        }))
    }
}

pub struct Is<R, P, T>
where
    R: Read,
    P: Pattern<R>,
{
    pattern: P,
    val: T,
    pd: PhantomData<R>,
}

impl<R, P, T> Pattern<R> for Is<R, P, T>
where
    R: Read,
    P: Pattern<R>,
    T: Clone,
{
    type Token = T;
    fn matching(&self, chars: &mut Chars<R>) -> TokenResult<Sp<Self::Token>> {
        Ok(self
            .pattern
            .matching(chars)?
            .map(|token| token.span.sp(self.val.clone())))
    }
}

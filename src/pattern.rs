//! The [`Pattern`] trait, its combinators, and some helper functions for patterns

use std::{marker::PhantomData, str::FromStr};

use crate::*;

/// Check if a `char` is not whitespace
pub fn not_whitespace(c: char) -> bool {
    !c.is_whitespace()
}

/// Create a [`Pattern`] from a [`CharPattern`]
pub fn chars<P>(pattern: P) -> CharPatternWrapper<P>
where
    P: CharPattern,
{
    pattern.pattern()
}

/// Two [`Pattern`]s that will attempt to be matched in order
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Default)]
pub struct Patterns<A, B> {
    /// The first pattern
    pub first: A,
    /// The second pattern
    pub second: B,
}

/**
Defines a token pattern
*/
pub trait Pattern<R>
where
    R: Read,
{
    /// The type of the token that is produced if the pattern matches
    type Token;
    /// Try to match the pattern and consume a token from [`Chars`]
    fn matching(&self, chars: &mut Chars<R>) -> TokenResult<Sp<Self::Token>>;
    /// Combine this pattern with another. If matching this pattern fails,
    /// the other pattern will be tried
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
    /// Combine this pattern with a [`CharPattern`]
    ///
    /// Equivalent to `pattern.or(char_pattern.pattern())`
    fn or_chars<B>(self, char_pattern: B) -> Patterns<Self, CharPatternWrapper<B>>
    where
        Self: Sized,
        B: CharPattern,
    {
        Patterns {
            first: self,
            second: char_pattern.pattern(),
        }
    }
    /// Create a pattern that first tries to match this one.
    /// Upon success, the resulting token is transformed using the given function.
    fn map<F>(self, f: F) -> Map<Self, F>
    where
        Self: Sized,
    {
        Map { pattern: self, f }
    }
    /// Create a pattern that attempt to parse to a token of a type that implements [`FromStr`]
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
    /// Create a pattern that first tries to match this one.
    /// Upon success, a the given value is returned instead.
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
    /// Change the token type to `()`
    fn skip(self) -> Skip<R, Self>
    where
        Self: Sized,
    {
        self.is(())
    }
    /// Combine this pattern with another and change their token types to `()`
    fn or_skip<B>(self, other: B) -> Patterns<Skip<R, Self>, Skip<R, B>>
    where
        Self: Sized,
        B: Pattern<R>,
    {
        Patterns {
            first: self.skip(),
            second: other.skip(),
        }
    }
}

impl<R> Pattern<R> for ()
where
    R: Read,
{
    type Token = ();
    fn matching(&self, _: &mut Chars<R>) -> TokenResult<Sp<Self::Token>> {
        Ok(None)
    }
}

impl<R, F, T> Pattern<R> for F
where
    R: Read,
    F: Fn(&mut Chars<R>) -> TokenResult<T>,
{
    type Token = T;
    fn matching(&self, chars: &mut Chars<R>) -> TokenResult<Sp<Self::Token>> {
        let tracker = chars.track();
        match self(chars) {
            Ok(Some(token)) => Ok(Some(tracker.loc.to(chars.loc).sp(token))),
            Ok(None) => {
                chars.revert(tracker);
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

/// A pattern than either accepts or rejects individual characters
pub trait CharPattern {
    /// Check if the pattern matches a character
    fn matches(&self, c: char) -> bool;
    /// Promote this to wrapper than implements [`Pattern`]
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

/// A wrapper for implementors of [`CharPattern`] that implements [`Pattern`]
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

/// The pattern produced by [`Pattern::map`]
pub struct Map<P, F> {
    pattern: P,
    f: F,
}

impl<R, P, F, U> Pattern<R> for Map<P, F>
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

/// The pattern produced by [`Pattern::parse`]
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

/// The pattern produced by [`Pattern::is`]
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

/// The pattern produced by [`Pattern::skip`]
pub type Skip<R, P> = Is<R, P, ()>;

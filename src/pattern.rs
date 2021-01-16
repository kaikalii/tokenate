//! The [`Pattern`] trait, its combinators, and some helper functions for patterns

use std::{
    marker::PhantomData,
    ops::{Bound, RangeBounds, RangeFrom, RangeInclusive},
    str::FromStr,
};

use crate::*;

/// Check if a `char` is not whitespace
pub fn not_whitespace(c: char) -> bool {
    !c.is_whitespace()
}

/// Create a [`Pattern`] from a [`CharPattern`]
pub fn chars<P>(pattern: P) -> TakeAtLeast<P>
where
    P: CharPattern,
{
    pattern.any()
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
pub trait Pattern {
    /// The type of the token that is produced if the pattern matches
    type Token;
    /// Try to match the pattern and consume a token from [`Chars`]
    fn matching(&self, chars: &mut Chars) -> TokenResult<Sp<Self::Token>>;
    /// Combine this pattern with another. If matching this pattern fails,
    /// the other pattern will be tried
    fn or<B>(self, other: B) -> Patterns<Self, B>
    where
        Self: Sized,
        B: Pattern<Token = Self::Token>,
    {
        Patterns {
            first: self,
            second: other,
        }
    }
    /// Combine this pattern with a [`CharPattern`]
    ///
    /// Equivalent to `pattern.or(char_pattern.any())`
    fn or_chars<B>(self, char_pattern: B) -> Patterns<Self, TakeAtLeast<B>>
    where
        Self: Sized,
        B: CharPattern,
    {
        Patterns {
            first: self,
            second: char_pattern.any(),
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
    fn parse<T>(self) -> Parse<Self, T>
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
    fn is<T>(self, val: T) -> Is<Self, T>
    where
        Self: Sized,
        T: Clone,
    {
        Is { pattern: self, val }
    }
    /// Create a pattern that first tries to match this one.
    /// Upon success, attempts to match the given second pattern and combine the
    /// resulting tokens with the given function
    fn join<B, F, T>(self, second: B, f: F) -> Join<Self, B, F>
    where
        Self: Sized,
        B: Pattern,
        F: Fn(Self::Token, B::Token) -> T,
    {
        Join {
            a: self,
            b: second,
            f,
        }
    }
    /// Change the token type to `()`
    fn skip(self) -> Skip<Self>
    where
        Self: Sized,
    {
        self.is(())
    }
    /// Combine this pattern with another and change their token types to `()`
    fn or_skip<B>(self, other: B) -> Patterns<Skip<Self>, Skip<B>>
    where
        Self: Sized,
        B: Pattern,
    {
        Patterns {
            first: self.skip(),
            second: other.skip(),
        }
    }
}

impl Pattern for () {
    type Token = ();
    fn matching(&self, _: &mut Chars) -> TokenResult<Sp<Self::Token>> {
        Ok(None)
    }
}

impl<'a> Pattern for &'a str {
    type Token = String;
    fn matching(&self, chars: &mut Chars) -> TokenResult<Sp<Self::Token>> {
        let tracker = chars.track();
        for c in self.chars() {
            if chars.take_if(c)?.is_none() {
                chars.revert(tracker);
                return Ok(None);
            }
        }
        Ok(Some(tracker.loc.to(chars.loc).sp((*self).into())))
    }
}

impl<F, T> Pattern for F
where
    F: Fn(&mut Chars) -> TokenResult<T>,
{
    type Token = T;
    fn matching(&self, chars: &mut Chars) -> TokenResult<Sp<Self::Token>> {
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

impl<A, B> Pattern for Patterns<A, B>
where
    A: Pattern,
    B: Pattern<Token = A::Token>,
{
    type Token = A::Token;
    fn matching(&self, chars: &mut Chars) -> TokenResult<Sp<Self::Token>> {
        match self.first.matching(chars) {
            Ok(None) => self.second.matching(chars),
            res => res,
        }
    }
}

impl<T> Pattern for Box<dyn Pattern<Token = T>> {
    type Token = T;
    fn matching(&self, chars: &mut Chars) -> TokenResult<Sp<Self::Token>> {
        Box::as_ref(&self).matching(chars)
    }
}

/// A pattern than either accepts or rejects individual characters
pub trait CharPattern {
    /// Check if the pattern matches a character
    fn matches(&self, c: char) -> bool;
    /// Promote this to wrapper than implements [`Pattern`] with [`Pattern::Token`] = [`String`]
    fn any(self) -> TakeAtLeast<Self>
    where
        Self: Sized,
    {
        self.take(1..)
    }
    /**
    Promote this to wrapper than implements [`Pattern`] with [`Pattern::Token`] = [`String`]
    and matches if the length of matched strings lies within the given range

    Unlike [`CharPattern::any`] or [`pattern::chars`] (which are equivalent), passing
    this function can match empty strings by passing a range that contains `0`. For a use
    case of this, see the example in the [`crate`] root.
    */
    fn take<N>(self, range: N) -> Take<Self, N>
    where
        Self: Sized,
        N: RangeBounds<usize>,
    {
        Take {
            pattern: self,
            range,
        }
    }
    /// Promote this to wrapper than implements [`Pattern`] with [`Pattern::Token`] = [`String`]
    /// and matches if the length of matched strings lies within the given range
    fn take_exact(self, n: usize) -> TakeExact<Self>
    where
        Self: Sized,
    {
        self.take(n..=n)
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

/// The pattern produced by [`CharPattern::any`] and [`chars`]
pub type TakeAtLeast<P> = Take<P, RangeFrom<usize>>;
/// The pattern produced by [`CharPattern::take_exact`]
pub type TakeExact<P> = Take<P, RangeInclusive<usize>>;

/// The pattern produced by [`Pattern::map`]
pub struct Map<P, F> {
    pattern: P,
    f: F,
}

impl<P, F, U> Pattern for Map<P, F>
where
    P: Pattern,
    F: Fn(P::Token) -> U,
{
    type Token = U;
    fn matching(&self, chars: &mut Chars) -> TokenResult<Sp<Self::Token>> {
        Ok(self
            .pattern
            .matching(chars)?
            .map(|token| token.map(&self.f)))
    }
}

/// The pattern produced by [`Pattern::parse`]
pub struct Parse<P, T>
where
    P: Pattern,
{
    pattern: P,
    pd: PhantomData<T>,
}

impl<P, T> Pattern for Parse<P, T>
where
    P: Pattern,
    P::Token: AsRef<str>,
    T: FromStr,
{
    type Token = T;
    fn matching(&self, chars: &mut Chars) -> TokenResult<Sp<Self::Token>> {
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
pub struct Is<P, T> {
    pattern: P,
    val: T,
}

impl<P, T> Pattern for Is<P, T>
where
    P: Pattern,
    T: Clone,
{
    type Token = T;
    fn matching(&self, chars: &mut Chars) -> TokenResult<Sp<Self::Token>> {
        Ok(self
            .pattern
            .matching(chars)?
            .map(|token| token.span.sp(self.val.clone())))
    }
}

/// The pattern produced by [`Pattern::skip`]
pub type Skip<P> = Is<P, ()>;

/// The pattern produced by [`Pattern::join`]
pub struct Join<A, B, F>
where
    A: Pattern,
    B: Pattern,
{
    a: A,
    b: B,
    f: F,
}

impl<A, B, F, T> Pattern for Join<A, B, F>
where
    A: Pattern,
    B: Pattern,
    F: Fn(A::Token, B::Token) -> T,
{
    type Token = T;
    fn matching(&self, chars: &mut Chars) -> TokenResult<Sp<Self::Token>> {
        Ok(if let Some(first) = self.a.matching(chars)? {
            self.b.matching(chars)?.map(|second| {
                first
                    .span
                    .start
                    .to(second.span.end)
                    .sp((self.f)(first.data, second.data))
            })
        } else {
            None
        })
    }
}

/// The pattern produced by [`CharPattern::take`]
pub struct Take<P, N> {
    pattern: P,
    range: N,
}

impl<P, N> Pattern for Take<P, N>
where
    P: CharPattern,
    N: RangeBounds<usize>,
{
    type Token = String;
    fn matching(&self, chars: &mut Chars) -> TokenResult<Sp<Self::Token>> {
        let mut token = String::new();
        let tracker = chars.track();
        while let Some(c) = chars.take_if(|c| self.pattern.matches(c))? {
            token.push(c);
            match self.range.end_bound() {
                Bound::Excluded(n) if token.len() + 1 == *n => break,
                Bound::Included(n) if token.len() == *n => break,
                _ => {}
            }
        }
        let long_enough = match self.range.start_bound() {
            Bound::Excluded(n) if token.len() > *n => true,
            Bound::Included(n) if token.len() >= *n => true,
            Bound::Unbounded => true,
            _ => false,
        };
        Ok(if long_enough {
            Some(tracker.loc.to(chars.loc).sp(token))
        } else {
            None
        })
    }
}

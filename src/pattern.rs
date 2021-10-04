//! The [`Pattern`] trait, its combinators, and some helper functions for patterns

#[cfg(feature = "debug")]
use std::{cell::Cell, iter::repeat};
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

/// Create a pattern that matches an identifier
///
/// The `first_char` pattern matches the first character in the identifier.
/// The `other_chars` pattern matches the rest.
pub fn ident<A, B>(first_char: A, other_chars: B) -> impl Pattern<Token = String>
where
    A: CharPattern,
    B: CharPattern,
{
    first_char
        .take_exact(1)
        .join(other_chars.take(..), |first, others| first + &others)
}

fn ptr_name<T>(r: &T) -> String
where
    T: ?Sized,
{
    format!(
        "<{}>",
        format!("{:x}", r as *const T as *const u8 as usize)
            .chars()
            .rev()
            .collect::<String>()
    )
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

thread_local! {
    #[cfg(feature = "debug")]
    static DEBUG_INDENT: Cell<usize> = Cell::new(0);
}

/**
Defines a token pattern
*/
pub trait Pattern {
    /// The type of the token that is produced if the pattern matches
    type Token;
    /// Try to match the pattern and consume a token from [`Chars`]
    ///
    /// This method should be implemented but not directly called
    fn try_match(&self, chars: &mut Chars) -> TokenResult<Sp<Self::Token>>;
    /// Try to match the pattern and consume a token from [`Chars`]
    ///
    /// This method should be called but not implemented
    #[allow(clippy::let_and_return)]
    fn matching(&self, chars: &mut Chars) -> TokenResult<Sp<Self::Token>> {
        #[cfg(feature = "debug")]
        {
            DEBUG_INDENT.with(|indent| {
                println!(
                    "{}trying {}",
                    repeat("    ").take(indent.get()).collect::<String>(),
                    self.name()
                );
                indent.set(indent.get() + 1);
            });
        }
        let tracker = chars.track();
        let res = match self.try_match(chars) {
            Ok(Some(token)) => {
                #[cfg(feature = "debug")]
                {
                    DEBUG_INDENT.with(|indent| {
                        indent.set(indent.get() - 1);
                        println!(
                            "{}{} succeeded",
                            repeat("    ").take(indent.get()).collect::<String>(),
                            self.name()
                        );
                    });
                }
                Ok(Some(token))
            }
            Ok(None) => {
                #[cfg(feature = "debug")]
                {
                    DEBUG_INDENT.with(|indent| {
                        indent.set(indent.get() - 1);
                        println!(
                            "{}{} failed",
                            repeat("    ").take(indent.get()).collect::<String>(),
                            self.name()
                        );
                    });
                }
                chars.revert(tracker);
                Ok(None)
            }
            Err(e) => Err(e),
        };
        res
    }
    /// Get a user-friendly name for the pattern
    fn name(&self) -> String {
        ptr_name(self)
    }
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
    fn try_match(&self, _: &mut Chars) -> TokenResult<Sp<Self::Token>> {
        Ok(None)
    }
    fn name(&self) -> String {
        "unit".into()
    }
}

impl<'a> Pattern for char {
    type Token = String;
    fn try_match(&self, chars: &mut Chars) -> TokenResult<Sp<Self::Token>> {
        self.take_exact(1).matching(chars)
    }
    fn name(&self) -> String {
        format!("{:?}", self)
    }
}

impl<'a> Pattern for &'a str {
    type Token = String;
    fn try_match(&self, chars: &mut Chars) -> TokenResult<Sp<Self::Token>> {
        let start_loc = chars.loc;
        for c in self.chars() {
            if chars.take_if(c)?.is_none() {
                return Ok(None);
            }
        }
        Ok(Some(start_loc.to(chars.loc).sp((*self).into())))
    }
    fn name(&self) -> String {
        format!("{:?}", self)
    }
}

impl<F, T> Pattern for F
where
    F: Fn(&mut Chars) -> TokenResult<T>,
{
    type Token = T;
    fn try_match(&self, chars: &mut Chars) -> TokenResult<Sp<Self::Token>> {
        let start_loc = chars.loc;
        match self(chars) {
            Ok(Some(token)) => Ok(Some(start_loc.to(chars.loc).sp(token))),
            Ok(None) => Ok(None),
            Err(e) => Err(e),
        }
    }
    fn name(&self) -> String {
        format!("fn({})", ptr_name(self))
    }
}

impl<A, B> Pattern for Patterns<A, B>
where
    A: Pattern,
    B: Pattern<Token = A::Token>,
{
    type Token = A::Token;
    fn try_match(&self, chars: &mut Chars) -> TokenResult<Sp<Self::Token>> {
        match self.first.matching(chars) {
            Ok(None) => self.second.matching(chars),
            res => res,
        }
    }
    fn name(&self) -> String {
        format!("({} or {})", self.first.name(), self.second.name())
    }
}

impl<T> Pattern for Box<dyn Pattern<Token = T>> {
    type Token = T;
    fn try_match(&self, chars: &mut Chars) -> TokenResult<Sp<Self::Token>> {
        Box::as_ref(self).matching(chars)
    }
    fn name(&self) -> String {
        format!("box({})", Box::as_ref(self).name())
    }
}

/// A pattern than either accepts or rejects individual characters
pub trait CharPattern {
    /// Check if the pattern matches a character
    fn matches(&self, c: char) -> bool;
    /// Get a user friendly name for the pattern
    fn name(&self) -> String {
        ptr_name(self)
    }
    /// Promote this to a wrapper than implements [`Pattern`] with [`Pattern::Token`] = [`String`]
    fn any(self) -> TakeAtLeast<Self>
    where
        Self: Sized,
    {
        self.take(1..)
    }
    /**
    Promote this to a wrapper than implements [`Pattern`] with [`Pattern::Token`] = [`String`]
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
    /// Promote this to a wrapper than implements [`Pattern`] with [`Pattern::Token`] = [`String`]
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
    fn name(&self) -> String {
        format!("{:?}", self)
    }
}

impl<'a> CharPattern for &'a str {
    fn matches(&self, c: char) -> bool {
        self.contains(c)
    }
    fn name(&self) -> String {
        format!("{:?}", self)
    }
}

impl<F> CharPattern for F
where
    F: Fn(char) -> bool,
{
    fn matches(&self, c: char) -> bool {
        self(c)
    }
    fn name(&self) -> String {
        format!("fn({})", ptr_name(self))
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
    fn try_match(&self, chars: &mut Chars) -> TokenResult<Sp<Self::Token>> {
        Ok(self
            .pattern
            .matching(chars)?
            .map(|token| token.map(&self.f)))
    }
    fn name(&self) -> String {
        format!("map({})", self.pattern.name())
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
    fn try_match(&self, chars: &mut Chars) -> TokenResult<Sp<Self::Token>> {
        Ok(self.pattern.matching(chars)?.and_then(|token| {
            token
                .data
                .as_ref()
                .parse::<T>()
                .ok()
                .map(|parsed| token.span.sp(parsed))
        }))
    }
    fn name(&self) -> String {
        format!("parse({})", self.pattern.name())
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
    fn try_match(&self, chars: &mut Chars) -> TokenResult<Sp<Self::Token>> {
        Ok(self
            .pattern
            .matching(chars)?
            .map(|token| token.span.sp(self.val.clone())))
    }
    fn name(&self) -> String {
        format!("is({})", self.pattern.name())
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
    fn try_match(&self, chars: &mut Chars) -> TokenResult<Sp<Self::Token>> {
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
    fn name(&self) -> String {
        format!("{} + {}", self.a.name(), self.b.name())
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
    fn try_match(&self, chars: &mut Chars) -> TokenResult<Sp<Self::Token>> {
        let mut token = String::new();
        let start_loc = chars.loc;
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
            Some(start_loc.to(chars.loc).sp(token))
        } else {
            None
        })
    }
    fn name(&self) -> String {
        format!("take({})", self.pattern.name())
    }
}

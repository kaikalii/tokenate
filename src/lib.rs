#![warn(missing_docs)]

/*!
This crate implements a simple interface for building lexical analyzers.

# Usage

## The [`Chars`] struct

[`Chars`] wraps a [`Read`]er, parses characters from it, and allows reversion to previous positions.
It is the main struct used for lexing.

## The [`Pattern`] trait

Lexers are defined by a series of patterns that attempt to match and consume input from the input.
Patterns can be transformed and composed to create the lexer rules.

[`Pattern`] is implemented for
- `&str`, which attempts to match itself exactly
- Anything that implements `Fn(&mut Chars) -> TokenResult<T>` where `T` is the token type
- The [`pattern::Patterns`] struct, which tries two patterns in order
- Any of the combinators from the [`pattern`] module, which are created via methods of [`Pattern`] and [`CharPattern`]

## The [`CharPattern`] trait

Rather than matching strings of text, the [`CharPattern`] trait accepts or rejects individual characters.
A [`CharPattern`] can be promoted to a [`Pattern`] using either [`pattern::chars`] or [`CharPattern::any`].

# Custom [`Pattern`] functions

Custom patterns can be created easily by simply defining functions with the signature
`fn(&mut Chars) -> TokenResult<T>` where `T` is the token type.

Here is a simple example example of a pattern than matches a string literal:

```
use tokenate::*;

fn string_literal(chars: &mut Chars) -> TokenResult<String> {
    Ok(if chars.take_if(|c| c == '"')?.is_some() {
        let mut arg = String::new();
        let mut escaped = false;
        while let Some(c) = chars.take()? {
            match c {
                '\\' if escaped.take() => arg.push('\\'),
                '\\' => escaped = true,
                '"' if escaped.take() => arg.push('"'),
                '"' => break,
                'n' if escaped.take() => arg.push('\n'),
                'r' if escaped.take() => arg.push('\r'),
                't' if escaped.take() => arg.push('\t'),
                c if escaped => return Err(
                    chars.error(format!("Invalid escape char: {:?}", c))
                ),
                c => arg.push(c),
            }
        }
        Some(arg)
    } else {
        None
    })
}

assert_eq!(
    "Hi \nthere!",
    Chars::new(r#""Hi \nthere!""#.as_bytes())
        .tokens(&string_literal, &char::is_whitespace.any())
        .next()
        .unwrap()
        .unwrap()
        .data
)
```

# Simple Lexer Example

This simple examble attempts to tokenize a simple grammar that consists of boolean literals, floating-point number literals, and C-like identifiers.

```
use tokenate::*;

/// Our token type
#[derive(Debug, PartialEq)]
enum Token {
    Number(f32),
    Bool(bool),
    Ident(String),
}

// The bool pattern
let bools = "true".is(true).or("false".is(false)).map(Token::Bool);

// The number pattern
let numbers = "0123456789-+.e".any().parse::<f32>().map(Token::Number);

// Helper functions for the ident pattern
let ident_start = |c: char| c.is_alphabetic() && (c as u32) < 127 || c == '_';
let ident_body = |c: char| ident_start(c) || c.is_digit(10);
// The ident pattern
let idents = pattern::ident(ident_start, ident_body).map(Token::Ident);

// The full pattern
// It is important that idents come after bools here, or else "true" and "false" would
// get tokenized as idents
let pattern = bools.or(numbers).or(idents);

// The pattern used to skip whitespace
// Without this, the tokenization would fail upon encountering whitespace
let skip = char::is_whitespace.any();

// Our test input
let input = "true foo -3.2 fals _2";

// Lex
let tokens = Chars::new(input.as_bytes())
    .tokenize(&pattern, &skip)
    .unwrap();

let expected = vec![
    Token::Bool(true),
    Token::Ident("foo".into()),
    Token::Number(-3.2),
    Token::Ident("fals".into()),
    Token::Ident("_2".into()),
];

assert_eq!(tokens, expected);
```

# More Examples

More examples can be found [on GitHub](https://github.com/kaikalii/tokenate/tree/main/examples).
*/

pub mod pattern;

use std::{
    error::Error,
    fmt::{self, Debug, Display, Formatter},
    io::{self, Bytes, Read},
    ops::{BitOr, BitOrAssign},
};

use smallvec::SmallVec;
use unicode_reader::CodePoints;

pub use pattern::{CharPattern, Pattern};

const INVALID_INPUT_MAX_LEN: usize = 30;

/// An error type for [`LexError`]
#[derive(Debug)]
pub enum LexErrorKind {
    /// An IO error
    Io(io::Error),
    /// No patterns matched the remaining input
    InvalidInput(String),
    /// A custom message
    Custom(String),
}

impl Display for LexErrorKind {
    fn fmt(&self, f: &mut Formatter) -> fmt::Result {
        match self {
            LexErrorKind::Io(e) => Display::fmt(e, f),
            LexErrorKind::InvalidInput(s) => write!(
                f,
                "Unable to tokenize {:?}{}",
                s,
                if s.len() >= INVALID_INPUT_MAX_LEN {
                    "..."
                } else {
                    ""
                }
            ),
            LexErrorKind::Custom(message) => write!(f, "Unable to parse: `{}`", message),
        }
    }
}

impl Error for LexErrorKind {}

impl From<io::Error> for LexErrorKind {
    fn from(e: io::Error) -> Self {
        LexErrorKind::Io(e)
    }
}

impl From<String> for LexErrorKind {
    fn from(e: String) -> Self {
        LexErrorKind::Custom(e)
    }
}

impl<'a> From<&String> for LexErrorKind {
    fn from(e: &String) -> Self {
        LexErrorKind::Custom(e.clone())
    }
}

impl<'a> From<&str> for LexErrorKind {
    fn from(e: &str) -> Self {
        LexErrorKind::Custom(e.into())
    }
}

/// An error that occurs when lexing
#[derive(Debug)]
pub struct LexError {
    /// The error kind
    pub kind: LexErrorKind,
    /// The location in the input where the error occured
    pub loc: Loc,
}

impl Display for LexError {
    fn fmt(&self, f: &mut Formatter) -> fmt::Result {
        write!(f, "Error at {}\n{}", self.loc, self.kind)
    }
}

impl Error for LexError {}

/// A result type for a complete lexical analysis
pub type LexResult<T> = Result<T, LexError>;
/// A result type for individual token matches
pub type TokenResult<T> = LexResult<Option<T>>;

/// A location in a lexer input
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct Loc {
    /// The line
    pub line: usize,
    /// The column
    pub column: usize,
}

impl Loc {
    /// Create a new location
    pub fn new(line: usize, column: usize) -> Self {
        Loc { line, column }
    }
    /// Create a [`Span`] with this location as the start
    pub fn to(self, end: Self) -> Span {
        Span::new(self, end)
    }
}

impl Debug for Loc {
    fn fmt(&self, f: &mut Formatter) -> fmt::Result {
        Display::fmt(self, f)
    }
}

impl Display for Loc {
    fn fmt(&self, f: &mut Formatter) -> fmt::Result {
        write!(f, "{}:{}", self.line, self.column)
    }
}

/// A span from one [`Loc`] in the lexer input to another
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct Span {
    /// The starting location
    pub start: Loc,
    /// The ending location
    pub end: Loc,
}

impl Span {
    /// Create a new span
    pub fn new(start: Loc, end: Loc) -> Self {
        Span { start, end }
    }
    /// Create a [`Sp`] using this span
    pub fn sp<T>(&self, data: T) -> Sp<T> {
        Sp::new(data, self.span())
    }
}

impl Display for Span {
    fn fmt(&self, f: &mut Formatter) -> fmt::Result {
        write!(f, "{} - {}", self.start, self.end)
    }
}

impl BitOr for Span {
    type Output = Self;
    /// Get the smallest span that contains 2 spans
    fn bitor(mut self, other: Self) -> Self::Output {
        self |= other;
        self
    }
}

impl BitOrAssign for Span {
    fn bitor_assign(&mut self, other: Self) {
        self.start = self.start.min(other.start);
        self.end = self.end.max(other.end);
    }
}

/// A piece of data with an attached [`Span`]
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct Sp<T> {
    /// The spanned data
    pub data: T,
    /// The span
    pub span: Span,
}

impl<T> Sp<T> {
    /// Create a new spanned data
    pub fn new(data: T, span: Span) -> Self {
        Sp { data, span }
    }
    /// Map the data to a new value or type while keeping the same span
    pub fn map<F, U>(self, f: F) -> Sp<U>
    where
        F: FnOnce(T) -> U,
    {
        Sp {
            data: f(self.data),
            span: self.span,
        }
    }
    /// Combine two pieces of spanned data with the given function and span
    /// the result with a span than contains both originals spans
    pub fn join<U, F, J>(self, other: Sp<U>, f: F) -> Sp<J>
    where
        F: FnOnce(T, U) -> J,
    {
        Sp {
            data: f(self.data, other.data),
            span: self.span | other.span,
        }
    }
    /// Get a spanned reference to the data
    pub fn as_ref(&self) -> Sp<&T> {
        self.span.sp(&self.data)
    }
}

impl<T> PartialEq<T> for Sp<T>
where
    T: PartialEq,
{
    fn eq(&self, other: &T) -> bool {
        &self.data == other
    }
}

impl<T> Debug for Sp<T>
where
    T: Debug,
{
    fn fmt(&self, f: &mut Formatter) -> fmt::Result {
        Debug::fmt(&self.data, f)?;
        write!(f, " [{}]", self.span)
    }
}

impl<T> Display for Sp<T>
where
    T: Display,
{
    fn fmt(&self, f: &mut Formatter) -> fmt::Result {
        write!(f, "{} [{}]", self.data, self.span)
    }
}

/// Trait for getting [`Span`]s from things
pub trait Spanned {
    /// Get the span
    fn span(&self) -> Span;
}

impl<'a, T> Spanned for &'a T
where
    T: Spanned,
{
    fn span(&self) -> Span {
        T::span(self)
    }
}

impl<'a, T> Spanned for &'a mut T
where
    T: Spanned,
{
    fn span(&self) -> Span {
        T::span(self)
    }
}

impl Spanned for Span {
    fn span(&self) -> Span {
        *self
    }
}

impl<T> Spanned for Sp<T> {
    fn span(&self) -> Span {
        self.span
    }
}

impl<A, B> Spanned for (A, B)
where
    A: Spanned,
    B: Spanned,
{
    fn span(&self) -> Span {
        self.0.span() | self.1.span()
    }
}

impl<A, B, C> Spanned for (A, B, C)
where
    A: Spanned,
    B: Spanned,
    C: Spanned,
{
    fn span(&self) -> Span {
        self.0.span() | self.1.span() | self.2.span()
    }
}

impl<A, B, C, D> Spanned for (A, B, C, D)
where
    A: Spanned,
    B: Spanned,
    C: Spanned,
    D: Spanned,
{
    fn span(&self) -> Span {
        self.0.span() | self.1.span() | self.2.span() | self.3.span()
    }
}

impl<T, E> Spanned for Result<T, E>
where
    T: Spanned,
    E: Spanned,
{
    fn span(&self) -> Span {
        match self {
            Ok(s) => s.span(),
            Err(e) => e.span(),
        }
    }
}

/// Trait for getting optional [`Span`]s from things
pub trait MaybeSpanned {
    /// Try to get the span
    fn maybe_span(&self) -> Option<Span>;
    /// Get the span if it exists, otherwise use the default
    fn span_or<U>(&self, default: &U) -> Span
    where
        U: Spanned,
    {
        self.maybe_span().unwrap_or_else(|| default.span())
    }
}

impl MaybeSpanned for () {
    fn maybe_span(&self) -> Option<Span> {
        None
    }
}

impl<T> MaybeSpanned for T
where
    T: Spanned,
{
    fn maybe_span(&self) -> Option<Span> {
        Some(self.span())
    }
}

impl<T> MaybeSpanned for Option<T>
where
    T: MaybeSpanned,
{
    fn maybe_span(&self) -> Option<Span> {
        self.as_ref().and_then(MaybeSpanned::maybe_span)
    }
}

impl<T> MaybeSpanned for [T]
where
    T: MaybeSpanned,
{
    fn maybe_span(&self) -> Option<Span> {
        self.first()
            .and_then(MaybeSpanned::maybe_span)
            .map(|first| {
                self.iter()
                    .skip(1)
                    .filter_map(MaybeSpanned::maybe_span)
                    .fold(first, |acc, span| acc | span)
            })
    }
}

/// Parses characters from a reader
pub struct Chars<'a> {
    chars: CodePoints<Bytes<Box<dyn Read + 'a>>>,
    history: SmallVec<[(char, Loc); 32]>,
    cursor: usize,
    revert_trackers: usize,
    loc: Loc,
}

impl<'a, R> From<R> for Chars<'a>
where
    R: Read + 'a,
{
    fn from(reader: R) -> Self {
        Chars::new(reader)
    }
}

impl<'a> Chars<'a> {
    /// Create a new `Chars` from a reader
    pub fn new<R>(reader: R) -> Self
    where
        R: Read + 'a,
    {
        let reader: Box<dyn Read> = Box::new(reader);
        Chars {
            chars: reader.bytes().into(),
            history: SmallVec::new(),
            cursor: 0,
            revert_trackers: 0,
            loc: Loc::new(1, 1),
        }
    }
    /// Get the current input location
    pub fn loc(&self) -> Loc {
        self.loc
    }
    fn put_back(&mut self) {
        self.cursor -= 1;
    }
    /// Get a reversion handle to the current input position
    ///
    /// [`Chars::tokenize`], [`Chars::tokens`], and [`Chars::into_tokens] handle
    /// this automatically, so you shouldn't normally need to call this function.
    pub fn track(&mut self) -> RevertHandle {
        self.revert_trackers += 1;
        RevertHandle {
            loc: self.loc,
            cursor: self.cursor,
        }
    }
    /// Revert to the position defined by a reversion handle
    ///
    /// This is used to revert to a previous input position when a pattern fails to match.
    /// [`Chars::tokenize`], [`Chars::tokens`], and [`Chars::into_tokens] handle
    /// this automatically, so you shouldn't normally need to call this function.
    ///
    /// It is genreally a logic error to pass this function a handle which was created
    /// from a different [`Chars`].
    pub fn revert(&mut self, handle: RevertHandle) {
        self.loc = handle.loc;
        self.cursor = handle.cursor;
        self.revert_trackers -= 1;
        if self.revert_trackers == 0 {
            self.cursor = 0;
            self.history.clear();
        }
    }
    /// Peek at the next character without consuming it
    pub fn peek(&mut self) -> LexResult<Option<char>> {
        Ok(if let Some((c, _)) = self.history.get(self.cursor) {
            Some(*c)
        } else {
            let loc = self.loc;
            self.take()?.map(|c| {
                self.put_back();
                self.loc = loc;
                c
            })
        })
    }
    /// Take a character
    pub fn take(&mut self) -> LexResult<Option<char>> {
        let c = if self.cursor < self.history.len() {
            Some(self.history[self.cursor].0)
        } else if let Some(c) = self.chars.next().transpose().map_err(|io_error| LexError {
            kind: io_error.into(),
            loc: self.loc,
        })? {
            self.history.push((c, self.loc));
            Some(c)
        } else {
            None
        };
        Ok(if let Some(c) = c {
            self.cursor += 1;
            match c {
                '\n' => {
                    self.loc.line += 1;
                    self.loc.column = 1;
                }
                _ => self.loc.column += 1,
            }
            Some(c)
        } else {
            None
        })
    }
    /// Get a consuming iterator over the characters
    pub fn take_iter<'b>(&'b mut self) -> TakeIter<'a, 'b> {
        TakeIter { chars: self }
    }
    /// Take a character if it satisfies the [`CharPattern`]
    pub fn take_if<P>(&mut self, pattern: P) -> LexResult<Option<char>>
    where
        P: CharPattern,
    {
        Ok(self.peek()?.and_then(|c| {
            if pattern.matches(c) {
                self.take().unwrap();
                Some(c)
            } else {
                None
            }
        }))
    }
    /// Turn a [`LexErrorKind`] into a [`LexError`] with the current location
    pub fn error<E>(&self, error: E) -> LexError
    where
        E: Into<LexErrorKind>,
    {
        LexError {
            kind: error.into(),
            loc: self.loc(),
        }
    }
    /// Create a [`LexError`] with an error representing a failure to match any patterns
    pub fn invalid_input(&mut self) -> LexError {
        let loc = self.loc();
        LexError {
            kind: LexErrorKind::Custom(
                self.take_iter()
                    .filter_map(Result::ok)
                    .take(INVALID_INPUT_MAX_LEN)
                    .collect(),
            ),
            loc,
        }
    }
    /// Get an iterator over tokens matching `matching` and skipping `skip`
    pub fn tokens<'b, M, S>(&'b mut self, matching: &'b M, skip: &'b S) -> Tokens<'a, 'b, M, S>
    where
        M: Pattern,
        S: Pattern,
    {
        Tokens {
            chars: self,
            matching,
            skip,
        }
    }
    /// Consume self into an iterator over tokens matching `matching` and skipping `skip`
    pub fn into_tokens<M, S>(self, matching: M, skip: S) -> IntoTokens<'a, M, S>
    where
        M: Pattern,
        S: Pattern,
    {
        IntoTokens {
            chars: self,
            matching,
            skip,
        }
    }
    /**
    Attempt to fully tokenize the remaining input

    Tokens generated by the `matching` pattern will be returned in order.

    Tokens matching the `skip` pattern will consume input but not generate tokens.
    This is useful for things like whitespace.
    */
    pub fn tokenize<M, S>(&mut self, matching: &M, skip: &S) -> LexResult<Vec<Sp<M::Token>>>
    where
        M: Pattern,
        S: Pattern,
    {
        self.tokens(matching, skip).collect()
    }
}

/// The iterator returned from [`Chars::tokens`]
pub struct Tokens<'a, 'b, M, S> {
    chars: &'b mut Chars<'a>,
    matching: &'b M,
    skip: &'b S,
}

impl<M, S> Iterator for Tokens<'_, '_, M, S>
where
    M: Pattern,
    S: Pattern,
{
    type Item = LexResult<Sp<M::Token>>;
    fn next(&mut self) -> Option<Self::Item> {
        loop {
            match self.chars.peek() {
                Ok(Some(_)) => {}
                Ok(None) => return None,
                Err(e) => return Some(Err(e)),
            }
            let tracker = self.chars.track();
            match self.matching.matching(self.chars) {
                Ok(Some(token)) => return Some(Ok(token)),
                Ok(None) => match self.skip.matching(self.chars) {
                    Ok(Some(_)) => {}
                    Ok(None) => {
                        self.chars.revert(tracker);
                        return Some(Err(self.chars.invalid_input()));
                    }
                    Err(e) => return Some(Err(e)),
                },
                Err(e) => return Some(Err(e)),
            }
        }
    }
}

/// The iterator returned from [`Chars::into_tokens`]
pub struct IntoTokens<'a, M, S> {
    chars: Chars<'a>,
    matching: M,
    skip: S,
}

impl<M, S> Iterator for IntoTokens<'_, M, S>
where
    M: Pattern,
    S: Pattern,
{
    type Item = LexResult<Sp<M::Token>>;
    fn next(&mut self) -> Option<Self::Item> {
        self.chars.tokens(&self.matching, &self.skip).next()
    }
}

/// The iterator returned by [`Chars::take_iter`]
pub struct TakeIter<'a, 'b> {
    chars: &'b mut Chars<'a>,
}

impl Iterator for TakeIter<'_, '_> {
    type Item = LexResult<char>;
    fn next(&mut self) -> Option<Self::Item> {
        self.chars.take().transpose()
    }
}

/// A handle to a position in the lexer input which can be used to revert to that position
#[derive(Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct RevertHandle {
    loc: Loc,
    cursor: usize,
}

impl RevertHandle {
    /// Get the location of the reversion handle
    pub fn loc(&self) -> Loc {
        self.loc
    }
}

/// Get the state of a [`bool`] and set it to `false` in one line
pub trait BoolTake {
    /// Get the state and set to `false`
    fn take(&mut self) -> bool;
}

impl BoolTake for bool {
    fn take(&mut self) -> bool {
        let res = *self;
        *self = false;
        res
    }
}

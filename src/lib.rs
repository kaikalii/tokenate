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
                c if escaped =>
                    return Err(LexError::Custom(format!("Invalid escape char: {:?}", c))),
                c => arg.push(c),
            }
        }
        Some(arg)
    } else {
        None
    })
}

Chars::new("\" Hi there!\"".as_bytes()).tokenize(&string_literal, &()).unwrap();
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
let numbers = "0123456789-.e".any().parse::<f32>().map(Token::Number);

// Helper functions for the ident pattern
let ident_start = |c: char| c.is_alphabetic() && (c as u32) < 127 || c == '_';
let ident_body = |c: char| ident_start(c) || c.is_digit(10);
// The ident pattern
let idents = ident_start
    .take_exact(1)
    .join(ident_body.take(..), |start, body| start + &body)
    .map(Token::Ident);

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
*/

pub mod pattern;

use std::{
    error::Error,
    fmt::{self, Display, Formatter},
    io::{self, Bytes, Read},
};

use smallvec::SmallVec;
use unicode_reader::CodePoints;

pub use pattern::{CharPattern, Pattern};

const INVALID_INPUT_MAX_LEN: usize = 30;

/// An error that occurs when lexing
#[derive(Debug)]
pub enum LexError {
    /// An IO error
    IO(io::Error),
    /// No patterns matched the remaining input
    InvalidInput(String),
    /// A custom message
    Custom(String),
}

impl Display for LexError {
    fn fmt(&self, f: &mut Formatter) -> fmt::Result {
        match self {
            LexError::IO(e) => e.fmt(f),
            LexError::InvalidInput(s) => write!(
                f,
                "Unable to tokenize {:?}{}",
                s,
                if s.len() >= INVALID_INPUT_MAX_LEN {
                    "..."
                } else {
                    ""
                }
            ),
            LexError::Custom(message) => message.fmt(f),
        }
    }
}

impl Error for LexError {}

impl From<io::Error> for LexError {
    fn from(e: io::Error) -> Self {
        LexError::IO(e)
    }
}

/// A result type for a complete lexical analysis
pub type LexResult<T> = Result<T, LexError>;
/// A result type for individual token matches
pub type TokenResult<T> = LexResult<Option<T>>;

/// A location in a lexer input
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
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
    pub fn sp<T>(self, data: T) -> Sp<T> {
        Sp::new(data, self)
    }
}

impl Display for Span {
    fn fmt(&self, f: &mut Formatter) -> fmt::Result {
        write!(f, "{} - {}", self.start, self.end)
    }
}

/// A piece of data with an attached [`Span`]
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
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
}

impl<T> PartialEq<T> for Sp<T>
where
    T: PartialEq,
{
    fn eq(&self, other: &T) -> bool {
        &self.data == other
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

/// Parses characters from a reader
pub struct Chars<'a> {
    chars: CodePoints<Bytes<Box<dyn Read + 'a>>>,
    history: SmallVec<[char; 32]>,
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
    /// [`Chars::tokenize`] handles this automatically, so you shouldn't normally need to call this function
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
    /// [`Chars::tokenize`] handles this automatically, so you shouldn't normally need to call this function
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
    pub fn peek(&mut self) -> io::Result<Option<char>> {
        Ok(if let Some(c) = self.history.get(self.cursor) {
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
    pub fn take(&mut self) -> io::Result<Option<char>> {
        let c = if self.cursor < self.history.len() {
            Some(self.history[self.cursor])
        } else if let Some(c) = self.chars.next().transpose()? {
            self.history.push(c);
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
    /// Get an iterator over the characters
    pub fn take_iter<'b>(&'b mut self) -> TakeIter<'a, 'b> {
        TakeIter { chars: self }
    }
    /// Take a character if it satisfies the [`CharPattern`]
    pub fn take_if<P>(&mut self, pattern: P) -> io::Result<Option<char>>
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
    /// Create a [`Result::Err`] with an error representing a failure to match any patterns
    pub fn invalid_input<T>(&mut self) -> LexResult<T> {
        Err(LexError::InvalidInput(
            self.take_iter()
                .filter_map(Result::ok)
                .take(INVALID_INPUT_MAX_LEN)
                .collect(),
        ))
    }
    /// Attempt to match a pattern and consume a token
    pub fn matching<P>(&mut self, pattern: &P) -> TokenResult<Sp<P::Token>>
    where
        P: Pattern,
    {
        pattern.matching(self)
    }
    /**
    Attempt to fully tokenize the remaining input

    Tokens generated by the `matching` pattern will be returned in order.

    Tokens matching the `skip` pattern will consume input but not generate tokens.
    This is useful for things like whitespace.
    */
    pub fn tokenize<M, S, T>(&mut self, matching: &M, skip: &S) -> LexResult<Vec<Sp<T>>>
    where
        M: Pattern<Token = T>,
        S: Pattern,
    {
        let mut tokens = Vec::new();
        while self.peek()?.is_some() {
            let tracker = self.track();
            if let Some(token) = self.matching(matching)? {
                tokens.push(token);
            } else if self.matching(skip)?.is_none() {
                self.revert(tracker);
                return self.invalid_input();
            }
        }
        Ok(tokens)
    }
}

/// The iterator returned by [`Chars::take_iter`]
pub struct TakeIter<'a, 'b> {
    chars: &'b mut Chars<'a>,
}

impl Iterator for TakeIter<'_, '_> {
    type Item = io::Result<char>;
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

/// Get the state of a [`bool`] and setting it to `false` in one line
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

#![warn(missing_docs)]

/*!
This crate implements a simple interface for building lexical analyzers.
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

/// A piece of data with an attached span
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

impl<T> Display for Sp<T>
where
    T: Display,
{
    fn fmt(&self, f: &mut Formatter) -> fmt::Result {
        write!(f, "{} [{}]", self.data, self.span)
    }
}

/// Parses characters from a reader
pub struct Chars<R>
where
    R: Read,
{
    chars: CodePoints<Bytes<R>>,
    history: SmallVec<[char; 32]>,
    cursor: usize,
    revert_trackers: usize,
    loc: Loc,
}

impl<R> From<R> for Chars<R>
where
    R: Read,
{
    fn from(reader: R) -> Self {
        Chars::new(reader)
    }
}

impl<R> Chars<R>
where
    R: Read,
{
    /// Create a new `Chars` from a reader
    pub fn new(reader: R) -> Self {
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
    fn track(&mut self) -> ReverHandle {
        self.revert_trackers += 1;
        ReverHandle {
            loc: self.loc,
            cursor: self.cursor,
        }
    }
    fn revert(&mut self, handle: ReverHandle) {
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
    pub fn take_iter(&mut self) -> impl Iterator<Item = io::Result<char>> + '_ {
        std::iter::from_fn(move || self.take().transpose())
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
        P: Pattern<R>,
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
        M: Pattern<R, Token = T>,
        S: Pattern<R>,
        T: fmt::Debug,
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

struct ReverHandle {
    loc: Loc,
    cursor: usize,
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

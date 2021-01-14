pub mod pattern;

use std::{
    fmt::{self, Display, Formatter},
    io::{self, Bytes, Read},
};

use unicode_reader::CodePoints;

pub use pattern::Pattern;

const NO_MATCHING_PATTERN_MESSAGE_LEN: usize = 30;

#[derive(Debug)]
pub enum LexError {
    IO(io::Error),
    NoMatchingPattern(String),
}

impl Display for LexError {
    fn fmt(&self, f: &mut Formatter) -> fmt::Result {
        match self {
            LexError::IO(e) => e.fmt(f),
            LexError::NoMatchingPattern(text) => {
                if text.len() > NO_MATCHING_PATTERN_MESSAGE_LEN {
                    write!(f, "No pattern matching {:?}...", text)
                } else {
                    write!(f, "No pattern matching {:?}", text)
                }
            }
        }
    }
}

pub type LexResult<T> = Result<T, LexError>;

#[derive(Debug)]
pub enum TokenError {
    Error(LexError),
    Unmatched,
}

pub type TokenResult<T> = Result<T, TokenError>;

impl From<io::Error> for LexError {
    fn from(error: io::Error) -> Self {
        LexError::IO(error)
    }
}

impl From<LexError> for TokenError {
    fn from(error: LexError) -> Self {
        TokenError::Error(error)
    }
}

impl From<io::Error> for TokenError {
    fn from(error: io::Error) -> Self {
        TokenError::Error(error.into())
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct Loc {
    pub line: usize,
    pub column: usize,
}

impl Loc {
    pub fn new(line: usize, column: usize) -> Self {
        Loc { line, column }
    }
    pub fn to(self, end: Self) -> Span {
        Span::new(self, end)
    }
}

impl Display for Loc {
    fn fmt(&self, f: &mut Formatter) -> fmt::Result {
        write!(f, "{}:{}", self.line, self.column)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct Span {
    pub start: Loc,
    pub end: Loc,
}

impl Span {
    pub fn new(start: Loc, end: Loc) -> Self {
        Span { start, end }
    }
    pub fn sp<T>(self, data: T) -> Sp<T> {
        Sp::new(data, self)
    }
}

impl Display for Span {
    fn fmt(&self, f: &mut Formatter) -> fmt::Result {
        write!(f, "{} - {}", self.start, self.end)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct Sp<T> {
    pub data: T,
    pub span: Span,
}

impl<T> Sp<T> {
    pub fn new(data: T, span: Span) -> Self {
        Sp { data, span }
    }
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

pub struct Chars<R = Box<dyn Read>>
where
    R: Read,
{
    chars: CodePoints<Bytes<R>>,
    put_back: Vec<char>,
    history: Vec<char>,
    loc: Loc,
}

impl<R> Chars<R>
where
    R: Read,
{
    pub fn new(reader: R) -> Self {
        Chars {
            chars: reader.bytes().into(),
            put_back: Vec::new(),
            history: Vec::new(),
            loc: Loc::new(1, 1),
        }
    }
    pub fn loc(&self) -> Loc {
        self.loc
    }
    pub fn peek(&mut self) -> io::Result<Option<char>> {
        Ok(if let Some(c) = self.put_back.last().copied() {
            Some(c)
        } else {
            let loc = self.loc;
            self.take()?.map(|c| {
                self.put_back.push(c);
                self.loc = loc;
                c
            })
        })
    }
    pub fn take(&mut self) -> io::Result<Option<char>> {
        let c = if let Some(c) = self.put_back.pop() {
            Some(c)
        } else if let Some(c) = self.chars.next().transpose()? {
            Some(c)
        } else {
            None
        };
        Ok(if let Some(c) = c {
            self.history.push(c);
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
    pub fn take_if<F>(&mut self, f: F) -> io::Result<Option<char>>
    where
        F: Fn(char) -> bool,
    {
        Ok(self.peek()?.and_then(|c| {
            if f(c) {
                self.take().unwrap();
                Some(c)
            } else {
                None
            }
        }))
    }
    fn revert(&mut self, n: usize, loc: Loc) {
        self.loc = loc;
        for _ in 0..(self.history.len() - n) {
            self.put_back.extend(self.history.pop());
        }
    }
}

struct PatternConfig<R, T>
where
    R: Read,
{
    skip: bool,
    pattern: Box<dyn Pattern<R, Token = T>>,
}

pub struct TokenPatterns<R, T>
where
    R: Read,
{
    patterns: Vec<PatternConfig<R, T>>,
}

impl<R, T> Default for TokenPatterns<R, T>
where
    R: Read,
{
    fn default() -> Self {
        TokenPatterns {
            patterns: Vec::new(),
        }
    }
}

impl<R, T> TokenPatterns<R, T>
where
    R: Read,
{
    pub fn new() -> Self {
        Self::default()
    }
    pub fn with<P>(mut self, pattern: P) -> Self
    where
        P: Fn(&mut Chars<R>) -> TokenResult<T> + 'static,
    {
        self.patterns.push(PatternConfig {
            skip: false,
            pattern: Box::new(pattern),
        });
        self
    }
    pub fn skip<Pattern>(mut self, pattern: Pattern) -> Self
    where
        Pattern: Fn(&mut Chars<R>) -> TokenResult<T> + 'static,
    {
        self.patterns.push(PatternConfig {
            skip: true,
            pattern: Box::new(pattern),
        });
        self
    }
    pub fn tokenize(&self, reader: R) -> LexResult<Vec<Sp<T>>> {
        let mut args = Vec::new();
        let mut chars = Chars::new(reader);
        while chars.peek()?.is_some() {
            let mut matched = false;
            let start_len = chars.history.len();
            let start_loc = chars.loc;
            for cfg in &self.patterns {
                match cfg.pattern.matches(&mut chars) {
                    Ok(token) => {
                        if !cfg.skip {
                            args.push(start_loc.to(chars.loc).sp(token));
                        }
                        matched = true;
                        break;
                    }
                    Err(TokenError::Unmatched) => chars.revert(start_len, start_loc),
                    Err(TokenError::Error(e)) => return Err(e),
                }
            }
            if !matched {
                return Err(LexError::NoMatchingPattern(
                    std::iter::from_fn(|| chars.take().transpose())
                        .filter_map(Result::ok)
                        .take(NO_MATCHING_PATTERN_MESSAGE_LEN)
                        .collect(),
                ));
            }
        }
        Ok(args)
    }
}

pub trait BoolTake {
    fn take(&mut self) -> bool;
}

impl BoolTake for bool {
    fn take(&mut self) -> bool {
        let res = *self;
        *self = false;
        res
    }
}

pub trait OrUnmatched<T> {
    fn or_unmatched(self) -> TokenResult<T>;
}

impl<T> OrUnmatched<T> for Option<T> {
    fn or_unmatched(self) -> TokenResult<T> {
        self.ok_or(TokenError::Unmatched)
    }
}

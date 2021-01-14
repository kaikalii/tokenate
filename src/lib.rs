use std::{
    fmt::{self, Display, Formatter},
    io::{self, Bytes, Read},
};

use unicode_reader::CodePoints;

#[derive(Debug)]
pub enum LexError {
    IO(io::Error),
    NoMatchingPattern,
}

impl Display for LexError {
    fn fmt(&self, f: &mut Formatter) -> fmt::Result {
        match self {
            LexError::IO(e) => e.fmt(f),
            LexError::NoMatchingPattern => write!(f, "No matching pattern"),
        }
    }
}

pub type LexResult<T> = Result<T, LexError>;

#[derive(Debug)]
pub enum LexFailure {
    Error(LexError),
    Unmatched,
    Skip,
}

pub type LexControl<T> = Result<T, LexFailure>;

impl From<io::Error> for LexError {
    fn from(error: io::Error) -> Self {
        LexError::IO(error)
    }
}

impl From<LexError> for LexFailure {
    fn from(error: LexError) -> Self {
        LexFailure::Error(error)
    }
}

impl From<io::Error> for LexFailure {
    fn from(error: io::Error) -> Self {
        LexFailure::Error(error.into())
    }
}

pub const fn unmatched<T>() -> LexControl<T> {
    Err(LexFailure::Unmatched)
}

pub const fn skip<T>() -> LexControl<T> {
    Err(LexFailure::Skip)
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
    pub fn peek(&mut self) -> LexControl<char> {
        Ok(if let Some(c) = self.put_back.last().copied() {
            c
        } else {
            let c = self.take()?;
            self.put_back.push(c);
            c
        })
    }
    pub fn take(&mut self) -> LexControl<char> {
        if let Some(c) = self.put_back.pop() {
            self.history.push(c);
            Ok(c)
        } else if let Some(c) = self.chars.next().transpose()? {
            self.history.push(c);
            Ok(c)
        } else {
            unmatched()
        }
    }
    pub fn take_if<F>(&mut self, f: F) -> LexControl<char>
    where
        F: Fn(char) -> bool,
    {
        let c = self.peek()?;
        if f(c) {
            self.take().unwrap();
            Ok(c)
        } else {
            unmatched()
        }
    }
    pub fn lex<F, T>(&mut self, f: F) -> LexResult<Option<Sp<T>>>
    where
        F: Fn(&mut Self) -> LexControl<T>,
    {
        let start_len = self.history.len();
        let start_loc = self.loc;
        match f(self) {
            Ok(token) => Ok(Some(start_loc.to(self.loc).sp(token))),
            Err(e) => {
                self.loc = start_loc;
                for _ in 0..(self.history.len() - start_len) {
                    self.put_back.extend(self.history.pop());
                }
                match e {
                    LexFailure::Error(e) => Err(e),
                    _ => Ok(None),
                }
            }
        }
    }
    pub fn if_char<F>(&mut self, f: F) -> LexControl<String>
    where
        F: Fn(char) -> bool,
    {
        let mut s = String::new();
        while let Ok(c) = self.take_if(&f) {
            s.push(c)
        }
        if s.is_empty() {
            Err(LexFailure::Unmatched)
        } else {
            Ok(s)
        }
    }
    pub fn charset(&mut self, set: &[char]) -> LexControl<String> {
        self.if_char(|c| set.contains(&c))
    }
    pub fn whitespace(&mut self) -> LexControl<String> {
        self.if_char(char::is_whitespace)
    }
}

type DynTokenPattern<R, T> = dyn Fn(&mut Chars<R>) -> LexControl<T>;

pub struct TokenPatterns<R, T>
where
    R: Read,
{
    patterns: Vec<Box<DynTokenPattern<R, T>>>,
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
    pub fn with<F>(mut self, f: F) -> Self
    where
        F: Fn(&mut Chars<R>) -> LexControl<T> + 'static,
    {
        self.patterns.push(Box::new(f));
        self
    }
    pub fn skip<F>(mut self, f: F) -> Self
    where
        F: Fn(&mut Chars<R>) -> LexControl<T> + 'static,
    {
        self.patterns.push(Box::new(move |chars| match f(chars) {
            Ok(_) => skip(),
            e => e,
        }));
        self
    }
    pub fn tokenize(&self, reader: R) -> LexResult<Vec<Sp<T>>> {
        let mut args = Vec::new();
        let mut chars = Chars::new(reader);
        while chars.peek().is_ok() {
            let mut matched = false;
            for pattern in &self.patterns {
                let start_len = chars.history.len();
                let start_loc = chars.loc;
                match pattern(&mut chars) {
                    Ok(token) => {
                        args.push(start_loc.to(chars.loc).sp(token));
                        matched = true;
                        break;
                    }
                    Err(LexFailure::Skip) => {
                        matched = true;
                        break;
                    }
                    Err(LexFailure::Unmatched) => {
                        chars.loc = start_loc;
                        for _ in 0..(chars.history.len() - start_len) {
                            chars.put_back.extend(chars.history.pop());
                        }
                    }
                    Err(LexFailure::Error(e)) => return Err(e),
                }
            }
            if !matched {
                return Err(LexError::NoMatchingPattern);
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

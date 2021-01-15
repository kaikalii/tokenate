pub mod pattern;

use std::{
    fmt::{self, Display, Formatter},
    io::{self, Bytes, Read},
};

use unicode_reader::CodePoints;

pub use pattern::{CharPattern, Pattern};

const INVALID_INPUT_MAX_LEN: usize = 30;

#[derive(Debug, thiserror::Error)]
pub enum LexError {
    #[error("{0}")]
    IO(#[from] io::Error),
    #[error("Unable to tokenize {}", format_invalid_input(.0))]
    InvalidInput(String),
}

fn format_invalid_input(s: &str) -> String {
    format!(
        "Unable to tokenize {:?}{}",
        s,
        if s.len() >= INVALID_INPUT_MAX_LEN {
            "..."
        } else {
            ""
        }
    )
}

pub type LexResult<T> = Result<T, LexError>;
pub type TokenResult<T> = LexResult<Option<T>>;

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

pub struct Chars<R>
where
    R: Read,
{
    chars: CodePoints<Bytes<R>>,
    history: Vec<char>,
    cursor: usize,
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
    pub fn new(reader: R) -> Self {
        Chars {
            chars: reader.bytes().into(),
            history: Vec::new(),
            cursor: 0,
            loc: Loc::new(1, 1),
        }
    }
    pub fn loc(&self) -> Loc {
        self.loc
    }
    fn put_back(&mut self) {
        self.cursor -= 1;
    }
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
    pub fn take_iter(&mut self) -> impl Iterator<Item = io::Result<char>> + '_ {
        std::iter::from_fn(move || self.take().transpose())
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
    fn revert(&mut self, cursor: usize, loc: Loc) {
        self.loc = loc;
        self.cursor = cursor;
    }
    pub fn invalid_input<T>(&mut self) -> LexResult<T> {
        Err(LexError::InvalidInput(
            self.take_iter()
                .filter_map(Result::ok)
                .take(INVALID_INPUT_MAX_LEN)
                .collect(),
        ))
    }
    pub fn matching<P>(&mut self, pattern: &P) -> TokenResult<Sp<P::Token>>
    where
        P: Pattern<R>,
    {
        pattern.matching(self)
    }
    pub fn tokenize<M, S, T>(&mut self, matching: &M, skip: &S) -> LexResult<Vec<Sp<T>>>
    where
        M: Pattern<R, Token = T>,
        S: Pattern<R>,
        T: fmt::Debug,
    {
        let mut tokens = Vec::new();
        while self.peek()?.is_some() {
            let start_cursor = self.cursor;
            let start_loc = self.loc;
            if let Some(token) = self.matching(matching)? {
                tokens.push(token);
            } else if self.matching(skip)?.is_none() {
                self.revert(start_cursor, start_loc);
                return self.invalid_input();
            }
        }
        Ok(tokens)
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

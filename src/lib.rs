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

#[derive(Debug)]
pub enum LexError {
    IO(io::Error),
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
    pub fn new(reader: R) -> Self {
        Chars {
            chars: reader.bytes().into(),
            history: SmallVec::new(),
            cursor: 0,
            revert_trackers: 0,
            loc: Loc::new(1, 1),
        }
    }
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

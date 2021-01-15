use std::{
    fmt::{self, Display, Formatter},
    io::{self, Bytes, Read},
};

use unicode_reader::CodePoints;

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
    fn revert(&mut self, n: usize, loc: Loc) {
        self.loc = loc;
        for _ in 0..(self.history.len() - n) {
            self.put_back.extend(self.history.pop());
        }
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

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Default)]
pub struct Patterns<A, B> {
    pub first: A,
    pub second: B,
}

pub trait Pattern<R>
where
    R: Read,
{
    type Token;
    fn matching(&self, chars: &mut Chars<R>) -> TokenResult<Sp<Self::Token>>;
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
    fn or_chars<B>(self, other: B) -> Patterns<Self, CharPatternWrapper<B>>
    where
        Self: Sized,
        B: CharPattern,
    {
        Patterns {
            first: self,
            second: other.pattern(),
        }
    }
}

impl<R, F, T> Pattern<R> for F
where
    R: Read,
    F: Fn(&mut Chars<R>) -> TokenResult<T>,
{
    type Token = T;
    fn matching(&self, chars: &mut Chars<R>) -> TokenResult<Sp<Self::Token>> {
        let start_size = chars.history.len();
        let start_loc = chars.loc;
        match self(chars) {
            Ok(Some(token)) => Ok(Some(start_loc.to(chars.loc).sp(token))),
            Ok(None) => {
                chars.revert(start_size, start_loc);
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

pub trait CharPattern {
    fn matches(&self, c: char) -> bool;
    fn pattern(self) -> CharPatternWrapper<Self>
    where
        Self: Sized,
    {
        CharPatternWrapper(self)
    }
    fn or<B>(self, other: B) -> Patterns<CharPatternWrapper<Self>, CharPatternWrapper<B>>
    where
        Self: Sized,
        B: CharPattern,
    {
        Patterns {
            first: self.pattern(),
            second: other.pattern(),
        }
    }
}

impl CharPattern for char {
    fn matches(&self, c: char) -> bool {
        self == &c
    }
}

impl CharPattern for str {
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

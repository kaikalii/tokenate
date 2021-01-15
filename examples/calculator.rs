use tokenate::*;

#[derive(Debug, Clone)]
pub enum Token {
    Number(f32),
    Add,
    Sub,
    Mul,
    Div,
    OpenParen,
    CloseParen,
}

pub fn tokenize(input: &str) -> LexResult<Vec<Sp<Token>>> {
    let mut tokens = Vec::new();
    let mut chars = Chars::new(input.as_bytes());
    // Pattern for matching numbers
    let number = "-0123456789.e".pattern().parse::<f32>().map(Token::Number);
    // All patterns in order
    let patterns = number
        .or('+'.pattern().is(Token::Add))
        .or('-'.pattern().is(Token::Sub))
        .or('*'.pattern().is(Token::Mul))
        .or('/'.pattern().is(Token::Div))
        .or('('.pattern().is(Token::OpenParen))
        .or(')'.pattern().is(Token::CloseParen));
    while chars.peek()?.is_some() {
        if let Some(token) = chars.matching(&patterns)? {
            tokens.push(token);
        } else if chars.matching(&char::is_whitespace.pattern())?.is_none() {
            return chars.invalid_input();
        }
    }
    Ok(tokens)
}

fn main() {
    let inputs = ["1 + 2", "6 / 3 -4", "(-1 + 2) * (3 + 4)", "6.02e23"];
    for input in &inputs {
        println!("input: {}", input);
        println!("tokens:");
        for token in tokenize(input).unwrap() {
            println!("  {:<20} {}", format!("{:?}", token.data), token.span)
        }
        println!();
    }
}

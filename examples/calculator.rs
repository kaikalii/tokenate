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

fn main() {
    // Pattern for matching numbers
    let number = "-0123456789.e".pattern().parse::<f32>().map(Token::Number);
    // All patterns in order
    let patterns = '+'
        .pattern()
        .is(Token::Add)
        .or('-'.pattern().is(Token::Sub))
        .or('*'.pattern().is(Token::Mul))
        .or('/'.pattern().is(Token::Div))
        .or('('.pattern().is(Token::OpenParen))
        .or(')'.pattern().is(Token::CloseParen))
        .or(number);

    // A list of test inputs
    let inputs = ["1 + 2", "6 / 3 - 4.1", "(-1 + 2 / .3) * (3 + 4)", "6.02e23"];
    for input in &inputs {
        println!("input: {}", input);
        println!("tokens:");
        for token in Chars::new(input.as_bytes())
            .tokenize(&patterns, &char::is_whitespace.pattern())
            .unwrap()
        {
            println!("  {:<20} {}", format!("{:?}", token.data), token.span)
        }
        println!();
    }
}

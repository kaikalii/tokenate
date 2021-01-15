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
    let number = "-0123456789.e".any().parse::<f32>().map(Token::Number);
    // All patterns in order
    let patterns = '+'
        .any()
        .is(Token::Add)
        .or('-'.any().is(Token::Sub))
        .or('*'.any().is(Token::Mul))
        .or('/'.any().is(Token::Div))
        .or('('.any().is(Token::OpenParen))
        .or(')'.any().is(Token::CloseParen))
        .or(number);

    // A list of test inputs
    let inputs = ["1 + 2", "6 / 3 - 4.1", "(-1 + 2 / .3) * (3 + 4)", "6.02e23"];
    for input in &inputs {
        println!("input: {}", input);
        println!("tokens:");
        for token in Chars::new(input.as_bytes())
            .tokenize(&patterns, &char::is_whitespace.any())
            .unwrap()
        {
            println!("  {:<20} {}", format!("{:?}", token.data), token.span)
        }
        println!();
    }
}

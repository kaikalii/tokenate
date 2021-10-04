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
    let int = "0123456789".any();
    let signed_int = "+-".take(0..=1).concat(int.clone());
    let number = signed_int
        .clone()
        .or_default()
        .concat(".".take_exact(1).concat(int).or_default())
        .concat("e".any().concat(signed_int).or_default())
        .parse::<f32>()
        .map(Token::Number);
    // All patterns in order
    let patterns = number
        .or('+'.is(Token::Add))
        .or('-'.is(Token::Sub))
        .or('*'.is(Token::Mul))
        .or('/'.is(Token::Div))
        .or('('.is(Token::OpenParen))
        .or(')'.is(Token::CloseParen));

    // A list of test inputs
    let inputs = [
        "1 + 2",
        "6./3 - 4.1",
        "(-1 + 2 / .3) * (3 + 4)",
        "6.02e23",
        "1--4 * .1e3",
    ];
    for input in inputs {
        println!("input: {}", input);
        println!("tokens:");
        for token in Chars::new(input.as_bytes())
            .tokenize(&patterns, &char::is_whitespace.any())
            .unwrap_or_else(|e| panic!("{}", e))
        {
            println!("  {:<20} {}", format!("{:?}", token.data), token.span)
        }
        println!();
    }
}

/*!
This example implements a simple command-line style argument parser
where arguments are seperated by whitespace unless delimited by double quotes
*/

use std::io;

use tokenate::*;

/// Try to tokenize a quoted arg
fn quoted_arg<R: io::Read>(chars: &mut Chars<R>) -> TokenResult<String> {
    chars.take_if(|c| c == '"')?.or_unmatched()?;
    let mut arg = String::new();
    let mut escaped = false;
    while let Some(c) = chars.take()? {
        match c {
            '"' if escaped.take() => arg.push('"'),
            '"' => break,
            c => arg.push(c),
        }
    }
    Ok(arg)
}

/// Try to tokenize an unquoted arg
fn unquoted_arg<R: io::Read>(chars: &mut Chars<R>) -> TokenResult<String> {
    let c = chars.take_if(|c| !c.is_whitespace())?.or_unmatched()?;
    let mut arg = String::from(c);
    while let Some(c) = chars.take_if(|c| !c.is_whitespace())? {
        arg.push(c);
    }
    Ok(arg)
}

fn main() {
    let inputs = [
        "arg1 arg2 arg3",
        r#"arg1 "arg2a arg2b" arg3"#,
        r#"cd "C:\Program Files""#,
    ];
    let patterns = TokenPatterns::new()
        .with(quoted_arg)
        .with(unquoted_arg)
        .skip(pattern::whitespace);
    for &input in &inputs {
        println!("input: {}", input);
        println!("args: ");
        for arg in patterns.tokenize(input.as_bytes()).unwrap() {
            println!("  {:<20}{}", arg.data, arg.span)
        }
        println!();
    }
}

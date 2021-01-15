/*!
This example implements a simple command-line style argument parser
where arguments are seperated by whitespace unless delimited by double quotes
*/

use std::io::Read;

use tokenate::*;

/// Try to tokenize a quoted arg
fn quoted_arg<R: Read>(chars: &mut Chars<R>) -> TokenResult<String> {
    Ok(if chars.take_if(|c| c == '"')?.is_some() {
        let mut arg = String::new();
        let mut escaped = false;
        while let Some(c) = chars.take()? {
            match c {
                '"' if escaped.take() => arg.push('"'),
                '"' => break,
                c => arg.push(c),
            }
        }
        Some(arg)
    } else {
        None
    })
}

fn tokenize_args<R: Read>(chars: &mut Chars<R>) -> LexResult<Vec<Sp<String>>> {
    let mut args = Vec::new();
    let patterns = quoted_arg.or_chars(|c: char| !c.is_whitespace());
    while chars.peek()?.is_some() {
        if let Some(arg) = chars.matching(&patterns)? {
            args.push(arg);
        } else {
            chars.take()?;
        }
    }
    Ok(args)
}

fn main() {
    let inputs = [
        "arg1 arg2 arg3",
        r#"arg1 "arg2a arg2b" arg3"#,
        r#"cd "C:\Program Files""#,
    ];
    for &input in &inputs {
        println!("input: {}", input);
        let mut chars = Chars::new(input.as_bytes());
        println!("args: ");
        for arg in tokenize_args(&mut chars).unwrap() {
            println!("  {:<20}{}", arg.data, arg.span)
        }
        println!();
    }
}

/*!
This example implements a simple command-line style argument parser
where arguments are seperated by whitespace unless delimited by double quotes
*/

use std::io;

use tokenate::*;

/// Try to tokeniz a quoted arg
fn quoted_arg<R: io::Read>(chars: &mut Chars<R>) -> LexControl<String> {
    chars.take_if(|c| c == '"')?;
    let mut arg = String::new();
    let mut escaped = false;
    while let Ok(c) = chars.take() {
        match c {
            '\\' if escaped.take() => arg.push('\\'),
            '\\' => escaped = true,
            '"' if escaped.take() => arg.push('"'),
            '"' => break,
            c => arg.push(c),
        }
    }
    Ok(arg)
}

/// Try to tokeniz an unquoted arg
fn unquoted_arg<R: io::Read>(chars: &mut Chars<R>) -> LexControl<String> {
    let c = if let Ok(c) = chars.take_if(|c| !c.is_whitespace()) {
        c
    } else {
        return unmatched();
    };
    let mut arg = String::from(c);
    while let Ok(c) = chars.take_if(|c| !c.is_whitespace()) {
        arg.push(c);
    }
    Ok(arg)
}

fn main() {
    let patterns = TokenPatterns::new()
        .with(quoted_arg)
        .with(unquoted_arg)
        .skip(Chars::whitespace);
    for &input in &[
        "arg1 arg2 arg3",
        r#"arg1 "arg2a arg2b" arg3"#,
        r#"cd "C:\\Program Files""#,
    ] {
        println!("input: {}", input);
        println!("args: ");
        for arg in patterns.tokenize(input.as_bytes()).unwrap() {
            println!("  {}", arg.data)
        }
        println!();
    }
}

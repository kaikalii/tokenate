/*!
This example implements a simple command-line style argument parser
where arguments are seperated by whitespace unless delimited by double quotes
*/

use tokenate::*;

/// Try to tokenize a quoted arg
fn quoted_arg(chars: &mut Chars) -> TokenResult<String> {
    Ok(if chars.take_if(|c| c == '"')?.is_some() {
        let mut arg = String::new();
        while let Some(c) = chars.take()? {
            match c {
                '"' => break,
                c => arg.push(c),
            }
        }
        Some(arg)
    } else {
        None
    })
}

fn main() {
    // The patterns used to match args
    // `quoted_arg` will match arguments surrounded in quotes
    // `pattern::not_whitespace` will match arguments without quotes
    // The `or` combinator is a method of the `Pattern` trait, and it creates a new pattern
    // which will try to match either of the two patterns in order
    let patterns = quoted_arg.or_chars(pattern::not_whitespace);
    // A list of test inputs
    let inputs = [
        "arg1 arg2 arg3",
        r#"arg1 "arg2a arg2b" arg3"#,
        r#"cd "C:\Program Files""#,
    ];
    for input in inputs {
        println!("input: {}", input);
        println!("args: ");

        for arg in Chars::new(input.as_bytes())
            .tokenize(&patterns, &char::is_whitespace.any())
            .unwrap()
        {
            println!("  {:<20}{}", arg.data, arg.span)
        }
        println!();
    }
}

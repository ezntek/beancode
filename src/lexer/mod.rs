pub mod types;

use std::{fs::File, io::BufReader};

use types::*;

pub struct Lexer {
    filename: String,
    reader: BufReader<File>,
    cur: usize,
    row: usize,
    bol: usize,
}

impl Lexer {
    pub fn main(filename: String) -> Self {
        let file = File::open(&filename).expect("failed to open file");
        let reader = BufReader::new(file);
        Lexer {
            filename,
            reader,
            cur: 0,
            row: 1,
            bol: 0,
        }
    }

    pub fn tokenize() -> Vec<Token> {
        let mut res = Vec::new();
        res
    }
}

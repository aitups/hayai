//! GBNF grammar-constrained decoding.
//!
//! Implements the llama.cpp `.gbnf` grammar syntax (literals, character classes,
//! alternation, grouping, `* + ?` repetition, rule references). The grammar is
//! compiled to an NFA whose accept/return edges are resolved with a call stack,
//! so **right-recursive** grammars (e.g. JSON) work. Left recursion is bounded
//! and fails closed rather than looping.
//!
//! The public surface is small: parse once, then per generated token call
//! [`Grammar::advance_str`] (allowed?) and [`Grammar::is_accepting`] (may stop?).

use std::collections::{BTreeSet, HashMap};

/// Unpatched transition target.
const HOLE: u32 = u32::MAX;
/// Bound on the grammar call-stack depth (guards against left recursion).
const MAX_STACK: usize = 256;

/// Inclusive Unicode range set, optionally negated (`[^...]`).
#[derive(Debug, Clone, PartialEq)]
pub struct CharClass {
    ranges: Vec<(char, char)>,
    negated: bool,
}

impl CharClass {
    pub fn matches(&self, c: char) -> bool {
        let inside = self.ranges.iter().any(|&(lo, hi)| c >= lo && c <= hi);
        inside != self.negated
    }
}

#[derive(Debug, Clone)]
enum Ast {
    Empty,
    Char(char),
    Class(CharClass),
    Seq(Vec<Ast>),
    Alt(Vec<Ast>),
    /// `min..=max` repetitions (`max = None` → unbounded).
    Repeat(Box<Ast>, u32, Option<u32>),
    Ref(String),
}

#[derive(Debug, Clone)]
enum Elem {
    Eps(Vec<u32>),
    Char(char, Vec<u32>),
    Class(CharClass, Vec<u32>),
    Call { rule: u32, ret: u32 },
    /// End of a rule body: return to the caller via the runtime stack.
    End,
}

/// One NFA fragment: a start position plus exit slots to patch to a continuation.
#[derive(Clone)]
struct Frag {
    start: u32,
    /// `(elem index, target slot)` holes that must be patched to the next fragment.
    exits: Vec<(u32, usize)>,
}

struct Builder {
    elems: Vec<Elem>,
    rule_entry: Vec<u32>,
    rule_ids: HashMap<String, u32>,
}

impl Builder {
    fn emit(&mut self, e: Elem) -> u32 {
        self.elems.push(e);
        (self.elems.len() - 1) as u32
    }

    fn patch(&mut self, exits: &[(u32, usize)], target: u32) {
        for &(idx, slot) in exits {
            match &mut self.elems[idx as usize] {
                Elem::Eps(ts) | Elem::Char(_, ts) | Elem::Class(_, ts) => ts[slot] = target,
                Elem::Call { ret, .. } => *ret = target,
                Elem::End => {}
            }
        }
    }

    fn empty(&mut self) -> Frag {
        let idx = self.emit(Elem::Eps(vec![HOLE]));
        Frag {
            start: idx,
            exits: vec![(idx, 0)],
        }
    }

    fn concat(&mut self, a: Frag, b: Frag) -> Frag {
        self.patch(&a.exits, b.start);
        Frag {
            start: a.start,
            exits: b.exits,
        }
    }

    fn concat_all(&mut self, frags: Vec<Frag>) -> Option<Frag> {
        let mut it = frags.into_iter();
        let mut acc = it.next()?;
        for f in it {
            acc = self.concat(acc, f);
        }
        Some(acc)
    }

    fn alt(&mut self, frags: Vec<Frag>) -> Frag {
        let head = self.emit(Elem::Eps(vec![]));
        let mut exits = Vec::new();
        if let Elem::Eps(ts) = &mut self.elems[head as usize] {
            for f in &frags {
                ts.push(f.start);
            }
        }
        for f in frags {
            exits.extend(f.exits);
        }
        Frag {
            start: head,
            exits,
        }
    }

    fn compile(&mut self, ast: &Ast) -> Frag {
        match ast {
            Ast::Empty => self.empty(),
            Ast::Char(c) => {
                let idx = self.emit(Elem::Char(*c, vec![HOLE]));
                Frag {
                    start: idx,
                    exits: vec![(idx, 0)],
                }
            }
            Ast::Class(cc) => {
                let idx = self.emit(Elem::Class(cc.clone(), vec![HOLE]));
                Frag {
                    start: idx,
                    exits: vec![(idx, 0)],
                }
            }
            Ast::Seq(items) => {
                let frags = items.iter().map(|a| self.compile(a)).collect();
                self.concat_all(frags).unwrap_or_else(|| self.empty())
            }
            Ast::Alt(items) => {
                let frags = items.iter().map(|a| self.compile(a)).collect();
                self.alt(frags)
            }
            Ast::Ref(name) => {
                let rule = self.rule_ids.get(name).copied().unwrap_or(0);
                let idx = self.emit(Elem::Call { rule, ret: HOLE });
                Frag {
                    start: idx,
                    exits: vec![(idx, 0)],
                }
            }
            Ast::Repeat(inner, min, max) => self.compile_repeat(inner, *min, *max),
        }
    }

    fn compile_repeat(&mut self, inner: &Ast, min: u32, max: Option<u32>) -> Frag {
        let mut parts: Vec<Frag> = Vec::new();
        for _ in 0..min {
            parts.push(self.compile(inner));
        }
        match max {
            None => {
                // `min` copies, then loop: each iteration runs a fresh copy or exits.
                let copy = self.compile(inner);
                let loop_head = self.emit(Elem::Eps(vec![]));
                let exit_idx = self.emit(Elem::Eps(vec![HOLE]));
                if let Elem::Eps(ts) = &mut self.elems[loop_head as usize] {
                    ts.push(copy.start);
                    ts.push(exit_idx);
                }
                self.patch(&copy.exits, loop_head);
                let start = match self.concat_all(parts) {
                    Some(seq) => {
                        self.patch(&seq.exits, loop_head);
                        seq.start
                    }
                    None => loop_head,
                };
                Frag {
                    start,
                    exits: vec![(exit_idx, 0)],
                }
            }
            Some(m) => {
                for _ in min..m {
                    let copy = self.compile(inner);
                    let empty = self.empty();
                    let opt = self.alt(vec![copy, empty]);
                    parts.push(opt);
                }
                self.concat_all(parts).unwrap_or_else(|| self.empty())
            }
        }
    }
}

/// Compiled grammar. Cheap to clone; share across generations with `Arc`.
#[derive(Clone, Debug)]
pub struct Grammar {
    elems: Vec<Elem>,
    rule_entry: Vec<u32>,
    root_index: usize,
}

/// A grammar parse/compile error.
#[derive(Debug, Clone, thiserror::Error)]
#[error("{0}")]
pub struct GrammarError(pub String);

/// Current parser state: the set of `(position, return-stack)` items reachable
/// after consuming the emitted text, plus whether the grammar may stop here.
#[derive(Clone, PartialEq, Eq, Debug, Default)]
pub struct GrammarState {
    items: BTreeSet<(u32, Vec<u32>)>,
    accepting: bool,
}

impl GrammarState {
    pub fn is_dead(&self) -> bool {
        self.items.is_empty()
    }
}

impl Grammar {
    pub fn parse(src: &str) -> Result<Self, GrammarError> {
        let rules = Parser::new(src).parse_grammar()?;
        let root_index = rules
            .iter()
            .position(|(n, _)| n == "root")
            .ok_or_else(|| GrammarError("grammar has no `root` rule".into()))?;
        let mut rule_ids = HashMap::new();
        for (i, (name, _)) in rules.iter().enumerate() {
            rule_ids.insert(name.clone(), i as u32);
        }
        let mut b = Builder {
            elems: Vec::new(),
            rule_entry: vec![0; rules.len()],
            rule_ids,
        };
        for (i, (_, ast)) in rules.iter().enumerate() {
            let frag = b.compile(ast);
            let end = b.emit(Elem::End);
            b.patch(&frag.exits, end);
            b.rule_entry[i] = frag.start;
        }
        Ok(Self {
            elems: b.elems,
            rule_entry: b.rule_entry,
            root_index,
        })
    }

    pub fn initial_state(&self) -> GrammarState {
        let root = self.rule_entry[self.root_index];
        self.closure(vec![(root, Vec::new())])
    }

    /// Whether the grammar may stop (accept an empty continuation) in `state`.
    pub fn is_accepting(&self, state: &GrammarState) -> bool {
        state.accepting
    }

    /// Advance by a whole token string. Returns `None` if the token would leave
    /// the grammar (it is not a valid prefix of any accepted sentence).
    pub fn advance_str(&self, state: &GrammarState, text: &str) -> Option<GrammarState> {
        let mut cur = state.clone();
        for ch in text.chars() {
            cur = self.step(&cur, ch);
            if cur.is_dead() {
                return None;
            }
        }
        Some(cur)
    }

    fn step(&self, state: &GrammarState, ch: char) -> GrammarState {
        let mut seed: Vec<(u32, Vec<u32>)> = Vec::new();
        for (pos, st) in &state.items {
            match &self.elems[*pos as usize] {
                Elem::Char(c, ts) if *c == ch => {
                    for &t in ts {
                        seed.push((t, st.clone()));
                    }
                }
                Elem::Class(cc, ts) if cc.matches(ch) => {
                    for &t in ts {
                        seed.push((t, st.clone()));
                    }
                }
                _ => {}
            }
        }
        if seed.is_empty() {
            GrammarState::default()
        } else {
            self.closure(seed)
        }
    }

    fn closure(&self, seed: Vec<(u32, Vec<u32>)>) -> GrammarState {
        let mut items: BTreeSet<(u32, Vec<u32>)> = BTreeSet::new();
        let mut accepting = false;
        let mut work = seed;
        while let Some((pos, st)) = work.pop() {
            if st.len() > MAX_STACK {
                continue; // guard against left recursion: fail closed
            }
            if !items.insert((pos, st.clone())) {
                continue;
            }
            match &self.elems[pos as usize] {
                Elem::Eps(ts) => {
                    for &t in ts {
                        if t != HOLE {
                            work.push((t, st.clone()));
                        }
                    }
                }
                Elem::Call { rule, ret } => {
                    let mut s2 = st.clone();
                    s2.push(*ret);
                    work.push((self.rule_entry[*rule as usize], s2));
                }
                Elem::End => {
                    if let Some(r) = st.last().copied() {
                        let mut s2 = st.clone();
                        s2.pop();
                        work.push((r, s2));
                    } else {
                        accepting = true;
                    }
                }
                Elem::Char(..) | Elem::Class(..) => {}
            }
        }
        GrammarState { items, accepting }
    }
}

// ─────────────────────────── parser ───────────────────────────

#[derive(Debug, Clone, PartialEq)]
enum Tok {
    Ident(String),
    Lit(String),
    Class(CharClass),
    LParen,
    RParen,
    Pipe,
    Star,
    Plus,
    Opt,
    Define,
    Newline,
}

struct Parser {
    toks: Vec<Tok>,
    pos: usize,
}

impl Parser {
    fn new(src: &str) -> Self {
        Self {
            toks: lex(src),
            pos: 0,
        }
    }

    fn peek(&self) -> Option<&Tok> {
        self.toks.get(self.pos)
    }

    fn next(&mut self) -> Option<Tok> {
        let t = self.toks.get(self.pos).cloned();
        self.pos += 1;
        t
    }

    fn skip_newlines(&mut self) {
        while matches!(self.peek(), Some(Tok::Newline)) {
            self.pos += 1;
        }
    }

    fn parse_grammar(&mut self) -> Result<Vec<(String, Ast)>, GrammarError> {
        let mut rules = Vec::new();
        loop {
            self.skip_newlines();
            let name = match self.next() {
                Some(Tok::Ident(n)) => n,
                None => break,
                Some(t) => return Err(GrammarError(format!("expected rule name, got {t:?}"))),
            };
            match self.next() {
                Some(Tok::Define) => {}
                other => {
                    return Err(GrammarError(format!(
                        "expected `::=` after rule `{name}`, got {other:?}"
                    )))
                }
            }
            let body = self.parse_alts()?;
            rules.push((name, body));
            // A rule ends at a newline (or EOF).
            match self.peek() {
                Some(Tok::Newline) | None => {}
                Some(t) => {
                    return Err(GrammarError(format!(
                        "unexpected {t:?} after rule body (rules must be newline-separated)"
                    )))
                }
            }
        }
        Ok(rules)
    }

    fn parse_alts(&mut self) -> Result<Ast, GrammarError> {
        let mut alts = vec![self.parse_seq()?];
        while matches!(self.peek(), Some(Tok::Pipe)) {
            self.pos += 1;
            alts.push(self.parse_seq()?);
        }
        Ok(if alts.len() == 1 {
            alts.pop().unwrap()
        } else {
            Ast::Alt(alts)
        })
    }

    fn parse_seq(&mut self) -> Result<Ast, GrammarError> {
        let mut items = Vec::new();
        while let Some(t) = self.peek() {
            match t {
                Tok::Pipe | Tok::RParen | Tok::Newline => break,
                _ => items.push(self.parse_repeat()?),
            }
        }
        Ok(match items.len() {
            0 => Ast::Empty,
            1 => items.pop().unwrap(),
            _ => Ast::Seq(items),
        })
    }

    fn parse_repeat(&mut self) -> Result<Ast, GrammarError> {
        let atom = self.parse_atom()?;
        Ok(match self.peek() {
            Some(Tok::Star) => {
                self.pos += 1;
                Ast::Repeat(Box::new(atom), 0, None)
            }
            Some(Tok::Plus) => {
                self.pos += 1;
                Ast::Repeat(Box::new(atom), 1, None)
            }
            Some(Tok::Opt) => {
                self.pos += 1;
                Ast::Repeat(Box::new(atom), 0, Some(1))
            }
            _ => atom,
        })
    }

    fn parse_atom(&mut self) -> Result<Ast, GrammarError> {
        match self.next() {
            Some(Tok::Lit(s)) => {
                // A multi-char literal becomes a sequence.
                let chars: Vec<Ast> = s.chars().map(Ast::Char).collect();
                Ok(match chars.len() {
                    0 => Ast::Empty,
                    1 => chars.into_iter().next().unwrap(),
                    _ => Ast::Seq(chars),
                })
            }
            Some(Tok::Class(cc)) => Ok(Ast::Class(cc)),
            Some(Tok::Ident(name)) => Ok(Ast::Ref(name)),
            Some(Tok::LParen) => {
                let inner = self.parse_alts()?;
                match self.next() {
                    Some(Tok::RParen) => Ok(inner),
                    other => Err(GrammarError(format!("expected `)`, got {other:?}"))),
                }
            }
            other => Err(GrammarError(format!("unexpected token {other:?}"))),
        }
    }
}

fn lex(src: &str) -> Vec<Tok> {
    let chars: Vec<char> = src.chars().collect();
    let mut i = 0usize;
    let mut out = Vec::new();
    let n = chars.len();
    while i < n {
        let c = chars[i];
        if c == '\n' {
            out.push(Tok::Newline);
            i += 1;
            continue;
        }
        if c.is_whitespace() {
            i += 1;
            continue;
        }
        if c == '#' {
            while i < n && chars[i] != '\n' {
                i += 1;
            }
            continue;
        }
        match c {
            '(' => {
                out.push(Tok::LParen);
                i += 1;
            }
            ')' => {
                out.push(Tok::RParen);
                i += 1;
            }
            '|' => {
                out.push(Tok::Pipe);
                i += 1;
            }
            '*' => {
                out.push(Tok::Star);
                i += 1;
            }
            '+' => {
                out.push(Tok::Plus);
                i += 1;
            }
            '?' => {
                out.push(Tok::Opt);
                i += 1;
            }
            ':' => {
                if i + 2 < n && chars[i + 1] == ':' && chars[i + 2] == '=' {
                    out.push(Tok::Define);
                    i += 3;
                } else {
                    // stray ':' → treat as literal char
                    out.push(Tok::Lit(":".into()));
                    i += 1;
                }
            }
            '"' => {
                let (s, ni) = lex_string(&chars, i + 1);
                out.push(Tok::Lit(s));
                i = ni;
            }
            '[' => {
                let (cc, ni) = lex_class(&chars, i + 1);
                out.push(Tok::Class(cc));
                i = ni;
            }
            _ if is_ident_start(c) => {
                let start = i;
                while i < n && is_ident_char(chars[i]) {
                    i += 1;
                }
                out.push(Tok::Ident(chars[start..i].iter().collect()));
            }
            _ => {
                // Bare character (GBNF allows e.g. `,` unquoted).
                out.push(Tok::Lit(c.to_string()));
                i += 1;
            }
        }
    }
    out
}

fn is_ident_start(c: char) -> bool {
    c.is_ascii_alphabetic() || c == '_'
}
fn is_ident_char(c: char) -> bool {
    c.is_ascii_alphanumeric() || c == '_' || c == '-'
}

fn lex_string(chars: &[char], mut i: usize) -> (String, usize) {
    let mut s = String::new();
    let n = chars.len();
    while i < n {
        let c = chars[i];
        if c == '"' {
            return (s, i + 1);
        }
        if c == '\\' && i + 1 < n {
            let (ch, ni) = decode_escape(chars, i + 1);
            s.push(ch);
            i = ni;
            continue;
        }
        s.push(c);
        i += 1;
    }
    (s, i)
}

fn lex_class(chars: &[char], mut i: usize) -> (CharClass, usize) {
    let n = chars.len();
    let mut negated = false;
    if i < n && chars[i] == '^' {
        negated = true;
        i += 1;
    }
    let mut ranges: Vec<(char, char)> = Vec::new();
    while i < n && chars[i] != ']' {
        let (lo, ni) = read_class_char(chars, i);
        i = ni;
        if i + 1 < n && chars[i] == '-' && chars[i + 1] != ']' {
            let (hi, ni2) = read_class_char(chars, i + 1);
            i = ni2;
            ranges.push((lo, hi));
        } else {
            ranges.push((lo, lo));
        }
    }
    if i < n {
        i += 1; // consume ']'
    }
    (CharClass { ranges, negated }, i)
}

fn read_class_char(chars: &[char], i: usize) -> (char, usize) {
    if chars[i] == '\\' && i + 1 < chars.len() {
        decode_escape(chars, i + 1)
    } else {
        (chars[i], i + 1)
    }
}

fn decode_escape(chars: &[char], i: usize) -> (char, usize) {
    let n = chars.len();
    match chars.get(i).copied() {
        Some('n') => ('\n', i + 1),
        Some('r') => ('\r', i + 1),
        Some('t') => ('\t', i + 1),
        Some('0') => ('\0', i + 1),
        Some('\\') => ('\\', i + 1),
        Some('"') => ('"', i + 1),
        Some('\'') => ('\'', i + 1),
        Some(']') => (']', i + 1),
        Some('-') => ('-', i + 1),
        Some('[') => ('[', i + 1),
        Some('x') => {
            let mut v = 0u32;
            let mut k = i + 1;
            for _ in 0..2 {
                if let Some(d) = chars.get(k).and_then(|c| c.to_digit(16)) {
                    v = v * 16 + d;
                    k += 1;
                }
            }
            (char::from_u32(v).unwrap_or('\u{FFFD}'), k)
        }
        Some('u') => {
            let mut v = 0u32;
            let mut k = i + 1;
            for _ in 0..4 {
                if let Some(d) = chars.get(k).and_then(|c| c.to_digit(16)) {
                    v = v * 16 + d;
                    k += 1;
                }
            }
            (char::from_u32(v).unwrap_or('\u{FFFD}'), k)
        }
        Some(c) => (c, i + 1),
        None => ('\\', n),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn accepts(g: &Grammar, s: &str) -> bool {
        let st = g.initial_state();
        match g.advance_str(&st, s) {
            Some(st2) => g.is_accepting(&st2),
            None => false,
        }
    }

    fn prefix_ok(g: &Grammar, s: &str) -> bool {
        let st = g.initial_state();
        g.advance_str(&st, s).is_some()
    }

    #[test]
    fn literal_alternation_classes() {
        let g = Grammar::parse(r#"root ::= "a" | "b" | [0-9]"#).unwrap();
        assert!(accepts(&g, "a"));
        assert!(accepts(&g, "b"));
        assert!(accepts(&g, "7"));
        assert!(!accepts(&g, "c"));
        assert!(!prefix_ok(&g, "aa"));
    }

    #[test]
    fn repetition_and_grouping() {
        let g = Grammar::parse(r#"root ::= ("ab")+ "c"?"#).unwrap();
        assert!(accepts(&g, "ab"));
        assert!(accepts(&g, "ababc"));
        assert!(accepts(&g, "ababc"));
        assert!(!accepts(&g, "aba"));
    }

    #[test]
    fn char_class_ranges_and_negation() {
        let g = Grammar::parse(r#"root ::= [a-c]+"#).unwrap();
        assert!(accepts(&g, "abc"));
        assert!(!accepts(&g, "abd"));
        let g2 = Grammar::parse(r#"root ::= [^0-9]+"#).unwrap();
        assert!(accepts(&g2, "abc"));
        assert!(!accepts(&g2, "a1"));
    }

    #[test]
    fn right_recursion_json_like() {
        // A JSON-ish value grammar (right-recursive via `value` inside arrays).
        let g = Grammar::parse(
            r#"
root  ::= value
value ::= object | array | string | number
object ::= "{" (pair ("," pair)*)? "}"
pair  ::= string ":" value
array ::= "[" (value ("," value)*)? "]"
string ::= "\"" [a-z]* "\""
number ::= [0-9]+
"#,
        )
        .unwrap();
        assert!(accepts(&g, "42"));
        assert!(accepts(&g, "\"abc\""));
        assert!(accepts(&g, "[1,2,3]"));
        assert!(accepts(&g, "{\"a\":[9]}"));
        assert!(accepts(&g, "[{\"k\":[1,2]},3]"));
        assert!(!accepts(&g, "[1,2"));
        // A half-open array is a valid prefix but not accepting.
        let st = g.advance_str(&g.initial_state(), "[1,").unwrap();
        assert!(!g.is_accepting(&st));
    }

    #[test]
    fn escapes_in_literals_and_classes() {
        let g = Grammar::parse(r#"root ::= "\n" | "\t" | [\x41-\x43]"#).unwrap();
        assert!(accepts(&g, "\n"));
        assert!(accepts(&g, "\t"));
        assert!(accepts(&g, "B"));
    }
}

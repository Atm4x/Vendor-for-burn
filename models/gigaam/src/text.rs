pub fn normalize_raw_text(input: &str) -> String {
    let normalized = input.replace('ё', "е").replace('Ё', "Е").to_lowercase();
    let chars: Vec<char> = normalized
        .chars()
        .map(|character| match character {
            '’' | '‘' | 'ʻ' | 'ʼ' => '\'',
            other => other,
        })
        .collect();
    let mut output = String::with_capacity(input.len());
    for (index, &character) in chars.iter().enumerate() {
        match character {
            '\'' if index > 0
                && index + 1 < chars.len()
                && chars[index - 1].is_alphanumeric()
                && chars[index + 1].is_alphanumeric() =>
            {
                output.push(character)
            }
            '\'' => {}
            c if c.is_alphanumeric() || c.is_whitespace() => output.push(c),
            c if is_dash_punctuation(c) => output.push(' '),
            _ => {}
        }
    }
    output.split_whitespace().collect::<Vec<_>>().join(" ")
}

fn is_dash_punctuation(character: char) -> bool {
    matches!(
        character,
        '-' | '\u{58a}' | '\u{5be}' | '\u{1400}' | '\u{1806}'
            | '\u{2010}'..='\u{2015}' | '\u{2e17}' | '\u{2e1a}'
            | '\u{2e3a}'..='\u{2e3b}' | '\u{2e40}' | '\u{2e5d}'
            | '\u{301c}' | '\u{3030}' | '\u{30a0}'
            | '\u{fe31}'..='\u{fe32}' | '\u{fe58}' | '\u{fe63}'
            | '\u{ff0d}' | '\u{10ead}'
    )
}

#[cfg(test)]
mod tests {
    use super::normalize_raw_text;

    #[test]
    fn python_normalization_cases() {
        let cases = [
            ("Hello WORLD", "hello world"),
            ("MiXeD CaSe", "mixed case"),
            ("foo   bar\t\nbaz", "foo bar baz"),
            ("  leading and trailing  ", "leading and trailing"),
            ("Hello, world!", "hello world"),
            ("из-за дождя", "из за дождя"),
            ("word-internal hyphen", "word internal hyphen"),
            ("a.b,c-d", "abc d"),
            ("...!!!", ""),
            ("дом 9400 кв", "дом 9400 кв"),
            ("Привет, мир!", "привет мир"),
            ("ЁЖИК ёлка", "ежик елка"),
            ("გამარჯობა, მსოფლიო!", "გამარჯობა მსოფლიო"),
            ("ქართული ენა — 2024", "ქართული ენა 2024"),
            ("Привет, world!", "привет world"),
            ("Окей, let's go!", "окей let's go"),
            ("I don't know.", "i don't know"),
            ("We're here, can't stop!", "we're here can't stop"),
            ("O'Brien", "o'brien"),
            ("it’s fine", "it's fine"),
            ("o‘zbek g‘oz", "o'zbek g'oz"),
            ("'quoted' text", "quoted text"),
            ("dogs' toys", "dogs toys"),
            (
                "Oʻzbekiston — mustaqil davlat!",
                "o'zbekiston mustaqil davlat",
            ),
            ("Сәлем, әлем!", "сәлем әлем"),
            ("Қазақстан Республикасы", "қазақстан республикасы"),
            ("Салам, дүйнө!", "салам дүйнө"),
            ("Кыргыз Республикасы — 2024", "кыргыз республикасы 2024"),
            ("Բարև, աշխարհ։", "բարև աշխարհ"),
            ("Հայաստանի Հանրապետություն", "հայաստանի հանրապետություն"),
            ("100 մետրանոց նավը։", "100 մետրանոց նավը"),
            ("Ո՞վ է այնտեղ՝ Արամը։", "ով է այնտեղ արամը"),
            (
                "մյուս դեպքերում, գրանցվելու՜ կարիք.",
                "մյուս դեպքերում գրանցվելու կարիք",
            ),
            ("", ""),
            ("   ", ""),
        ];
        for (input, expected) in cases {
            assert_eq!(normalize_raw_text(input), expected, "{input:?}");
        }
    }
}

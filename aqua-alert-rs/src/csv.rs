//! Minimal, dependency-free RFC4180 CSV parser (quoted fields, embedded commas,
//! escaped quotes, CRLF or LF). Used by the offline sample loader and the
//! USGS/EPA Water Quality Portal client.

/// Parse CSV text into rows of fields.
pub fn parse(text: &str) -> Vec<Vec<String>> {
    let mut rows: Vec<Vec<String>> = Vec::new();
    let mut row: Vec<String> = Vec::new();
    let mut field = String::new();
    let mut in_quotes = false;
    let mut chars = text.chars().peekable();

    while let Some(c) = chars.next() {
        if in_quotes {
            if c == '"' {
                if chars.peek() == Some(&'"') {
                    field.push('"');
                    chars.next();
                } else {
                    in_quotes = false;
                }
            } else {
                field.push(c);
            }
        } else {
            match c {
                '"' => in_quotes = true,
                ',' => row.push(std::mem::take(&mut field)),
                '\r' => { /* swallow; the following '\n' ends the row */ }
                '\n' => {
                    row.push(std::mem::take(&mut field));
                    rows.push(std::mem::take(&mut row));
                }
                _ => field.push(c),
            }
        }
    }
    // flush a final field/row when the text does not end in a newline
    if !field.is_empty() || !row.is_empty() {
        row.push(field);
        rows.push(row);
    }
    rows
}

/// Parse into a header map (column -> value) per record, skipping blank lines.
pub fn parse_records(text: &str) -> Vec<std::collections::HashMap<String, String>> {
    let rows = parse(text);
    let mut it = rows.into_iter().filter(|r| !(r.len() == 1 && r[0].trim().is_empty()));
    let header = match it.next() {
        Some(h) => h,
        None => return Vec::new(),
    };
    it.map(|r| {
        header
            .iter()
            .enumerate()
            .map(|(i, col)| (col.clone(), r.get(i).cloned().unwrap_or_default()))
            .collect()
    })
    .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn handles_quotes_and_embedded_commas() {
        let text = "a,b\r\n1,\"x, y\"\r\n\"he said \"\"hi\"\"\",2\r\n";
        let rows = parse(text);
        assert_eq!(rows[0], vec!["a", "b"]);
        assert_eq!(rows[1], vec!["1", "x, y"]);
        assert_eq!(rows[2], vec!["he said \"hi\"", "2"]);
    }

    #[test]
    fn records_map_columns() {
        let text = "id,ts,v\nK1,2026-10-06T06:00,3.8\n";
        let recs = parse_records(text);
        assert_eq!(recs.len(), 1);
        assert_eq!(recs[0]["id"], "K1");
        assert_eq!(recs[0]["v"], "3.8");
    }
}

//! Values as HTML, for notebooks and pages that can show more than text: a table as a table,
//! a tensor as a grid of numbers.

use arrow::array::{Array, ArrayRef};
use arrow::util::display::{ArrayFormatter, FormatOptions};

use crate::value::Value;

/// The most rows a table shows, and the most rows and columns of a tensor.
const MAX_ROWS: usize = 50;
const MAX_TENSOR: usize = 20;

fn escape(text: &str) -> String {
    text.replace('&', "&amp;")
        .replace('<', "&lt;")
        .replace('>', "&gt;")
}

/// `v` as an HTML fragment, or `None` when plain text shows it as well as anything.
pub fn html(v: &Value) -> Option<String> {
    match v {
        Value::Table(t) => {
            let batch = &t.batch;
            let schema = batch.schema();
            let mut out = String::from("<table class=\"telvax-table\"><thead><tr>");
            for f in schema.fields() {
                out.push_str(&format!(
                    "<th title=\"{}\">{}</th>",
                    escape(&f.data_type().to_string()),
                    escape(f.name())
                ));
            }
            out.push_str("</tr></thead><tbody>");
            let options = FormatOptions::default().with_null("null");
            let formatters: Vec<_> = batch
                .columns()
                .iter()
                .map(|c| ArrayFormatter::try_new(c.as_ref(), &options).ok())
                .collect();
            for row in 0..batch.num_rows().min(MAX_ROWS) {
                out.push_str("<tr>");
                for f in &formatters {
                    let cell = f
                        .as_ref()
                        .map_or(String::new(), |f| f.value(row).to_string());
                    out.push_str(&format!("<td>{}</td>", escape(&cell)));
                }
                out.push_str("</tr>");
            }
            out.push_str("</tbody></table>");
            let more = batch.num_rows().saturating_sub(MAX_ROWS);
            out.push_str(&format!(
                "<p>{} row{} × {} column{}{}</p>",
                batch.num_rows(),
                if batch.num_rows() == 1 { "" } else { "s" },
                batch.num_columns(),
                if batch.num_columns() == 1 { "" } else { "s" },
                if more > 0 {
                    format!(" ({more} more not shown)")
                } else {
                    String::new()
                }
            ));
            Some(out)
        }
        Value::Tensor(t) => Some(tensor(&t.shape, &t.array)),
        _ => None,
    }
}

fn tensor(shape: &[usize], array: &ArrayRef) -> String {
    let dims: Vec<String> = shape.iter().map(|d| d.to_string()).collect();
    let title = format!("Tensor {} [{}]", array.data_type(), dims.join(", "));
    let options = FormatOptions::default();
    let Ok(f) = ArrayFormatter::try_new(array.as_ref(), &options) else {
        return format!("<p>{}</p>", escape(&title));
    };
    let cell = |i: usize| escape(&f.value(i).to_string());
    let mut out = format!("<p><b>{}</b></p>", escape(&title));
    // A grid for one or two dimensions; anything else is listed flat.
    let (rows, cols) = match shape {
        [n] => (1, *n),
        [r, c] => (*r, *c),
        _ => (1, array.len()),
    };
    out.push_str("<table class=\"telvax-tensor\"><tbody>");
    for r in 0..rows.min(MAX_TENSOR) {
        out.push_str("<tr>");
        for c in 0..cols.min(MAX_TENSOR) {
            out.push_str(&format!("<td>{}</td>", cell(r * cols + c)));
        }
        if cols > MAX_TENSOR {
            out.push_str("<td>…</td>");
        }
        out.push_str("</tr>");
    }
    if rows > MAX_TENSOR {
        out.push_str("<tr><td>…</td></tr>");
    }
    out.push_str("</tbody></table>");
    out
}

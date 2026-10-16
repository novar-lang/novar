use std::sync::Arc;

use arrow::array::{ArrayRef, Float32Array, Int64Array, StringArray};
use arrow::record_batch::RecordBatch;
use telvax_runtime::Value;
use telvax_runtime::rich::html;

fn table(rows: usize) -> Value {
    let ids: ArrayRef = Arc::new(Int64Array::from_iter_values(0..rows as i64));
    let names: ArrayRef = Arc::new(StringArray::from_iter_values(
        (0..rows).map(|i| format!("<row {i}> & co")),
    ));
    let scores: ArrayRef = Arc::new(Float32Array::from_iter_values(
        (0..rows).map(|i| i as f32 / 2.0),
    ));
    let batch =
        RecordBatch::try_from_iter([("id", ids), ("name", names), ("score", scores)]).unwrap();
    Value::Table(Arc::new(telvax_runtime::TableVal {
        schema: "Row".into(),
        batch,
    }))
}

#[test]
fn a_table_is_an_html_table_with_escaped_cells() {
    let page = html(&table(3)).expect("a table");
    assert!(page.starts_with("<table class=\"telvax-table\">"), "{page}");
    assert_eq!(page.matches("<th ").count(), 3);
    assert_eq!(page.matches("<tr>").count(), 4, "a header and three rows");
    assert!(page.contains("&lt;row 1&gt; &amp; co"), "{page}");
    assert!(page.contains("<td>0.5</td>"), "{page}");
    assert!(page.contains("3 rows × 3 columns"), "{page}");
}

#[test]
fn a_long_table_shows_the_first_rows_and_says_how_many_are_left() {
    let page = html(&table(120)).expect("a table");
    assert_eq!(page.matches("<tr>").count(), 51);
    assert!(
        page.contains("120 rows") && page.contains("70 more not shown"),
        "{page}"
    );
}

#[test]
fn values_that_text_shows_well_have_no_html() {
    assert!(html(&Value::Int(3)).is_none());
    assert!(html(&Value::str("text")).is_none());
}

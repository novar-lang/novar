//! Measures ingestion and stream-processing throughput of the runtime.
//!
//!     cargo run --release -p telvax_bench --bin ingest [-- --rows N]

use std::sync::Arc;
use std::time::Instant;

use telvax_runtime::{Interp, TensorVal, Value, embed_text};

fn load(src: &str) -> Interp {
    let (ast, perrs) = telvax_parser::parse(src);
    assert!(perrs.is_empty(), "{perrs:?}");
    let (hir, diags) = telvax_hir::lower(&ast.unwrap());
    assert!(
        diags.is_empty(),
        "{:?}",
        diags.iter().map(|d| &d.message).collect::<Vec<_>>()
    );
    Interp::new(hir).unwrap()
}

fn rate(label: &str, n: f64, unit: &str, secs: f64, extra: &str) {
    println!(
        "{label:<34} {:>12.0} {unit}/s   ({:.3} s){extra}",
        n / secs,
        secs
    );
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let rows: usize = args
        .windows(2)
        .find(|w| w[0] == "--rows")
        .map_or(1_000_000, |w| w[1].parse().unwrap());
    let dir = std::env::temp_dir().join(format!("telvax-ingest-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();

    let src = r#"
        struct Event { id: i64, user: i64, amount: f32, label: String }
        fn read_csv(p: String) -> Table<Event> { std::table::read_csv(p) }
        fn read_pq(p: String) -> Table<Event> { std::table::read_parquet(p) }
        fn write_pq(t: Table<Event>, p: String) { std::table::write_parquet(t, p); }
        fn rows(t: Table<Event>) -> u64 { t.len() }
        fn mean_amount(t: Table<Event>) -> f32 { std::tensor::mean(t.amount) }
        fn big(t: Table<Event>) -> Table<Event> { t |> std::table::sql("SELECT * FROM t WHERE amount > 90.0") }
        fn totals(t: Table<Event>) -> Table<Event> {
            t |> std::table::sql("SELECT min(id) AS id, user, sum(amount) AS amount, 'sum' AS label FROM t GROUP BY user")
        }
        fn matmul(a: Tensor<f32, [256, 256]>, b: Tensor<f32, [256, 256]>) -> Tensor<f32, [256, 256]> { std::tensor::matmul(a, b) }
    "#;
    let i = load(src);

    println!("ingestion and stream processing ({rows} rows, release build)\n");

    // CSV file with `rows` events.
    let csv = dir.join("events.csv");
    let mut text = String::with_capacity(rows * 32);
    text.push_str("id,user,amount,label\n");
    for n in 0..rows {
        text.push_str(&format!(
            "{n},{},{:.2},{}\n",
            n % 5000,
            (n % 10000) as f32 / 100.0,
            if n % 2 == 0 { "even" } else { "odd" }
        ));
    }
    std::fs::write(&csv, &text).unwrap();
    let mb = text.len() as f64 / 1e6;

    let t = Instant::now();
    let table = i
        .call("read_csv", vec![Value::str(csv.to_string_lossy().as_ref())])
        .unwrap();
    let secs = t.elapsed().as_secs_f64();
    rate(
        "CSV -> Arrow table",
        rows as f64,
        "rows",
        secs,
        &format!("  {:.0} MB/s", mb / secs),
    );

    let pq = dir.join("events.parquet");
    let t = Instant::now();
    i.call(
        "write_pq",
        vec![table.clone(), Value::str(pq.to_string_lossy().as_ref())],
    )
    .unwrap();
    rate(
        "Arrow table -> Parquet",
        rows as f64,
        "rows",
        t.elapsed().as_secs_f64(),
        "",
    );

    let t = Instant::now();
    let table2 = i
        .call("read_pq", vec![Value::str(pq.to_string_lossy().as_ref())])
        .unwrap();
    rate(
        "Parquet -> Arrow table",
        rows as f64,
        "rows",
        t.elapsed().as_secs_f64(),
        "",
    );
    assert!(matches!(i.call("rows", vec![table2]).unwrap(), Value::Int(n) if n as usize == rows));

    let t = Instant::now();
    let mean = i.call("mean_amount", vec![table.clone()]).unwrap();
    rate(
        "column mean (zero-copy column)",
        rows as f64,
        "rows",
        t.elapsed().as_secs_f64(),
        &format!("  mean={mean}"),
    );

    let t = Instant::now();
    let filtered = i.call("big", vec![table.clone()]).unwrap();
    rate(
        "SQL filter (DataFusion)",
        rows as f64,
        "rows",
        t.elapsed().as_secs_f64(),
        "",
    );
    let _ = filtered;

    let t = Instant::now();
    i.call("totals", vec![table.clone()]).unwrap();
    rate(
        "SQL group-by (DataFusion)",
        rows as f64,
        "rows",
        t.elapsed().as_secs_f64(),
        "",
    );

    // Vector ingestion: embed + upsert into the in-memory store.
    let db = i.db_for("memory://bench").unwrap();
    let n = 20_000;
    let t = Instant::now();
    for k in 0..n {
        let text = format!(
            "document {k} about tensors pipelines and schemas {}",
            k % 97
        );
        db.insert("kb", &format!("d{k}"), &text, embed_text(&text))
            .unwrap();
    }
    rate(
        "vector ingestion (768-d, offline)",
        n as f64,
        "docs",
        t.elapsed().as_secs_f64(),
        "",
    );

    let a = Value::Tensor(Arc::new(TensorVal::from_f32(
        vec![256, 256],
        (0..65536).map(|x| (x % 13) as f32).collect(),
    )));
    let t = Instant::now();
    let reps = 20;
    for _ in 0..reps {
        i.call("matmul", vec![a.clone(), a.clone()]).unwrap();
    }
    let secs = t.elapsed().as_secs_f64();
    let flops = 2.0 * 256f64.powi(3) * reps as f64;
    println!(
        "{:<34} {:>12.2} GFLOP/s  ({:.3} s)",
        "matmul 256x256 f32 (interpreter)",
        flops / secs / 1e9,
        secs
    );

    let _ = std::fs::remove_dir_all(&dir);
}

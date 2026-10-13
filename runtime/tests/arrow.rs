use std::sync::Arc;

use novar_runtime::{Interp, TensorVal, Value};
use serde_json::json;

fn load(src: &str) -> Interp {
    let (ast, perrs) = novar_parser::parse(src);
    assert!(perrs.is_empty(), "parse: {perrs:?}");
    let (hir, diags) = novar_hir::lower(&ast.unwrap());
    assert!(
        diags.is_empty(),
        "check: {:?}",
        diags.iter().map(|d| &d.message).collect::<Vec<_>>()
    );
    Interp::new(hir).expect("globals should initialise")
}

fn run(src: &str, f: &str, args: Vec<Value>) -> Value {
    load(src)
        .call(f, args)
        .unwrap_or_else(|e| panic!("runtime error: {e}"))
}

fn float(v: Value) -> f64 {
    match v {
        Value::Float(x) => x,
        other => panic!("expected float, got {other:?}"),
    }
}

fn int(v: Value) -> i64 {
    match v {
        Value::Int(i) => i,
        other => panic!("expected int, got {other:?}"),
    }
}

fn tensor(v: Value) -> Arc<TensorVal> {
    match v {
        Value::Tensor(t) => t,
        other => panic!("expected tensor, got {other:?}"),
    }
}

fn f32s(shape: &[usize], data: &[f32]) -> Value {
    Value::Tensor(Arc::new(TensorVal::from_f32(shape.to_vec(), data.to_vec())))
}

fn scratch(name: &str) -> String {
    let dir = std::env::temp_dir().join(format!("novar-arrow-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    dir.join(name).to_string_lossy().into_owned()
}

#[test]
fn elementwise_and_reductions() {
    let src = "
        fn add(a: Tensor<f32, [3]>, b: Tensor<f32, [3]>) -> Tensor<f32, [3]> { std::tensor::add(a, b) }
        fn chain(a: Tensor<f32, [4]>) -> Tensor<f32, [4]> {
            a |> std::tensor::scale(2.0) |> std::tensor::relu() |> std::tensor::scale(0.5)
        }
        fn total(a: Tensor<f32, [4]>) -> f32 { std::tensor::sum(a) }
        fn avg(a: Tensor<f32, [4]>) -> f32 { std::tensor::mean(a) }
        fn best(a: Tensor<f32, [4]>) -> u64 { std::tensor::argmax(a) }
        fn dot(a: Tensor<f32, [3]>, b: Tensor<f32, [3]>) -> f32 { std::tensor::dot(a, b) }
    ";
    let sum = tensor(run(
        src,
        "add",
        vec![
            f32s(&[3], &[1.0, 2.0, 3.0]),
            f32s(&[3], &[10.0, 20.0, 30.0]),
        ],
    ));
    assert_eq!(sum.to_f32_vec(), [11.0, 22.0, 33.0]);
    let relu = tensor(run(src, "chain", vec![f32s(&[4], &[-1.0, 2.0, -3.0, 4.0])]));
    assert_eq!(relu.to_f32_vec(), [0.0, 2.0, 0.0, 4.0]);
    let v = f32s(&[4], &[1.0, 5.0, 2.0, 0.0]);
    assert_eq!(float(run(src, "total", vec![v.clone()])), 8.0);
    assert_eq!(float(run(src, "avg", vec![v.clone()])), 2.0);
    assert_eq!(int(run(src, "best", vec![v])), 1);
    assert_eq!(
        float(run(
            src,
            "dot",
            vec![f32s(&[3], &[1.0, 2.0, 3.0]), f32s(&[3], &[4.0, 5.0, 6.0])]
        )),
        32.0
    );
}

#[test]
fn matmul_transpose_and_softmax() {
    let src = "
        fn mm(a: Tensor<f32, [2, 3]>, b: Tensor<f32, [3, 2]>) -> Tensor<f32, [2, 2]> { std::tensor::matmul(a, b) }
        fn mv(a: Tensor<f32, [2, 3]>, v: Tensor<f32, [3]>) -> Tensor<f32, [2]> { std::tensor::matmul(a, v) }
        fn tr(a: Tensor<f32, [2, 3]>) -> Tensor<f32, [3, 2]> { std::tensor::transpose(a) }
        fn sm(a: Tensor<f32, [2, 3]>) -> Tensor<f32, [2, 3]> { std::tensor::softmax(a) }
    ";
    let a = f32s(&[2, 3], &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
    let b = f32s(&[3, 2], &[7.0, 8.0, 9.0, 10.0, 11.0, 12.0]);
    let c = tensor(run(src, "mm", vec![a.clone(), b]));
    assert_eq!(
        (c.shape.clone(), c.to_f32_vec()),
        (vec![2, 2], vec![58.0, 64.0, 139.0, 154.0])
    );
    let v = tensor(run(
        src,
        "mv",
        vec![a.clone(), f32s(&[3], &[1.0, 0.0, -1.0])],
    ));
    assert_eq!(
        (v.shape.clone(), v.to_f32_vec()),
        (vec![2], vec![-2.0, -2.0])
    );
    let t = tensor(run(src, "tr", vec![a.clone()]));
    assert_eq!(
        (t.shape.clone(), t.to_f32_vec()),
        (vec![3, 2], vec![1.0, 4.0, 2.0, 5.0, 3.0, 6.0])
    );
    let s = tensor(run(src, "sm", vec![a]));
    let rows = s.to_f32_vec();
    assert!(
        (rows[..3].iter().sum::<f32>() - 1.0).abs() < 1e-6
            && (rows[3..].iter().sum::<f32>() - 1.0).abs() < 1e-6
    );
    assert!(rows[2] > rows[1] && rows[1] > rows[0]);
}

#[test]
fn constructors_use_the_declared_type() {
    let src = "
        fn z() -> Tensor<f32, [2, 2]> { std::tensor::zeros() }
        fn fv(v: Vec<f32>) -> Tensor<f32, [2, 2]> { std::tensor::from_vec(v) }
        fn rs(a: Tensor<f32, [2, 3]>) -> Tensor<f32, [3, 2]> { std::tensor::reshape(a) }
        fn back(a: Tensor<f32, [3]>) -> Vec<f32> { std::tensor::to_vec(a) }
        fn ints() -> Tensor<i64, [3]> { std::tensor::from_vec([1, 2, 3]) }
    ";
    assert_eq!(tensor(run(src, "z", vec![])).to_f32_vec(), [0.0; 4]);
    let v = Value::vec(vec![
        Value::Float(1.0),
        Value::Float(2.0),
        Value::Float(3.0),
        Value::Float(4.0),
    ]);
    let t = tensor(run(src, "fv", vec![v]));
    assert_eq!(
        (t.shape.clone(), t.to_f32_vec()),
        (vec![2, 2], vec![1.0, 2.0, 3.0, 4.0])
    );
    let i = load(src);
    let r = tensor(
        i.call("rs", vec![f32s(&[2, 3], &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0])])
            .unwrap(),
    );
    assert_eq!(r.shape, [3, 2]);
    let wrong = Value::vec(vec![Value::Float(1.0)]);
    assert!(
        i.call("fv", vec![wrong])
            .unwrap_err()
            .message
            .contains("needs 4 elements")
    );
    assert!(
        matches!(i.call("back", vec![f32s(&[3], &[1.0, 2.0, 3.0])]).unwrap(), Value::Vec(v) if v.len() == 3)
    );
    assert_eq!(
        tensor(i.call("ints", vec![]).unwrap()).to_f64_vec(),
        [1.0, 2.0, 3.0]
    );
}

#[test]
fn runtime_shape_checks_catch_dynamic_mismatches() {
    let src = "fn add(a: Tensor<f32, [N]>, b: Tensor<f32, [N]>) -> Tensor<f32, [N]> { std::tensor::add(a, b) }";
    let i = load(src);
    let e = i
        .call(
            "add",
            vec![f32s(&[2], &[1.0, 2.0]), f32s(&[3], &[1.0, 2.0, 3.0])],
        )
        .unwrap_err();
    assert!(e.message.contains("shapes differ"), "{e}");
}

const TABLE_SRC: &str = r#"
    struct Row { id: i64, score: f32, name: String, ok: bool }
    fn load(path: String) -> Table<Row> { std::table::read_csv(path) }
    fn scores(t: Table<Row>) -> Tensor<f32, [N]> { t.score }
    fn names(t: Table<Row>) -> Vec<String> { t.name }
    fn count(t: Table<Row>) -> u64 { t.len() }
    fn second(t: Table<Row>) -> Row { t[1] }
    fn total(t: Table<Row>) -> f32 { let mut s = 0.0; for r in t { s += r.score; } s }
    fn top(t: Table<Row>) -> Table<Row> { t |> std::table::sql("SELECT * FROM t WHERE score > 1.5 ORDER BY score DESC") }
    fn good(t: Table<Row>) -> Table<Row> { t |> std::table::filter(|r| r.ok) }
    fn avg(t: Table<Row>) -> f32 { t.score |> std::tensor::mean() }
    fn save_csv(t: Table<Row>, path: String) { std::table::write_csv(t, path); }
    fn save_pq(t: Table<Row>, path: String) { std::table::write_parquet(t, path); }
    fn load_pq(path: String) -> Table<Row> { std::table::read_parquet(path) }
    fn rebuild(t: Table<Row>) -> Table<Row> { std::table::from_rows(std::table::to_rows(t)) }
"#;

fn csv() -> String {
    let path = scratch("rows.csv");
    std::fs::write(
        &path,
        "id,score,name,ok\n1,1.0,ann,true\n2,2.5,bob,false\n3,3.5,cy,true\n",
    )
    .unwrap();
    path
}

#[test]
fn csv_ingestion_columns_and_rows() {
    let i = load(TABLE_SRC);
    let t = i.call("load", vec![Value::str(csv())]).unwrap();
    assert_eq!(int(i.call("count", vec![t.clone()]).unwrap()), 3);
    let scores = tensor(i.call("scores", vec![t.clone()]).unwrap());
    assert_eq!(scores.to_f32_vec(), [1.0, 2.5, 3.5]);
    let Value::Vec(names) = i.call("names", vec![t.clone()]).unwrap() else {
        panic!()
    };
    assert_eq!(
        names
            .snapshot()
            .iter()
            .map(|v| v.to_string())
            .collect::<Vec<_>>(),
        ["ann", "bob", "cy"]
    );
    assert_eq!(
        i.call("second", vec![t.clone()]).unwrap().to_string(),
        "Row { id: 2, score: 2.5, name: bob, ok: false }"
    );
    assert_eq!(float(i.call("total", vec![t.clone()]).unwrap()), 7.0);
    assert_eq!(
        float(i.call("avg", vec![t]).unwrap()),
        f64::from((7.0_f64 / 3.0) as f32)
    );
}

#[test]
fn numeric_columns_share_the_arrow_buffer() {
    let i = load(TABLE_SRC);
    let t = i.call("load", vec![Value::str(csv())]).unwrap();
    let Value::Table(table) = &t else { panic!() };
    let col = tensor(i.call("scores", vec![t.clone()]).unwrap());
    let idx = table.batch.schema().index_of("score").unwrap();
    assert_eq!(
        col.f32_slice().unwrap().as_ptr(),
        table.batch.column(idx).to_data().buffers()[0].as_ptr() as *const f32,
        "reading a column must not copy it"
    );
}

#[test]
fn sql_filter_and_round_trips() {
    let i = load(TABLE_SRC);
    let t = i.call("load", vec![Value::str(csv())]).unwrap();
    let top = i.call("top", vec![t.clone()]).unwrap();
    assert_eq!(
        top.to_json(),
        json!([
            {"id": 3, "score": 3.5, "name": "cy", "ok": true},
            {"id": 2, "score": 2.5, "name": "bob", "ok": false},
        ])
    );
    let good = i.call("good", vec![t.clone()]).unwrap();
    assert_eq!(int(i.call("count", vec![good]).unwrap()), 2);

    let out = scratch("out.csv");
    i.call("save_csv", vec![t.clone(), Value::str(out.as_str())])
        .unwrap();
    let again = i.call("load", vec![Value::str(out)]).unwrap();
    assert!(again.equals(&t), "CSV round trip");

    let pq = scratch("out.parquet");
    i.call("save_pq", vec![t.clone(), Value::str(pq.as_str())])
        .unwrap();
    let back = i.call("load_pq", vec![Value::str(pq)]).unwrap();
    assert!(back.equals(&t), "Parquet round trip");

    let rebuilt = i.call("rebuild", vec![t.clone()]).unwrap();
    assert!(rebuilt.equals(&t), "rows round trip");
}

#[test]
fn tables_report_bad_input() {
    let i = load(TABLE_SRC);
    let missing = i
        .call("load", vec![Value::str("/nonexistent/x.csv")])
        .unwrap_err();
    assert!(missing.message.contains("cannot open"), "{missing}");
    let bad = scratch("bad.csv");
    std::fs::write(&bad, "id,score,name,ok\nx,1.0,ann,true\n").unwrap();
    assert!(i.call("load", vec![Value::str(bad)]).is_err());
    let t = i.call("load", vec![Value::str(csv())]).unwrap();
    let sql = i.call("top", vec![t]).unwrap();
    assert!(matches!(sql, Value::Table(_)));
}

#[test]
fn tables_and_tensors_travel_as_json() {
    let src = r#"
        struct Row { id: i64, score: f32 }
        struct In { rows: Table<Row>, w: Tensor<f32, [2]> }
        struct Out { n: u64, weighted: Tensor<f32, [N]>, rows: Table<Row> }
        @endpoint(path = "/score", method = "POST")
        fn score(p: In) -> Result<Out, String> {
            Ok(Out { n: p.rows.len(), weighted: p.rows.score |> std::tensor::scale(2.0), rows: p.rows })
        }
    "#;
    let i = load(src);
    let body = json!({"rows": [{"id": 1, "score": 0.5}, {"id": 2, "score": 1.5}], "w": [1.0, 2.0]});
    let out = i.call_endpoint("score", &body).unwrap();
    assert_eq!(out["n"], 2);
    assert_eq!(out["weighted"], json!([1.0, 3.0]));
    assert_eq!(out["rows"], body["rows"]);
}

#[test]
fn int8_quantized_matmul_stays_close_to_float() {
    let src = "fn mm(a: Tensor<f32, [4, 8]>, b: Tensor<f32, [8, 3]>) -> Tensor<f32, [4, 3]> { std::tensor::matmul(a, b) }";
    let a: Vec<f32> = (0..32)
        .map(|i| ((i * 37 % 19) as f32 - 9.0) / 4.0)
        .collect();
    let b: Vec<f32> = (0..24)
        .map(|i| ((i * 53 % 23) as f32 - 11.0) / 7.0)
        .collect();
    let (ta, tb) = (f32s(&[4, 8], &a), f32s(&[8, 3], &b));

    let exact = load(src);
    let fp = tensor(exact.call("mm", vec![ta.clone(), tb.clone()]).unwrap()).to_f32_vec();

    let quant = load(src);
    quant.set_quantization(novar_hir::Quantization::Int8);
    let q = tensor(quant.call("mm", vec![ta, tb]).unwrap()).to_f32_vec();

    assert_ne!(fp, q, "quantization must actually change the arithmetic");
    let scale = fp.iter().fold(0.0f32, |m, v| m.max(v.abs()));
    let worst = fp
        .iter()
        .zip(&q)
        .map(|(x, y)| (x - y).abs())
        .fold(0.0f32, f32::max);
    assert!(
        worst / scale < 0.03,
        "relative error {} too large",
        worst / scale
    );
}

#[test]
fn int8_quantization_handles_zeros_and_exact_cases() {
    let src = "fn mm(a: Tensor<f32, [2, 2]>, b: Tensor<f32, [2, 2]>) -> Tensor<f32, [2, 2]> { std::tensor::matmul(a, b) }";
    let i = load(src);
    i.set_quantization(novar_hir::Quantization::Int8);
    let zero = f32s(&[2, 2], &[0.0; 4]);
    assert_eq!(
        tensor(i.call("mm", vec![zero.clone(), zero]).unwrap()).to_f32_vec(),
        [0.0; 4]
    );
    // Values on the int8 grid multiply exactly: scales are max/127 and entries are multiples.
    let a = f32s(&[2, 2], &[127.0, -127.0, 0.0, 127.0]);
    let id = f32s(&[2, 2], &[1.0, 0.0, 0.0, 1.0]);
    assert_eq!(
        tensor(i.call("mm", vec![a, id]).unwrap()).to_f32_vec(),
        [127.0, -127.0, 0.0, 127.0]
    );
}

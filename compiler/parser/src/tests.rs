use super::*;

fn ok_expr(src: &str) -> ExprKind {
    let (e, errs) = parse_expr(src);
    assert!(errs.is_empty(), "{src}: {errs:?}");
    e.unwrap().node
}

fn ok_program(src: &str) -> Program {
    let (p, errs) = parse(src);
    assert!(errs.is_empty(), "{errs:#?}\n{src}");
    p.unwrap()
}

fn only_fn(src: &str) -> Fn {
    match ok_program(src).items.into_iter().next() {
        Some(Item::Fn(f)) => f,
        other => panic!("expected a function, got {other:?}"),
    }
}

fn show(e: &Expr) -> String {
    match &e.node {
        ExprKind::Lit(Lit::Int(n)) => n.to_string(),
        ExprKind::Path(p) => p
            .iter()
            .map(|i| i.node.as_str())
            .collect::<Vec<_>>()
            .join("."),
        ExprKind::Binary { op, lhs, rhs } => format!("({:?} {} {})", op, show(lhs), show(rhs)),
        ExprKind::Pipe { lhs, rhs } => format!("(|> {} {})", show(lhs), show(rhs)),
        ExprKind::Call { callee, args } => {
            let a: Vec<_> = args.iter().map(show).collect();
            format!("{}({})", show(callee), a.join(", "))
        }
        ExprKind::Field { base, name } => format!("{}.{}", show(base), name.node),
        ExprKind::Unary { op, expr } => format!("({:?} {})", op, show(expr)),
        other => format!("{other:?}"),
    }
}

fn sexp(src: &str) -> String {
    let (e, errs) = parse_expr(src);
    assert!(errs.is_empty(), "{src}: {errs:?}");
    show(&e.unwrap())
}

#[test]
fn precedence() {
    assert_eq!(sexp("1 + 2 * 3"), "(Add 1 (Mul 2 3))");
    assert_eq!(sexp("a or b and c == d"), "(Or a (And b (Eq c d)))");
    assert_eq!(sexp("-a.b"), "(Neg a.b)");
    assert_eq!(sexp("(1 + 2) * 3"), "(Mul (Add 1 2) 3)");
    assert_eq!(sexp("not a and b"), "(And (Not a) b)");
    assert_eq!(sexp("not a == b"), "(Not (Eq a b))");
}

#[test]
fn pipe_is_lowest_and_left_associative() {
    assert_eq!(sexp("a + 1 |> f() |> g(2)"), "(|> (|> (Add a 1) f()) g(2))");
}

#[test]
fn calls_fields_and_paths() {
    assert_eq!(sexp("a.b.c(1, 2).d"), "a.b.c(1, 2).d");
    // `std`, capitalized names and imported names start a path; other names are fields.
    assert!(
        matches!(ok_expr("std.text.trim(x,)"), ExprKind::Call { callee, .. }
        if matches!(&callee.node, ExprKind::Path(p) if p.len() == 3))
    );
    assert!(
        matches!(ok_expr("Shape.Circle(2.0)"), ExprKind::Call { callee, .. }
        if matches!(&callee.node, ExprKind::Path(p) if p.len() == 2))
    );
    assert!(
        matches!(ok_expr("counts.get(w)"), ExprKind::Call { callee, .. }
        if matches!(&callee.node, ExprKind::Field { .. }))
    );
}

#[test]
fn imported_names_start_paths() {
    let p = ok_program("import util\n\nfn f() -> Int:\n    return util.clamp(3)\n");
    let Item::Fn(f) = &p.items[1] else { panic!() };
    let Some(t) = f.body.tail.as_deref() else {
        panic!()
    };
    assert!(matches!(&t.node, ExprKind::Call { callee, .. }
        if matches!(&callee.node, ExprKind::Path(p) if p.len() == 2)));
}

#[test]
fn arrow_functions() {
    assert!(matches!(ok_expr("map(d => d.text)"), ExprKind::Call { .. }));
    let ExprKind::Call { args, .. } = ok_expr("f((a, b) => a + b, () => 1, x => x)") else {
        panic!()
    };
    assert!(matches!(&args[0].node, ExprKind::Closure { params, .. } if params.len() == 2));
    assert!(matches!(&args[1].node, ExprKind::Closure { params, .. } if params.is_empty()));
    assert!(matches!(&args[2].node, ExprKind::Closure { params, .. } if params.len() == 1));
    // A parenthesized tuple is still a tuple.
    assert!(matches!(ok_expr("(a, b)"), ExprKind::Tuple(_)));
    // Double parentheses take one tuple argument apart.
    let ExprKind::Closure { params, body, .. } = ok_expr("((i, x)) => i + x") else {
        panic!()
    };
    assert_eq!(params.len(), 1);
    assert!(matches!(body.node, ExprKind::Block(_)));
}

#[test]
fn construction_names_every_field() {
    let ExprKind::StructLit { path, fields } = ok_expr("Foo(a: 1, b: 2,)") else {
        panic!()
    };
    assert_eq!(path.len(), 1);
    assert_eq!(fields.len(), 2);
    assert!(matches!(ok_expr("Foo(1, 2)"), ExprKind::Call { .. }));
    // A call to a function keeps the names for the checker.
    let ExprKind::Call { args, .. } = ok_expr("search(v, limit: 3)") else {
        panic!()
    };
    assert!(matches!(&args[1].node, ExprKind::Named { name, .. } if name.node == "limit"));
}

#[test]
fn membership_and_conditional_expressions() {
    assert_eq!(sexp("x in xs"), "xs.contains(x)");
    assert_eq!(sexp("x not in xs"), "(Not xs.contains(x))");
    assert!(matches!(ok_expr("1 if c else 2"), ExprKind::If { .. }));
}

#[test]
fn fstring_interpolation() {
    let ExprKind::FString(parts) = ok_expr(r#"f"hi {name}!\n{a + 1}""#) else {
        panic!()
    };
    assert_eq!(parts.len(), 4);
    assert!(matches!(&parts[0], FPart::Text(t) if t == "hi "));
    assert!(matches!(&parts[1], FPart::Expr(_)));
    assert!(matches!(&parts[2], FPart::Text(t) if t == "!\n"));
    let FPart::Expr(e) = &parts[3] else { panic!() };
    assert_eq!(show(e), "(Add a 1)");
    // Spans of interpolated expressions point into the original source.
    assert_eq!(e.span, 15..20);
}

#[test]
fn imports() {
    let p = ok_program("from std.http import Request, Response\nimport a.B\n");
    let Item::Import(i) = &p.items[0] else {
        panic!()
    };
    assert_eq!(i.path.len(), 2);
    assert_eq!(i.group.len(), 2);
    let Item::Import(i) = &p.items[1] else {
        panic!()
    };
    assert_eq!((i.path.len(), i.group.len()), (2, 0));
}

#[test]
fn types_have_capitalized_names_and_inline_shapes() {
    let f = only_fn(
        "fn f(x: Tensor[f32, 768], y: List[Tensor[i8, N, 4]], z: Int?, \
         w: Map[Text, (Int, Float)]) -> List: ...\n",
    );
    assert_eq!(f.params[0].ty.path[0].node, "Tensor");
    assert!(matches!(&f.params[0].ty.args[0], TypeArg::Type(t) if t.path[0].node == "f32"));
    assert!(matches!(&f.params[0].ty.args[1], TypeArg::Shape(d) if d == &[Dim::Lit(768)]));
    assert_eq!(f.params[1].ty.path[0].node, "Vec");
    let TypeArg::Type(inner) = &f.params[1].ty.args[0] else {
        panic!()
    };
    assert!(matches!(&inner.args[1], TypeArg::Shape(d)
        if matches!(d.as_slice(), [Dim::Named(n), Dim::Lit(4)] if n.node == "N")));
    assert_eq!(f.params[2].ty.path[0].node, "Option");
    assert_eq!(f.params[3].ty.path[0].node, "Map");
    assert_eq!(f.ret.as_ref().unwrap().path[0].node, "Vec");
}

#[test]
fn the_last_expression_is_the_value() {
    let f = only_fn("fn f(x: Int) -> Int:\n    let y = x + 1\n    y * 2\n");
    assert_eq!(f.body.stmts.len(), 1);
    assert!(f.body.tail.is_some());
    // `return` at the end means the same.
    let f = only_fn("fn f(x: Int) -> Int:\n    return x\n");
    assert!(f.body.stmts.is_empty() && f.body.tail.is_some());
}

#[test]
fn a_function_without_a_return_type_keeps_its_statements() {
    let f = only_fn("fn f(x: Int):\n    print(x)\n");
    assert_eq!(f.body.stmts.len(), 1);
    assert!(f.body.tail.is_none());
}

#[test]
fn the_last_expression_of_every_branch_is_the_value() {
    let f = only_fn(
        "fn f(n: Int) -> Text:\n    if n < 0:\n        \"neg\"\n    elif n == 0:\n        \
         return \"zero\"\n    else:\n        \"pos\"\n",
    );
    assert!(f.body.stmts.is_empty());
    let Some(Spanned {
        node: ExprKind::If {
            then, otherwise, ..
        },
        ..
    }) = f.body.tail.as_deref()
    else {
        panic!("{:?}", f.body)
    };
    assert!(then.tail.is_some());
    assert!(matches!(
        otherwise.as_deref(),
        Some(Spanned {
            node: ExprKind::If { .. },
            ..
        })
    ));
}

#[test]
fn an_early_return_stays_a_statement() {
    let f = only_fn("fn f(n: Int) -> Int:\n    if n < 0:\n        return 0\n    n\n");
    assert_eq!(f.body.stmts.len(), 1);
    assert!(f.body.tail.is_some());
}

#[test]
fn let_is_fixed_and_var_can_change() {
    let f = only_fn("fn f():\n    let a = 1\n    var b: Int = 2\n    b += a\n");
    assert!(matches!(&f.body.stmts[0], Stmt::Let(l) if !l.mutable && l.name.node == "a"));
    assert!(matches!(&f.body.stmts[1], Stmt::Let(l) if l.mutable && l.ty.is_some()));
    assert!(matches!(
        &f.body.stmts[2],
        Stmt::Expr(Spanned {
            node: ExprKind::Assign { .. },
            ..
        })
    ));
}

#[test]
fn a_plain_assignment_does_not_declare() {
    let f = only_fn("fn f():\n    x = 1\n");
    assert!(matches!(
        &f.body.stmts[0],
        Stmt::Expr(Spanned {
            node: ExprKind::Assign { .. },
            ..
        })
    ));
}

#[test]
fn destructuring() {
    let f = only_fn("fn f():\n    let a, b = g()\n    var c, d = g()\n");
    // A temporary and one binding per name, twice.
    assert_eq!(f.body.stmts.len(), 6);
    assert!(matches!(&f.body.stmts[1], Stmt::Let(l) if l.name.node == "a" && !l.mutable));
    assert!(matches!(&f.body.stmts[4], Stmt::Let(l) if l.name.node == "c" && l.mutable));
}

#[test]
fn loops_and_ranges() {
    let f = only_fn(
        "fn f(v: List[Int], n: Int) -> Int:\n    var total = 0\n    var i = 0\n    while i < n:\n        \
         total += v[i] * 2\n        i = i + 1\n    for x in v:\n        total -= x\n    \
         for k in range(n):\n        total += k\n    for k in range(1, 4):\n        total *= k\n    \
         total\n",
    );
    let kinds: Vec<_> = f
        .body
        .stmts
        .iter()
        .map(|s| match s {
            Stmt::Let(_) => "let",
            Stmt::Expr(e) => match &e.node {
                ExprKind::While { .. } => "while",
                ExprKind::For {
                    iter: ForIter::Range { start, .. },
                    ..
                } => {
                    if matches!(start.node, ExprKind::Lit(Lit::Int(0))) {
                        "range(n)"
                    } else {
                        "range(a, b)"
                    }
                }
                ExprKind::For { .. } => "for-in",
                _ => "expr",
            },
        })
        .collect();
    assert_eq!(
        kinds,
        ["let", "let", "while", "for-in", "range(n)", "range(a, b)"]
    );
    assert!(f.body.tail.is_some());
}

#[test]
fn for_with_two_names() {
    let f = only_fn("fn f(xs: List[Int]):\n    for i, x in xs.enumerate():\n        print(i)\n");
    let Stmt::Expr(Spanned {
        node: ExprKind::For { body, .. },
        ..
    }) = &f.body.stmts[0]
    else {
        panic!()
    };
    // `i` and `x` are bound at the top of the body.
    assert!(matches!(&body.stmts[0], Stmt::Let(l) if l.name.node == "i"));
    assert!(matches!(&body.stmts[1], Stmt::Let(l) if l.name.node == "x"));
}

#[test]
fn if_is_pattern_becomes_a_match() {
    let f = only_fn(
        "fn f(xs: List[Int]):\n    var ys = xs\n    if xs.first() is Some(x):\n        print(x)\n    else:\n        \
         print(0)\n    while ys.pop() is Some(y):\n        print(y)\n",
    );
    assert!(matches!(
        &f.body.stmts[1],
        Stmt::Expr(Spanned {
            node: ExprKind::Match { arms, .. },
            ..
        }) if arms.len() == 2
    ));
    assert!(matches!(
        &f.body.stmts[2],
        Stmt::Expr(Spanned {
            node: ExprKind::While { .. },
            ..
        })
    ));
}

#[test]
fn match_with_cases() {
    let p = ok_program(
        "enum Shape:\n    Circle(Float)\n    Rect(Float, Float)\n    Empty\n\n\
         fn area(s: Shape) -> Float:\n    match s:\n        case Shape.Circle(r):\n            \
         r * r * 3.0\n        case Shape.Rect(w, h):\n            w * h\n        \
         case Shape.Empty:\n            0.0\n\n\
         fn classify(n: Int) -> Text:\n    match n:\n        case 0:\n            \"zero\"\n        \
         case _:\n            \"many\"\n",
    );
    let Item::Enum(e) = &p.items[0] else { panic!() };
    assert_eq!(e.variants.len(), 3);
    assert_eq!(e.variants[1].fields.len(), 2);
    let Item::Fn(f) = &p.items[1] else { panic!() };
    let ExprKind::Match { arms, .. } = &f.body.tail.as_ref().unwrap().node else {
        panic!("{:?}", f.body)
    };
    assert_eq!(arms.len(), 3);
    assert!(matches!(arms[2].pat.node, PatKind::Variant { .. }));
    assert!(matches!(&arms[0].body.node, ExprKind::Block(b) if b.tail.is_some()));
    let Item::Fn(g) = &p.items[2] else { panic!() };
    let ExprKind::Match { arms, .. } = &g.body.tail.as_ref().unwrap().node else {
        panic!()
    };
    assert!(matches!(arms[1].pat.node, PatKind::Wild));
    assert!(matches!(arms[0].pat.node, PatKind::Lit(Lit::Int(0))));
}

#[test]
fn types_traits_and_generics() {
    let p = ok_program(
        "trait Shape:\n    fn area(self) -> Float: ...\n    fn describe(self) -> Text:\n        \
         f\"area {self.area()}\"\n\n\
         type Circle(Shape):\n    r: Float\n\n    fn area(self) -> Float:\n        \
         self.r * self.r\n\n\
         type Stack[T]:\n    items: List[T]\n\n    fn push(self, x: T):\n        \
         self.items.push(x)\n\n\
         extend Circle:\n    fn twice(self) -> Float:\n        self.area() * 2.0\n",
    );
    let kinds: Vec<_> = p
        .items
        .iter()
        .map(|i| match i {
            Item::Trait(_) => "trait",
            Item::Struct(_) => "struct",
            Item::Impl(_) => "impl",
            _ => "other",
        })
        .collect();
    assert_eq!(kinds, ["trait", "struct", "impl", "struct", "impl", "impl"]);
    let Item::Trait(t) = &p.items[0] else {
        panic!()
    };
    assert!(t.fns[0].required && !t.fns[1].required);
    let Item::Impl(i) = &p.items[2] else { panic!() };
    assert_eq!(i.trait_name.as_ref().unwrap().node, "Shape");
    assert!(i.fns[0].params[0].name.node == "self");
    let Item::Impl(stack) = &p.items[4] else {
        panic!()
    };
    assert_eq!(stack.generics.len(), 1);
    assert_eq!(stack.ty_args.len(), 1);
}

#[test]
fn defaults_for_fields_and_parameters() {
    let p = ok_program(
        "type Query:\n    text: Text\n    top_k: Int = 5\n\nfn search(q: Query, limit: Int = 10):\n    pass\n",
    );
    let Item::Struct(s) = &p.items[0] else {
        panic!()
    };
    assert!(s.fields[0].default.is_none() && s.fields[1].default.is_some());
    let Item::Fn(f) = &p.items[1] else { panic!() };
    assert!(f.params[0].default.is_none() && f.params[1].default.is_some());
}

#[test]
fn type_can_still_be_a_name() {
    let f = only_fn("fn f(type: Text) -> Text:\n    type\n");
    assert_eq!(f.params[0].name.node, "type");
}

#[test]
fn private_names_start_with_an_underscore() {
    let p = ok_program("fn f():\n    pass\n\nfn _g():\n    pass\n");
    let Item::Fn(f) = &p.items[0] else { panic!() };
    let Item::Fn(g) = &p.items[1] else { panic!() };
    assert!(f.is_pub && !g.is_pub);
}

#[test]
fn endpoints_and_globals() {
    let p = ok_program(
        "let limit = 3\n\n@endpoint(POST, \"/x\")\nfn handle(p: In) -> Int:\n    limit\n\n\
         @endpoint(path: \"/y\", method: GET)\nfn other() -> Int:\n    1\n",
    );
    assert!(matches!(&p.items[0], Item::Let(l) if !l.mutable));
    let Item::Fn(f) = &p.items[1] else { panic!() };
    assert_eq!(f.attrs[0].name.node, "endpoint");
    let args: Vec<_> = f.attrs[0]
        .args
        .iter()
        .map(|(k, v)| (k.node.clone(), v.node.clone()))
        .collect();
    assert_eq!(args[0].0, "method");
    assert!(matches!(&args[0].1, ExprKind::Lit(Lit::Str(s)) if s == "POST"));
    assert_eq!(args[1].0, "path");
    let Item::Fn(g) = &p.items[2] else { panic!() };
    assert_eq!(g.attrs[0].args.len(), 2);
}

#[test]
fn a_pipeline_can_continue_on_the_next_lines() {
    let f =
        only_fn("fn f(q: Text) -> Text:\n    q\n        |> embed()\n        |> store.search(3)\n");
    assert!(matches!(
        f.body.tail.as_deref(),
        Some(Spanned {
            node: ExprKind::Pipe { .. },
            ..
        })
    ));
}

#[test]
fn errors_and_try() {
    let f = only_fn(
        "fn f(line: Text) -> Result[Int, Text]:\n    let key, value = split(line)?\n    Ok(1)\n",
    );
    assert_eq!(f.body.stmts.len(), 3);
    assert!(matches!(&f.body.stmts[0], Stmt::Let(l) if matches!(l.init.node, ExprKind::Try(_))));
}

#[test]
fn a_fn_inside_a_function_is_a_closure_bound_to_its_name() {
    let f = only_fn(
        "fn f(k: Int) -> Int:\n    fn add(a: Int):\n        let b = a + k\n        b\n    add(1)\n",
    );
    let Stmt::Let(l) = &f.body.stmts[0] else {
        panic!("{:?}", f.body)
    };
    assert_eq!(l.name.node, "add");
    let ExprKind::Closure {
        params,
        types,
        body,
    } = &l.init.node
    else {
        panic!()
    };
    assert_eq!(params.len(), 1);
    assert!(types[0].is_some());
    // The body is a block whose value is the final expression.
    assert!(matches!(&body.node, ExprKind::Block(b) if b.tail.is_some() && b.stmts.len() == 1));
}

#[test]
fn recovers_and_keeps_later_items() {
    let src = "fn a():\n    let x =\n    let y = 1\n\nfn b():\n    1 +\n\nfn c() -> Int:\n    3\n";
    let (p, errs) = parse(src);
    assert!(errs.len() >= 2, "{errs:?}");
    let names: Vec<_> = p
        .expect("recovery should still yield a program")
        .items
        .iter()
        .filter_map(|i| match i {
            Item::Fn(f) => Some(f.name.node.clone()),
            _ => None,
        })
        .collect();
    assert!(names.contains(&"c".to_string()), "{names:?}");
}

#[test]
fn recovers_from_garbage_between_items() {
    let (p, errs) =
        parse("type A:\n    x: Int\n\n??? )\n\nfn f():\n    pass\n\ntype B:\n    y: Int\n");
    assert!(!errs.is_empty());
    assert!(p.unwrap().items.len() >= 2);
}

#[test]
fn arrays_and_index_expressions() {
    assert!(matches!(ok_expr("[]"), ExprKind::Array(v) if v.is_empty()));
    assert!(matches!(ok_expr("[1, 2]"), ExprKind::Array(v) if v.len() == 2));
    assert!(matches!(ok_expr("a[i + 1][0]"), ExprKind::Index { .. }));
    assert!(matches!(ok_expr("f(x)[2].y"), ExprKind::Field { .. }));
    assert!(matches!(ok_expr("pair.0.1"), ExprKind::Field { .. }));
}

#[test]
fn spans_can_be_shifted_into_a_shared_offset_space() {
    let src = "fn f() -> Int:\n    1\n";
    let (plain, _) = parse(src);
    let (shifted, errs) = parse_at(src, 1000);
    assert!(errs.is_empty());
    let (Item::Fn(a), Item::Fn(b)) = (&plain.unwrap().items[0], &shifted.unwrap().items[0]) else {
        panic!()
    };
    assert_eq!(b.name.span.start, a.name.span.start + 1000);
    assert_eq!(b.span.end, a.span.end + 1000);
    // Interpolations inside f-strings are shifted too.
    let (p, _) = parse_at("fn g() -> Text:\n    f\"{a + 1}\"\n", 500);
    assert!(
        format!("{p:?}").contains("span: 5"),
        "expected shifted spans"
    );
}

#[test]
fn rag_example_parses() {
    let src = include_str!("../../../examples/rag-service.tvx");
    let p = ok_program(src);
    let fns: Vec<_> = p
        .items
        .iter()
        .filter_map(|i| match i {
            Item::Fn(f) => Some(f),
            _ => None,
        })
        .collect();
    assert_eq!(fns.len(), 4);
    let handler = fns.iter().find(|f| f.name.node == "handle_rag").unwrap();
    assert!(handler.is_pub);
    assert_eq!(handler.attrs[0].name.node, "endpoint");
    assert_eq!(handler.attrs[0].args.len(), 2);
    let embed = fns
        .iter()
        .find(|f| f.name.node == "generate_embedding")
        .unwrap();
    assert!(matches!(
        embed.body.tail.as_deref(),
        Some(Spanned {
            node: ExprKind::Pipe { .. },
            ..
        })
    ));
}

fn first_message(src: &str) -> String {
    let (_, errs) = parse(src);
    errs.into_iter()
        .next()
        .map(|e| e.message)
        .unwrap_or_default()
}

#[test]
fn syntax_errors_explain_what_to_write() {
    assert_eq!(
        first_message("fn f()\n    1\n"),
        "expected `:` at the end of this line"
    );
    assert_eq!(
        first_message("fn f():\n1\n"),
        "expected an indented block here"
    );
    assert_eq!(
        first_message("fn f():\n    let x = 1;\n"),
        "`;` is not used in Telvax"
    );
    assert_eq!(
        first_message("fn f():\n    1\n  2\n"),
        "this line is indented differently from the lines above it"
    );
    assert_eq!(
        first_message("fn f():\n\t1\n"),
        "tabs are not allowed in indentation"
    );
    assert_eq!(
        first_message("fn f():\n    for i in 0..3:\n        pass\n"),
        "`..` is not used in Telvax"
    );
}

#[test]
fn words_from_other_languages_get_a_hint() {
    let (_, errs) = parse("def f():\n    pass\n");
    assert!(
        errs.iter()
            .any(|e| e.message.contains("write `fn name(...)`"))
    );
    let (_, errs) = parse("class P:\n    x: Int\n");
    assert!(
        errs.iter()
            .any(|e| e.message.contains("write `type Name:`"))
    );
    let (_, errs) = parse("fn f():\n    mut x = 1\n");
    assert!(errs.iter().any(|e| e.message.contains("write `var`")));
    // A file that parses is left alone, even if a variable is called `def`.
    let (_, errs) = parse("fn f(def: Int) -> Int:\n    def\n");
    assert!(errs.is_empty(), "{errs:?}");
}

#[test]
fn function_types_and_typed_closure_parameters() {
    let f = only_fn(
        "fn f(g: (Int, Text) -> Bool, h: () -> (), k: (Int) -> (Int) -> Int) -> Int:\n    1\n",
    );
    let name = |t: &Type| t.path[0].node.clone();
    assert_eq!(name(&f.params[0].ty), "(fn)");
    assert_eq!(f.params[0].ty.args.len(), 3);
    assert_eq!(name(&f.params[1].ty), "(fn)");
    assert_eq!(f.params[1].ty.args.len(), 1);
    // The return type of a function type is the rest: `(Int) -> (Int) -> Int` curries.
    assert_eq!(f.params[2].ty.args.len(), 2);
    // A tuple is still a tuple.
    let f = only_fn("fn f(p: (Int, Text)) -> Int:\n    1\n");
    assert_eq!(f.params[0].ty.path[0].node, "(tuple)");
    let ExprKind::Closure { types, .. } = ok_expr("(x: Int, y) => x") else {
        panic!("not a closure")
    };
    assert!(types[0].is_some() && types[1].is_none());
}

#[test]
fn imports_can_be_renamed() {
    let p = ok_program("from a.b import C as D, e\nimport x.Y as Z\n");
    let Item::Import(i) = &p.items[0] else {
        panic!()
    };
    assert_eq!(i.group.len(), 2);
    assert_eq!(i.local_name(0, &i.group[0]).node, "D");
    assert_eq!(i.local_name(1, &i.group[1]).node, "e");
    let Item::Import(i) = &p.items[1] else {
        panic!()
    };
    assert_eq!(i.local_name(0, i.path.last().unwrap()).node, "Z");
    // `as` is still a name everywhere else.
    let (_, errs) = parse("fn f(as: Int) -> Int:\n    as\n");
    assert!(errs.is_empty(), "{errs:?}");
}

#[test]
fn mistakes_in_a_function_header_say_what_to_write() {
    let message = |src: &str| first_message(src);
    assert!(message("fn f(x: Int) -> Int {\n    x\n}\n").contains("not `{ }`"));
    assert!(message("fn f(x) -> Int:\n    x\n").contains("needs a type"));
    assert!(message("fn f(x: Int): Int:\n    x\n").contains("`-> Type`"));
    // The word's own message replaces the generic one.
    let (_, errs) = parse("def f():\n    pass\n");
    assert_eq!(errs.len(), 1, "{errs:?}");
    let (_, errs) = parse("use util::clamp\n");
    assert!(errs.iter().any(|e| e.message.contains("from a.b import C")));
}

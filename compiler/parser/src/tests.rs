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
    let p = ok_program("import util\n\ndef f() -> int:\n    return util.clamp(3)\n");
    let Item::Fn(f) = &p.items[1] else { panic!() };
    let Some(t) = f.body.tail.as_deref() else {
        panic!()
    };
    assert!(matches!(&t.node, ExprKind::Call { callee, .. }
        if matches!(&callee.node, ExprKind::Path(p) if p.len() == 2)));
}

#[test]
fn lambda_in_call() {
    assert!(matches!(
        ok_expr("map(lambda d: d.text)"),
        ExprKind::Call { .. }
    ));
    let ExprKind::Call { args, .. } = ok_expr("f(lambda a, b: a + b, lambda: 1)") else {
        panic!()
    };
    assert!(matches!(&args[0].node, ExprKind::Closure { params, .. } if params.len() == 2));
    assert!(matches!(&args[1].node, ExprKind::Closure { params, .. } if params.is_empty()));
}

#[test]
fn struct_construction_uses_keyword_arguments() {
    let ExprKind::StructLit { path, fields } = ok_expr("Foo(a=1, b=2,)") else {
        panic!()
    };
    assert_eq!(path.len(), 1);
    assert_eq!(fields.len(), 2);
    assert!(matches!(ok_expr("Foo(1, 2)"), ExprKind::Call { .. }));
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
fn types_are_spelled_like_python() {
    let f = only_fn(
        "def f(x: Tensor[float32, [768]], y: list[Tensor[int8, [N, 4]]], z: int | None, \
         w: dict[str, (int, float)]) -> list: ...\n",
    );
    assert_eq!(f.params[0].ty.path[0].node, "Tensor");
    assert!(matches!(&f.params[0].ty.args[0], TypeArg::Type(t) if t.path[0].node == "f32"));
    assert!(matches!(&f.params[0].ty.args[1], TypeArg::Shape(d) if d == &[Dim::Lit(768)]));
    assert_eq!(f.params[1].ty.path[0].node, "Vec");
    assert_eq!(f.params[2].ty.path[0].node, "Option");
    assert_eq!(f.params[3].ty.path[0].node, "Map");
    assert_eq!(f.ret.as_ref().unwrap().path[0].node, "Vec");
}

#[test]
fn return_at_the_end_is_the_value() {
    let f = only_fn("def f(x: int) -> int:\n    y = x + 1\n    return y * 2\n");
    assert_eq!(f.body.stmts.len(), 1);
    assert!(f.body.tail.is_some());
}

#[test]
fn return_in_every_branch_is_the_value() {
    let f = only_fn(
        "def f(n: int) -> str:\n    if n < 0:\n        return \"neg\"\n    elif n == 0:\n        \
         return \"zero\"\n    else:\n        return \"pos\"\n",
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
    let f = only_fn("def f(n: int) -> int:\n    if n < 0:\n        return 0\n    return n\n");
    assert_eq!(f.body.stmts.len(), 1);
    assert!(f.body.tail.is_some());
}

#[test]
fn first_assignment_declares_and_later_ones_reassign() {
    let f = only_fn("def f() -> int:\n    x = 1\n    x = x + 1\n    x += 2\n    return x\n");
    assert!(matches!(&f.body.stmts[0], Stmt::Let(l) if l.mutable && l.name.node == "x"));
    assert!(matches!(
        &f.body.stmts[1],
        Stmt::Expr(Spanned {
            node: ExprKind::Assign { .. },
            ..
        })
    ));
    assert!(matches!(
        &f.body.stmts[2],
        Stmt::Expr(Spanned {
            node: ExprKind::Assign { .. },
            ..
        })
    ));
}

#[test]
fn assignment_in_an_inner_block_updates_the_outer_name() {
    let f = only_fn(
        "def f() -> int:\n    total = 0\n    for i in range(3):\n        total += i\n        seen = i\n    \
         return total\n",
    );
    let Stmt::Expr(Spanned {
        node: ExprKind::For { body, .. },
        ..
    }) = &f.body.stmts[1]
    else {
        panic!()
    };
    assert!(matches!(
        &body.stmts[0],
        Stmt::Expr(Spanned {
            node: ExprKind::Assign { .. },
            ..
        })
    ));
    assert!(matches!(&body.stmts[1], Stmt::Let(l) if l.name.node == "seen"));
}

#[test]
fn typed_declaration_and_destructuring() {
    let f = only_fn("def f():\n    x: int = 1\n    a, b = g()\n");
    assert!(matches!(&f.body.stmts[0], Stmt::Let(l) if l.ty.is_some()));
    // `x`, then a temporary and one `let` per name.
    assert_eq!(f.body.stmts.len(), 4);
}

#[test]
fn loops_and_ranges() {
    let f = only_fn(
        "def f(v: list[int], n: int) -> int:\n    total = 0\n    i = 0\n    while i < n:\n        \
         total += v[i] * 2\n        i = i + 1\n    for x in v:\n        total -= x\n    \
         for k in range(n):\n        total += k\n    for k in range(1, 4):\n        total *= k\n    \
         return total\n",
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
fn for_with_two_names_and_tuples() {
    let f = only_fn("def f(xs: list[int]):\n    for i, x in xs.enumerate():\n        print(i)\n");
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
        "def f(xs: list[int]):\n    if xs.first() is Some(x):\n        print(x)\n    else:\n        \
         print(0)\n    while xs.pop() is Some(y):\n        print(y)\n",
    );
    assert!(matches!(
        &f.body.stmts[0],
        Stmt::Expr(Spanned {
            node: ExprKind::Match { arms, .. },
            ..
        }) if arms.len() == 2
    ));
    assert!(matches!(
        &f.body.stmts[1],
        Stmt::Expr(Spanned {
            node: ExprKind::While { .. },
            ..
        })
    ));
}

#[test]
fn match_with_cases() {
    let p = ok_program(
        "enum Shape:\n    Circle(float)\n    Rect(float, float)\n    Empty\n\n\
         def area(s: Shape) -> float:\n    match s:\n        case Shape.Circle(r):\n            \
         return r * r * 3.0\n        case Shape.Rect(w, h):\n            return w * h\n        \
         case Shape.Empty:\n            return 0.0\n\n\
         def classify(n: int) -> str:\n    match n:\n        case 0:\n            return \"zero\"\n        \
         case _:\n            return \"many\"\n",
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
fn classes_traits_and_generics() {
    let p = ok_program(
        "trait Shape:\n    def area(self) -> float: ...\n    def describe(self) -> str:\n        \
         return f\"area {self.area()}\"\n\n\
         class Circle(Shape):\n    r: float\n\n    def area(self) -> float:\n        \
         return self.r * self.r\n\n\
         class Stack[T]:\n    items: list[T]\n\n    def push(self, x: T):\n        \
         self.items.push(x)\n\n\
         extend Circle:\n    def twice(self) -> float:\n        return self.area() * 2.0\n",
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
fn private_names_start_with_an_underscore() {
    let p = ok_program("def f():\n    pass\n\ndef _g():\n    pass\n");
    let Item::Fn(f) = &p.items[0] else { panic!() };
    let Item::Fn(g) = &p.items[1] else { panic!() };
    assert!(f.is_pub && !g.is_pub);
}

#[test]
fn decorators_and_globals() {
    let p = ok_program(
        "limit = 3\n\n@endpoint(path=\"/x\", method=\"POST\")\ndef handle(p: In) -> int:\n    return limit\n",
    );
    assert!(matches!(&p.items[0], Item::Let(l) if !l.mutable));
    let Item::Fn(f) = &p.items[1] else { panic!() };
    assert_eq!(f.attrs[0].name.node, "endpoint");
    assert_eq!(f.attrs[0].args.len(), 2);
}

#[test]
fn a_pipeline_can_continue_on_the_next_lines() {
    let f = only_fn(
        "def f(q: str) -> str:\n    return q\n        |> embed()\n        |> store.search(3)\n",
    );
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
        "def f(line: str) -> Result[int, str]:\n    key, value = split(line)?\n    return Ok(1)\n",
    );
    assert_eq!(f.body.stmts.len(), 3);
    assert!(matches!(&f.body.stmts[0], Stmt::Let(l) if matches!(l.init.node, ExprKind::Try(_))));
}

#[test]
fn missing_colon_is_reported() {
    let (_, errs) = parse("def f()\n    return 1\n");
    assert!(!errs.is_empty());
}

#[test]
fn bad_indentation_is_reported() {
    let (_, errs) = parse("def f():\n        a = 1\n    b = 2\n");
    assert!(!errs.is_empty());
    let (_, errs) = parse("def f():\n\ta = 1\n");
    assert!(!errs.is_empty());
}

#[test]
fn recovers_and_keeps_later_items() {
    let src =
        "def a():\n    x = \n    y = 1\n\ndef b():\n    1 +\n\ndef c() -> int:\n    return 3\n";
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
        parse("class A:\n    x: int\n\n??? )\n\ndef f():\n    pass\n\nclass B:\n    y: int\n");
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
    let src = "def f() -> int:\n    return 1\n";
    let (plain, _) = parse(src);
    let (shifted, errs) = parse_at(src, 1000);
    assert!(errs.is_empty());
    let (Item::Fn(a), Item::Fn(b)) = (&plain.unwrap().items[0], &shifted.unwrap().items[0]) else {
        panic!()
    };
    assert_eq!(b.name.span.start, a.name.span.start + 1000);
    assert_eq!(b.span.end, a.span.end + 1000);
    // Interpolations inside f-strings are shifted too.
    let (p, _) = parse_at("def g() -> str:\n    return f\"{a + 1}\"\n", 500);
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

#[test]
fn a_def_inside_a_function_is_a_closure_bound_to_its_name() {
    let f = only_fn(
        "def f(k: int) -> int:\n    def add(a: int):\n        b = a + k\n        return b\n    return add(1)\n",
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
    // The body is a block whose value is the final `return`.
    assert!(matches!(&body.node, ExprKind::Block(b) if b.tail.is_some() && b.stmts.len() == 1));
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
        first_message("def f()\n    return 1\n"),
        "expected `:` at the end of this line"
    );
    assert_eq!(
        first_message("def f():\nreturn 1\n"),
        "expected an indented block here"
    );
    assert_eq!(
        first_message("def f():\n    x = 1;\n"),
        "`;` is not used in Telvax"
    );
    assert_eq!(
        first_message("def f():\n    return 1\n  x = 2\n"),
        "this line is indented differently from the lines above it"
    );
    assert_eq!(
        first_message("def f():\n\treturn 1\n"),
        "tabs are not allowed in indentation"
    );
    assert_eq!(
        first_message("def f():\n    for i in 0..3:\n        pass\n"),
        "`..` is not used in Telvax"
    );
}

#[test]
fn words_from_other_languages_get_a_hint() {
    let (_, errs) = parse("fn f() -> i32 { 1 }\n");
    assert!(
        errs.iter()
            .any(|e| e.message.contains("write `def name(...)`"))
    );
    let (_, errs) = parse("def f():\n    let x = 1\n");
    assert!(
        errs.iter()
            .any(|e| e.message.contains("write `name = value`"))
    );
    let (_, errs) = parse("struct P:\n    x: int\n");
    assert!(
        errs.iter()
            .any(|e| e.message.contains("write `class Name:`"))
    );
    // A file that parses is left alone, even if a variable is called `fn`.
    let (_, errs) = parse("def f(fn: int) -> int:\n    return fn\n");
    assert!(errs.is_empty(), "{errs:?}");
}

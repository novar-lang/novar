# Sintaxe do Telvax: didática, no estilo do Python

> Decisões tomadas: blocos por **indentação** e `:`; a sintaxe antiga (estilo Rust) é **substituída**, não mantida.
> Referência de motivação: `telvax-rumo-ao-1.0.md` (§1: "o alvo é o código de aplicação em Python"; §5.1 L1–L14).

## 1. Princípios

1. **Se o Python tem uma forma, o Telvax usa a mesma.** Quem lê Python lê Telvax sem tabela de tradução.
2. **Palavras antes de símbolos.** `and`, `or`, `not`, `is`, `in`, `elif`, `lambda`.
3. **Uma forma só** para cada coisa (sem `let`/`let mut`, sem `;`, sem `{ }`, sem `::`, sem `->` de closure).
4. **O que é do Telvax continua do Telvax**: tipos estáticos com inferência, `Tensor`/`Table`, `@endpoint`, e o pipeline `|>`. Eles não existem no Python, então não há o que copiar.
5. O compilador continua com a mesma AST: tudo o que muda aqui é **lexer + parser** (e as ferramentas que leem texto). Semântica, IR, interpretador e backend nativo não mudam.

## 2. Tabela de mudanças

| Antes (Rust) | Agora | Observação |
|---|---|---|
| `fn f(x: i32) -> i32 { x }` | `def f(x: int) -> int:` + `return x` | retorno explícito, como no Python |
| `{ ... }` | linha terminada em `:` + bloco indentado | 4 espaços; tabs são erro |
| `;` | quebra de linha | `()`, `[]`, `{}` abertos continuam a linha |
| `// comentário` | `# comentário` | |
| `let x = 1;` / `let mut x = 1;` | `x = 1` | a 1ª atribuição declara; as seguintes reatribuem |
| `let x: i64 = 1;` | `x: int = 1` | |
| `&&`, `\|\|`, `!` | `and`, `or`, `not` | |
| `true` / `false` | `True` / `False` | |
| `if a { } else if b { } else { }` | `if a:` / `elif b:` / `else:` | |
| `for i in 0..n` / `0..=n` | `for i in range(n)` / `range(0, n + 1)` | |
| `for (i, x) in v.enumerate()` | `for i, x in v.enumerate()` | sem parênteses |
| `let (a, b) = f();` | `a, b = f()` | |
| `\|x\| x + 1` / `\|\| 1` | `lambda x: x + 1` / `lambda: 1` | |
| `a::b::c()` | `a.b.c()` | `.` em tudo |
| `import a::b::{C, D};` | `from a.b import C, D` | `import a.b.C` também vale |
| `pub fn`, `pub struct` | `def`, `class` | privado = nome começando com `_` |
| `struct P { x: f64 }` | `class P:` com `x: float` | |
| `P { x: 1.0, y: 2.0 }` | `P(x=1.0, y=2.0)` | argumentos nomeados |
| `impl P { fn norm(self) }` | método dentro do `class` | |
| `impl Shape for Circle { }` | `class Circle(Shape):` | |
| `enum E { A(i32), B }` | `enum E:` com uma variante por linha | |
| `match x { A(r) => e, }` | `match x:` + `case A(r):` + bloco | |
| `if let Some(v) = e { }` | `if e is Some(v):` | idem `while … is …` |
| `Option<T>` | `T \| None` | |
| `Vec<T>`, `Map<K, V>`, `String` | `list[T]`, `dict[K, V]`, `str` | `[]` para genéricos, como no Python 3.9+ |
| `i64`, `f64` | `int`, `float` | tamanhos explícitos: `int32`, `uint8`, `float32`… (nomes do NumPy) |
| `fn f<T: A + B>(x: T)` | `def f[T: (A, B)](x: T)` | sintaxe do Python 3.12 (PEP 695) |
| `@endpoint(path = "/x")` | `@endpoint(path="/x")` | igual ao decorador do Python |
| `println(...)` | `print(...)` | `print` sempre termina a linha |
| `x \|> f(a)` | `x \|> f(a)` | **fica**: é do Telvax |
| `expr?` | `expr?` | **fica** por enquanto (ver §6) |

## 3. Exemplos

### Funções, tipos e laços

```python
class Point:
    x: float
    y: float

    def norm(self) -> float:
        return self.x * self.x + self.y * self.y

def main() -> int:
    p = Point(x=3.0, y=4.0)
    total = 0                    # literal sem tipo: o tipo vem dos usos
    for i in range(10):
        total += i
    print(f"norm = {p.norm()}, total = {total}")
    return total
```

### Condições, `in`, expressão condicional

```python
def kind(n: int) -> str:
    if n < 0:
        return "negativo"
    elif n == 0:
        return "zero"
    else:
        return "positivo"

def vowel(c: str) -> bool:
    return c in ["a", "e", "i", "o", "u"]

label = "par" if n % 2 == 0 else "ímpar"
```

### Enums e `match`

```python
enum Shape:
    Circle(float)
    Rect(float, float)
    Empty

def area(s: Shape) -> float:
    match s:
        case Shape.Circle(r):
            return r * r * 3.14159
        case Shape.Rect(w, h):
            return w * h
        case Shape.Empty:
            return 0.0
```

### Erros e opcionais

```python
def split_pair(line: str) -> Result[(str, str), str]:
    parts = line.split("=")
    if parts.len() != 2:
        return Err(f"linha inválida: {line}")
    return Ok((parts[0].trim(), parts[1].trim()))

def port_of(line: str) -> Result[int, str]:
    key, value = split_pair(line)?          # `?` devolve o Err mais cedo
    match value.to_int():
        case Some(n):
            return Ok(n)
        case None:
            return Err(f"não é número: {value}")

def main():
    stack = [1, 2, 3]
    while stack.pop() is Some(top):
        print(f"{top}")
    if "a b c".split(" ").first() is Some(first):
        print(f"primeira: {first}")
```

### Traits, genéricos e métodos

```python
trait Shape:
    def area(self) -> float: ...             # sem corpo: toda implementação precisa ter
    def describe(self) -> str:               # com corpo: valor padrão
        return f"área {self.area()}"

class Circle(Shape):
    r: float
    def area(self) -> float:
        return self.r * self.r * 3.14159

class Stack[T]:
    items: list[T]
    def push(self, x: T):
        self.items.push(x)
```

### Dados, tensores, serviços

```python
class Sale:
    id: int
    region: str
    units: int
    price: float

def main():
    sales: Table[Sale] = read_csv("sales.csv")
    print(f"preço médio: {sales.price.mean()}")
    by_region: Table[{region: str, revenue: float}] = sales.sql(
        "SELECT region, SUM(units * price) AS revenue FROM t GROUP BY region")
    for row in by_region:
        print(f"{row.region}: {row.revenue}")

def forward(x: Tensor[float32, [2, 3]], w: Tensor[float32, [3, 2]]) -> Tensor[float32, [2, 2]]:
    return x |> std.tensor.matmul(w) |> std.tensor.relu()

@endpoint(path="/v1/score", method="POST")
def score(input: In) -> Result[Out, Status]:
    if input.text.is_empty():
        return Err(Status.BadRequest("texto vazio"))
    answer = input.text |> llm.complete() |> std.text.trim()
    return Ok(Out(answer=answer))
```

Um pipeline longo quebra a linha com `|>` no começo da linha seguinte:

```python
    docs = query
        |> embed()
        |> store.search(top_k)
```

## 4. Regras do lexer e do parser

* **Indentação**: cada nível são 4 espaços (qualquer quantidade fixa por bloco é aceita; misturar tabs é erro). Linhas em branco e só com comentário não contam.
* Dentro de `()`, `[]`, `{}` as quebras de linha são ignoradas. Uma linha que **começa** com `|>` continua a anterior.
* Um corpo de uma linha só é permitido para `lambda` (`lambda x: x + 1`) e `case`/`if` simples não: **o corpo sempre vai na linha seguinte, indentado** (uma forma só).
* `return` no fim do corpo vira o valor do bloco; `if`/`elif`/`else`/`match` no fim propagam isso para cada ramo. Por isso a semântica (que trabalha com "expressão final") não muda.
* Declaração vs reatribuição: o parser guarda os nomes já atribuídos em cada função. `x = …` de um nome novo é declaração (mutável); de um nome existente é reatribuição. `x: T = …` sempre declara.
* Caminhos: `std.text.trim`, `Shape.Circle`, `util.clamp` e `modulo.Tipo.novo` são caminhos quando o primeiro nome é `std`, um nome importado ou começa com maiúscula; caso contrário `a.b` é acesso a campo/método.
* `P(x=1, y=2)` com argumentos nomeados constrói um struct; `Shape.Circle(2.0)` chama uma variante.
* `a in b` equivale a `b.contains(a)`; `a not in b` a `not b.contains(a)`.
* `x if c else y` é a expressão condicional.
* Palavras reservadas: `def class enum trait extend import from as return if elif else while for in is match case break continue pass lambda and or not True False None`.

## 5. Plano de implementação

1. **Lexer**: `#` como comentário; tokens `Newline`, `Indent`, `Dedent`; novas palavras reservadas; remoção de `::`, `{ }` em blocos (as chaves ficam só para o tipo-linha `Table[{…}]`), `;`, `&&`, `||`, `!`, `..`, `=>`.
2. **Parser**: reescrita das regras de item, bloco e expressão sobre a mesma AST (`telvax-ast` não muda).
3. **Migração mecânica**: o parser antigo fica por um tempo como `tools/migrate` (`telvax migrate arquivo.tvx`), que lê a sintaxe antiga e imprime a nova a partir da AST. Com ele convertem-se os 10 exemplos `.tvx`, os ~500 trechos de código embutidos nos testes em Rust e a documentação. Depois a ferramenta é apagada.
4. **Formatador** (`telvax fmt`): refeito sobre os novos tokens (indentação é significativa, então ele só normaliza espaços e linhas em branco).
5. **Linter**, **LSP** (hover, definição, renomear) e a **gramática do VS Code**: adaptados. O aviso "`mut` desnecessário" some (não há mais `mut`).
6. **Docs**: `tour.md`, `use-cases.md`, `status.md`, README, CHANGELOG, playground.

Critério de saída: todos os testes existentes passam com os fontes migrados; `telvax fmt --check` limpo em `examples/`; os exemplos rodam com a mesma saída de antes.

## 6. Pontos em aberto (segui o padrão; me diga se quer mudar)

* `Some(x)`, `Ok(x)`, `Err(e)` e o `?` **ficam** como estão: trocá-los por exceções (`raise`/`try`) é uma mudança de semântica, não de sintaxe, e pertence à 0.0.6 (L3 do documento "rumo ao 1.0").
* `pair.0` para tuplas continua (o `pair[0]` do Python não distingue tupla de lista na AST atual).
* Sem fatias (`xs[1:3]`), sem `**`, sem `//` (divisão inteira) e sem strings de várias linhas (`"""`) por enquanto; entram junto com L12.
* `lambda` com parâmetro anotado deixa de existir; use `def`.
* Modo script (arquivo sem `def main`) é o L13 e fica para depois; hoje o ponto de entrada continua sendo `main`.

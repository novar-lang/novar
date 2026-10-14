# Sintaxe do Telvax: didática, no estilo do Python

> **Implementada na 0.1.0.** Decisões: blocos por **indentação** e `:`; a sintaxe antiga (estilo Rust) foi
> **substituída**, não mantida. A seção 7 lista onde o resultado difere da proposta original.

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
| `expr?` | `expr?` | **fica** por enquanto (ver §7) |

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

* **Indentação**: cada nível são 4 espaços (o lexer aceita qualquer quantidade fixa por bloco, mas o `telvax fmt` normaliza para 4). Tabs na indentação são erro, e também voltar a um nível que nunca foi aberto. Linhas em branco e só com comentário não contam.
* Dentro de `()`, `[]`, `{}` as quebras de linha são ignoradas. Uma linha que **começa** com `|>` continua a anterior.
* **O corpo sempre vai na linha seguinte, indentado** (uma forma só). A exceção é `lambda x: x + 1`, que é uma expressão, e `def f(self) -> float: ...` num `trait`, que marca um método obrigatório.
* `return` no fim do corpo vira o valor do bloco; `if`/`elif`/`else`/`match` no fim propagam isso para cada ramo. Por isso a semântica (que trabalha com "expressão final") não mudou. Um `return` sem valor, ou no meio do corpo, continua sendo um `return`.
* Declaração vs reatribuição: o parser guarda os nomes já atribuídos em cada função, bloco a bloco. `x = …` de um nome novo é declaração (variável mutável); de um nome que já existe é reatribuição. `x: T = …` sempre declara. `x += 1` nunca declara.
* Caminhos: `std.text.trim`, `Shape.Circle`, `util.clamp` e `Modulo.Tipo.novo` são caminhos quando o primeiro nome é `std`, um nome trazido por `import`/`from` ou começa com maiúscula; caso contrário `a.b` é acesso a campo/método. (`import a.b.C` traz só `C`; `from a.b import C, D` traz `C` e `D`.)
* `P(x=1, y=2)` com argumentos nomeados constrói uma classe; `Shape.Circle(2.0)` chama uma variante.
* `a in b` equivale a `b.contains(a)`; `a not in b` a `not b.contains(a)`.
* `x if c else y` é a expressão condicional.
* Um `def` dentro de uma função é um fechamento ligado ao nome (`add = lambda …`, mas com várias linhas). Os parâmetros podem ter tipo; o retorno é inferido.
* `x is Padrão` só existe como condição de `if`/`elif`/`while`; `while xs.pop() is Some(top):` repete enquanto o padrão casar.
* Palavras reservadas: `def class enum trait extend import from return if elif else while for in is match case break continue pass lambda and or not True False`. `None`, `Some`, `Ok` e `Err` são nomes comuns (variantes de `Option`/`Result`).
* Um nome que começa com `_` é privado do arquivo; os outros são públicos. Não existe `pub`.

## 5. Mensagens de erro

O parser reconhece o que quem vem de outras linguagens costuma escrever e diz o que escrever:

| Escrito | Mensagem |
|---|---|
| `def f()` sem `:` | ``expected `:` at the end of this line`` |
| bloco sem indentação | `expected an indented block here` |
| `x = 1;` | ``\`;\` is not used in Telvax`` (nota: termine a linha) |
| `fn`, `let`, `mut`, `struct`, `impl`, `pub` no começo da linha | ``\`fn\` is not used in Telvax: write \`def name(...)\` `` e equivalentes |
| `for i in 0..3` | ``\`..\` is not used in Telvax`` (nota: `range(start, end)`) |
| `&&`, `!` | ``write \`and\``, ``write \`not\`` |
| `true`, `false`, `null` | dica: ``did you mean \`True\`?`` |
| tabs / indentação que não alinha | mensagens próprias |

Os tipos aparecem nas mensagens como são escritos no programa (`expected int32, found str`, `list[int]`, `int | None`), e os caminhos com ponto (`cannot find util.double`).

## 6. Ferramentas

* **`telvax fmt`** foi refeito para indentação: mantém cada quebra de linha e cada bloco e normaliza espaços, continuações de linha, comentários e linhas em branco. Verifica que o resultado tem os mesmos tokens (inclusive `Indent`/`Dedent`) antes de devolvê-lo.
* **`telvax lint`**: acabou o aviso "`mut` desnecessário"; uma função privada sem uso é a que começa com `_`.
* **LSP**: hover (`def norm(p: Point) -> float`, `origin: Point`), dicas de tipo para `x = …` e `for x in …`, completação de `std.` e de membros, renomear com a nova sintaxe.
* **VS Code**: gramática TextMate nova, regras de indentação (`:` abre bloco) e `#` como comentário. Playground e exemplos migrados.
* **`telvax-migrate`** (`tools/migrate`): lê a sintaxe antiga com uma cópia do parser antigo e imprime a nova a partir da AST, preservando comentários. Foi usada para converter os exemplos e os ~500 trechos de código dos testes. Sai numa versão futura.

## 7. Diferenças em relação à proposta e pontos em aberto

* `Some(x)`, `Ok(x)`, `Err(e)` e o `?` **ficam** como estão: trocá-los por exceções (`raise`/`try`) é uma mudança de semântica, não de sintaxe, e pertence à 0.0.6 (L3 do documento "rumo ao 1.0").
* **Intervalos fechados** (`0..=n`) deixaram de existir: use `range(0, n + 1)`. O custo é que um intervalo que termina no maior inteiro do tipo não é mais expressável (a AST ainda tem `inclusive`, sem uso).
* **`if`/`match` como valor** não existem: use `a if c else b`, ou atribua dentro dos ramos. Blocos como expressão (`z = { … }`) também não.
* **`mut`**: toda variável e todo parâmetro pode ser reatribuído. Continuam imutáveis a variável de `for`, os nomes de um padrão e os globais.
* `lambda` com parâmetro anotado deixou de existir; use um `def` dentro da função.
* `pair.0` para tuplas continua (o `pair[0]` do Python não distingue tupla de lista na AST atual).
* Sem fatias (`xs[1:3]`), sem `**`, sem `//` (divisão inteira) e sem strings de várias linhas (`"""`) por enquanto; entram junto com L12.
* Modo script (arquivo sem `def main`) é o L13 e fica para depois; o ponto de entrada continua sendo `main`. Um executável nativo ainda exige `def main() -> int32`.
* Valores impressos seguem o Python (`True`, `None`, `Point(x=1, y=2)`); `print` termina a linha e `std.io.print` não.

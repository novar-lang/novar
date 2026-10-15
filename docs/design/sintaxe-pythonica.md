# Sintaxe do Telvax: fácil como o Python, com personalidade própria

> **Implementada na 0.1.0.** Decisões: blocos por **indentação** e `:`; a sintaxe antiga (estilo Rust) foi
> **substituída**. O Telvax não é "Python com outro nome": copia o que é didático e troca o que não serve a
> dados, ML e IA.

## 1. Princípios

1. **Quem lê Python lê Telvax.** Indentação, `:`, `and`/`or`/`not`, `elif`, `for x in xs`, f-strings, `#`.
2. **Palavras próprias onde o Python é fraco.** `fn`, `let`/`var`, `type`, `x => x + 1`, `Int?`.
3. **O que o Python deixa para a execução, o Telvax checa antes**: tipos, formas de tensor, colunas de tabela, nomes de argumentos.
4. **O que é do Telvax continua do Telvax**: `|>`, `@endpoint`, `Tensor`, `Table`.
5. A AST, o verificador, o interpretador e o backend nativo são os mesmos; a mudança está no lexer, no parser e nas ferramentas que leem texto.

## 2. Dores do Python em dados, ML e IA, e a resposta

| Dor | O que o Telvax faz |
|---|---|
| Erro de forma de tensor só aparece rodando | `Tensor[f32, 2, 3]`: a forma é parte do tipo e `matmul` confere em compilação |
| Coluna digitada errada num DataFrame | `Table[Row]`: nomes e tipos das colunas são checados |
| Variável sobrescrita sem querer | `let` não muda; `var` muda. `x = 1` sem declarar é erro |
| `None` aparecendo onde não devia | `T?` é um tipo; `match` obriga a tratar `None` |
| Exceções escondidas em pipelines | `Result` e `?`; nada é lançado |
| Configuração com dezenas de argumentos | argumentos nomeados e valores padrão, conferidos |
| Encadear etapas lê de dentro para fora | `x |> f() |> g()` lê da esquerda para a direita |
| Servir um modelo exige framework e cola | `@endpoint(POST, "/rota")` vira rota HTTP validada |
| Tipos opcionais e ignorados | tipos estáticos com inferência; nada de `Any` |

## 3. A sintaxe

```telvax
from std.ai.embeddings import OllamaEmbedder

let embedder = OllamaEmbedder.new("nomic-embed-text")

type Query:
    text: Text
    top_k: Int = 5

type Reply:
    query: Text
    answer: Text

fn embed(text: Text) -> Tensor[f32, 768]:
    text
        |> std.text.clean_whitespace()
        |> std.text.truncate(512)
        |> embedder.embed()

@endpoint(POST, "/v1/rag/query")
fn handle(q: Query) -> Result[Reply, Status]:
    if q.text.is_empty():
        return Err(Status.BadRequest("empty query"))
    let docs = db.search(embed(q.text), limit: q.top_k)
    var answer = docs |> prompt(q.text) |> llm.complete()
    Ok(Reply(query: q.text, answer: answer.trim()))
```

| Forma | Exemplo |
|---|---|
| função | `fn f(a: Int, b: Int = 1) -> Int:` (o último valor é o resultado; `return` sai antes) |
| variável | `let x = 1` (fixa), `var x = 1` (muda), `let a, b = par` |
| tipos | `Int`, `Float`, `Text`, `Bool`, `List[T]`, `Map[K, V]`, `T?`, `Result[T, E]`, `Tensor[f32, 2, 3]`, `Table[Row]`, tuplas `(A, B)`, funções `(A, B) -> C` (`() -> C`; `(A) -> ()` quando não devolve nada); números de tamanho fixo `i8`..`i64`, `u8`..`u64`, `f32`, `f64` |
| tipo próprio | `type P:` com campos (`x: Float`, `n: Int = 5`) e métodos; `type C(Trait):`; `type Pilha[T]:` |
| enum, trait, extensão | `enum E:` (uma variante por linha), `trait T:` (`fn f(self) -> Int: ...` é obrigatório), `extend P:` |
| construção | `P(x: 1.0, y: 2.0)` |
| chamada com nomes | `search(q, limit: 3)` |
| função como valor | `x => x + 1`, `(a, b) => a + b`, `() => 1`; `((i, x)) => ...` abre uma tupla; um `fn` dentro de um `fn` é um fechamento de várias linhas |
| decisão | `if`/`elif`/`else`, `a if c else b`, `match x:` + `case Padrão:`, `if x is Some(v):`, `while x is Some(v):` |
| laços | `for x in xs:`, `for i, x in xs:`, `for i in range(n)`, `range(a, b)`, `while`, `break`, `continue`, `pass` |
| operadores | `and or not in`, `|>`, `?` |
| módulos | `from a.b import C, D`, `import a.b.C`, `.` em todo caminho (`std.text.trim`, `Shape.Circle`) |
| privado | nome começando com `_`; não existe `pub` |
| endpoint | `@endpoint(POST, "/rota")` ou `@endpoint(method: POST, path: "/rota")` |
| comentário | `# ...` |

## 4. Regras do lexer e do parser

* **Indentação**: cada nível são 4 espaços na saída do `telvax fmt`. Tabs na indentação são erro, e também voltar a um nível que nunca foi aberto. Linhas em branco e só com comentário não contam.
* Dentro de `()`, `[]`, `{}` as quebras de linha são ignoradas. Uma linha que **começa** com `|>` continua a anterior.
* **O corpo sempre vai na linha seguinte, indentado.** Exceção: `x => x + 1` é uma expressão, e `fn f(self) -> Float: ...` num `trait` marca um método obrigatório. `...` sozinho numa linha é um corpo vazio, como `pass`.
* **O valor de uma função** é a última expressão do corpo (ou do corpo de cada ramo de um `if`/`match` final), quando a função declara o tipo de retorno. Um `return` no meio continua sendo um `return`. Numa função de topo sem `->`, o tipo de retorno é o da expressão final (`fn dobro(x: Int): x * 2` devolve `Int`); `main`, `test_*`, endpoints, genéricas e métodos precisam do `->` para devolver um valor.
* `let` e `var` declaram; `x = v` só muda uma variável `var` (ou um campo/elemento dela). Variáveis de `for` e nomes de padrões não mudam.
* Caminhos: `std.text.trim`, `Shape.Circle`, `util.clamp` são caminhos quando o primeiro nome é `std`, um nome trazido por `import`/`from` ou começa com maiúscula; senão `a.b` é campo/método.
* `Nome(campo: valor, ...)` com todos os argumentos nomeados e `Nome` maiúsculo constrói um tipo; campos com valor padrão podem ser omitidos.
* Nomes e valores padrão em chamadas são resolvidos antes da verificação (`compiler/semantic/src/named.rs`): para funções e métodos definidos no programa os argumentos são reordenados e os padrões preenchidos; para o resto (biblioteca, métodos de valores sem tipo conhecido) o nome é só um rótulo. `x.m(...)` só usa a assinatura de `m` se um único tipo do programa tem um método com esse nome.
* `a in b` equivale a `b.contains(a)`; `x is Padrão` só existe como condição de `if`/`elif`/`while`.
* `type` não é palavra reservada: só abre uma declaração no começo de um item, então ainda pode ser nome de campo ou variável (`type: Text`).
* Palavras reservadas: `fn let var enum trait extend import from return if elif else while for in is match case break continue pass and or not True False`. `None`, `Some`, `Ok` e `Err` são nomes comuns.

## 5. Mensagens de erro

| Escrito | Mensagem |
|---|---|
| `fn f()` sem `:` | ``expected `:` at the end of this line`` |
| bloco sem indentação | `expected an indented block here` |
| `x = 1;` | ``\`;\` is not used in Telvax`` |
| `def`, `class`, `lambda`, `mut`, `struct`, `impl`, `pub` no começo da linha | ``\`def\` is not used in Telvax: write \`fn name(...)\` `` e equivalentes |
| `for i in 0..3` | ``\`..\` is not used in Telvax`` (nota: `range(start, end)`) |
| `x = 1` sem declarar | ``cannot find \`x\`...`` com a nota `declare it first with let x = ... or var x = ...` |
| `int`, `str`, `list` como tipos | ``cannot find type \`int\`...`` com `did you mean \`Int\`?` |
| `true`, `false`, `null` | ``did you mean \`True\`?`` |
| nome de argumento errado | ``no parameter \`speed\`` (e a lista dos que existem) |

Os tipos aparecem como são escritos no programa: ``expected `i32`, found `Text` ``, `List[Int]`, `Int?`, `Map[Text, Float]`.

## 6. Ferramentas

* **`telvax fmt`** mantém cada quebra de linha e cada bloco e normaliza espaços, continuações, comentários e linhas em branco; confere que o resultado tem os mesmos tokens (inclusive `Indent`/`Dedent`).
* **`telvax lint`**: variáveis e funções privadas sem uso (as privadas são as que começam com `_`).
* **LSP**: hover (`fn norm(p: Point) -> Float`, `let origin: Point`, `var total: f32`), dicas de tipo para `let`/`var`/`for`, completação de `std.` e de membros, renomear.
* **VS Code**: gramática TextMate, regras de indentação e `#` como comentário. Playground e exemplos migrados.
* **`telvax-migrate`** (`tools/migrate`): converte arquivos da sintaxe antiga (estilo Rust), mantendo os comentários. Sai numa versão futura.

## 7. O que ficou de fora (por enquanto)

* `Some`, `Ok`, `Err` e o `?` ficam: trocá-los é mudança de semântica (0.0.6 do roadmap).
* Uniões em tipos de retorno (`Reply | Error`) no lugar de `Result`.
* `match`/`if` como valor em qualquer posição (hoje só na posição final de uma função, e `a if c else b`).
* Intervalos fechados (`0..=n`): use `range(0, n + 1)`.
* Fatias (`xs[1:3]`), strings de várias linhas, `//` de divisão inteira, `**`; entram com L12 do roadmap.
* Modo script (arquivo sem `fn main`); um executável nativo ainda exige `fn main() -> i32`.
* Argumentos nomeados em chamadas de biblioteca são só rótulos (não reordenam).

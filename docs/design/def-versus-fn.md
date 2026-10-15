# `def` ou `fn`?

> **Avaliado. Decisão: manter `fn`.** Este texto guarda o raciocínio e diz o que faria a decisão mudar.
> Não há medição de uso por trás dele: a conclusão vem do desenho da linguagem.

## 1. A pergunta

O Telvax lê como Python quase em tudo (ver `sintaxe-pythonica.md`). Uma função, porém, começa com `fn`, e
quem vem do Python escreve `def`. Vale aceitar `def`, no lugar de `fn` ou ao lado dele?

## 2. O que existe hoje

* A palavra é `fn`. O lexer, o parser, a completação do servidor de linguagem e a gramática do editor a
  conhecem.
* `def` no começo de uma linha é um erro que diz o que escrever:
  ``\`def\` is not used in Telvax: write \`fn name(...)\` ``. É o único erro dado para essa linha.
* O migrador (`tools/migrate`) lê `def` de arquivos Python e escreve `fn`.
* Os exemplos têm 18 funções, e o tour e o desenho da sintaxe mostram as funções com `fn`.

## 3. As opções

| | Opção | Como fica |
|---|---|---|
| A | Só `fn` (hoje) | `fn area(r: Float) -> Float:` |
| B | Só `def` | `def area(r: Float) -> Float:` |
| C | As duas, com o mesmo significado | `fn` e `def` valem, e cada um escolhe |
| D | As duas, com significados diferentes | `def` com regras mais frouxas que `fn` |

## 4. Critérios

**Quem lê Python.** É o argumento a favor do `def`, e é real. O primeiro princípio da linguagem é "quem lê
Python lê Telvax". Uma função com tipos escritos, `def area(r: Float) -> Float:`, já é Python válido. O erro
atual custa uma linha de leitura, mas é uma barreira logo na primeira linha.

**O `def` promete regras do Python que o Telvax não tem.** Em Python, `def f(x):` vale, com parâmetro sem
tipo. No Telvax todo parâmetro tem tipo, e não existe `Any`. Com `def`, a mesma palavra teria regras
diferentes das que quem a conhece espera, e o erro viria depois, em outra linha: "a parameter needs a type".
Com `fn`, a palavra nova avisa que as regras são outras. Este é o segundo princípio: palavras próprias onde o
Python é fraco.

**Copiar e colar do Python continua não funcionando.** O ganho do `def` seria colar uma função do Python e
ela rodar. Mas o corpo quase nunca é aceito como está: `let` e `var` no lugar da atribuição simples, `type`
no lugar de `class`, `x => x + 1` no lugar de `lambda`, e `T?` e `Result` no lugar de `None` e exceções.
A primeira linha igual não faz o resto igual.

**Uma só forma de escrever.** A opção C dá duas grafias para a mesma coisa. Cada projeto escolheria uma,
o formatador não trocaria uma pela outra (ele confere que o resultado tem os mesmos tokens, então
normalizar seria uma regra nova),
os exemplos e a documentação teriam de mostrar uma, e o servidor de linguagem, a gramática do editor, o
migrador e as mensagens de erro teriam de tratar as duas. O custo técnico é pequeno (uma palavra a mais no
lexer). O custo real é a decisão repetida em cada arquivo.

**Duas palavras com significados diferentes (D) não têm o que significar.** Há linguagens com as duas
palavras e regras diferentes para tipos e erros (o Mojo é uma). No Telvax todo parâmetro tem tipo, então não
há diferença para dar.

**Trocar `fn` por `def` (B) quebra tudo que existe.** Todo exemplo, teste, documento e pacote usa `fn`. Seria
uma mudança incompatível (`feat!`) sem o ganho da opção C.

## 5. Avaliação

| Critério | A: só `fn` | B: só `def` | C: as duas | D: diferentes |
|---|---|---|---|---|
| Familiar a quem lê Python | em parte | sim | sim | em parte |
| Regras que a palavra promete | claras | enganosas | enganosas para `def` | claras |
| Uma forma só | sim | sim | **não** | não |
| Compatível com o que existe | sim | **não** | sim | sim |
| Custo de manter | nenhum | alto (migração) | médio | alto |

## 6. Recomendação

Manter **`fn`** (opção A) e manter o erro que diz o que escrever. A familiaridade do `def` é um bom
argumento, mas ele compra pouco: a primeira linha igual não faz o resto igual, e traz de volta a expectativa
de parâmetros sem tipo que a linguagem recusa em seguida.

O que já protege quem escreve `def`: a mensagem de erro, o migrador, e o tour, que começa por `fn`.

## 7. O que mudaria a decisão

* **Dados.** Hoje não há nenhum: o Telvax não mede o que as pessoas escrevem. Se o playground ou as
  mensagens de erro passarem a registrar quantas vezes `def` aparece em uma primeira tentativa, e o número for
  alto, a opção C (as duas grafias) volta a ser considerada, com o formatador normalizando para `fn`.
* **Parâmetros sem tipo.** Se a linguagem passasse a aceitar `def f(x):` com o tipo inferido, `def` teria
  o mesmo significado que no Python, e a opção D deixaria de ser vazia. Isso muda o princípio de que tudo
  tem tipo escrito ou inferido do uso, e é uma decisão de linguagem à parte.

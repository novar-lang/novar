// Programs offered in the playground. `run` is the function started by default.
export const EXAMPLES = [
  {
    name: "Hello, Telvax",
    run: "main",
    source: `def main() -> int32:
    print("hello, world")
    name = "Telvax"
    print(f"{name} has {name.len()} letters")
    return 0
`,
  },
  {
    name: "Primes",
    run: "main",
    source: `def is_prime(n: int) -> bool:
    if n < 2:
        return False
    d = 2
    while d * d <= n:
        if n % d == 0:
            return False
        d += 1
    return True

def main() -> int32:
    limit: int = 100
    count = 0
    for n in range(limit):
        if is_prime(n):
            count += 1
    print(f"primes below {limit}: {count}")
    return count
`,
  },
  {
    name: "Sieve of Eratosthenes",
    run: "main",
    source: `def main() -> int32:
    is_prime = [True, True, True, True, True, True, True, True, True, True,
        True, True, True, True, True, True, True, True, True, True,
        True, True, True, True, True, True, True, True, True, True]
    n = is_prime.len()
    is_prime[0] = False
    is_prime[1] = False

    p = 2
    while p * p < n:
        if is_prime[p]:
            multiple = p * p
            while multiple < n:
                is_prime[multiple] = False
                multiple += p
        p += 1

    line = ""
    count = 0
    for i in range(n):
        if is_prime[i]:
            line = line + f"{i} "
            count += 1
    print(line)
    return count
`,
  },
  {
    name: "Tensors with checked shapes",
    run: "forward",
    source: `# Shapes are part of the type: change [3, 2] to [4, 2] and the compiler objects.
def forward(x: Tensor[float32, [2, 3]], w: Tensor[float32, [3, 2]]) -> Tensor[float32, [2, 2]]:
    return x |> std.tensor.matmul(w) |> std.tensor.relu()
`,
    args: [[[1,-2,3],[4,5,-6]],[[1,0],[0,1],[1,1]]],
  },
  {
    name: "Pipelines and classes",
    run: "summarize",
    source: `class Review:
    stars: int
    text: str

def clean(s: str) -> str:
    return s |> std.text.clean_whitespace() |> std.text.trim() |> std.text.lowercase()

def summarize(reviews: list[Review]) -> str:
    texts = reviews |> std.iter.map(lambda r: clean(r.text))
    stars = 0
    for r in reviews:
        stars += r.stars
    return f"{reviews.len()} reviews, {stars} stars: " + (texts |> std.string.join(" | "))
`,
    args: [[{"stars":5,"text":"  Fast   AND reliable "},{"stars":4,"text":"Great docs"}]],
  },
  {
    name: "RAG endpoint (offline stand-ins)",
    run: "handle_rag",
    source: `import std.ai.embeddings.OllamaEmbedder
import std.ai.llm.OpenAIClient
import std.db.pgvector.PgVectorClient
import std.http.Status

embedder = OllamaEmbedder.new("nomic-embed-text")
vector_db = PgVectorClient.connect("postgresql://localhost:5432/vectors")
llm = OpenAIClient.new("gpt-4o-mini")

class QueryPayload:
    query: str
    top_k: int32

class RagResponse:
    query: str
    retrieved_context: Vector
    answer: str

def generate_embedding(text: str) -> Tensor[float32, [768]]:
    return text |> std.text.clean_whitespace() |> std.text.truncate(512) |> embedder.embed()

@endpoint(path="/v1/rag/query", method="POST")
def handle_rag(payload: QueryPayload) -> Result[RagResponse, Status]:
    if payload.query.is_empty():
        return Err(Status.BadRequest("Query string cannot be empty."))
    docs = vector_db.search_collection("knowledge_base") |> vector_db.with_vector(generate_embedding(payload.query)) |> vector_db.limit(payload.top_k) |> vector_db.execute()
    answer = f"Answering: {payload.query}" |> llm.complete() |> std.text.trim()
    context = docs |> std.iter.map(lambda d: d.text) |> std.collect()
    return Ok(RagResponse(query=payload.query, retrieved_context=context, answer=answer))
`,
    args: [{"query":"How are tensor shapes checked?","top_k":2}],
  },
];

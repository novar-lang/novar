// Programs offered in the playground. `run` is the function started by default.
export const EXAMPLES = [
  {
    name: "Hello, Novar",
    run: "main",
    source: `fn main() -> i32 {
    std::io::println("hello, world");
    let name = "Novar";
    std::io::println(f"{name} has {name.len()} letters");
    0
}
`,
  },
  {
    name: "Primes",
    run: "main",
    source: `fn is_prime(n: i64) -> bool {
    if n < 2 { return false; }
    let mut d = 2;
    while d * d <= n {
        if n % d == 0 { return false; }
        d += 1;
    }
    true
}

fn main() -> i32 {
    let limit: i64 = 100;
    let mut count = 0;
    for n in 0..limit {
        if is_prime(n) { count += 1; }
    }
    std::io::println(f"primes below {limit}: {count}");
    count
}
`,
  },
  {
    name: "Sieve of Eratosthenes",
    run: "main",
    source: `fn main() -> i32 {
    let mut is_prime = [true, true, true, true, true, true, true, true, true, true,
                        true, true, true, true, true, true, true, true, true, true,
                        true, true, true, true, true, true, true, true, true, true];
    let n = is_prime.len();
    is_prime[0] = false;
    is_prime[1] = false;

    let mut p = 2;
    while p * p < n {
        if is_prime[p] {
            let mut multiple = p * p;
            while multiple < n {
                is_prime[multiple] = false;
                multiple += p;
            }
        }
        p += 1;
    }

    let mut line = "";
    let mut count = 0;
    for i in 0..n {
        if is_prime[i] {
            line = line + f"{i} ";
            count += 1;
        }
    }
    std::io::println(line);
    count
}
`,
  },
  {
    name: "Tensors with checked shapes",
    run: "forward",
    source: `// Shapes are part of the type: change [3, 2] to [4, 2] and the compiler objects.
fn forward(x: Tensor<f32, [2, 3]>, w: Tensor<f32, [3, 2]>) -> Tensor<f32, [2, 2]> {
    x |> std::tensor::matmul(w) |> std::tensor::relu()
}
`,
    args: [[[1, -2, 3], [4, 5, -6]], [[1, 0], [0, 1], [1, 1]]],
  },
  {
    name: "Pipelines and structs",
    run: "summarize",
    source: `struct Review { stars: i64, text: String }

fn clean(s: String) -> String {
    s |> std::text::clean_whitespace() |> std::text::trim() |> std::text::lowercase()
}

fn summarize(reviews: Vec<Review>) -> String {
    let texts = reviews |> std::iter::map(|r| clean(r.text));
    let mut stars = 0;
    for r in reviews { stars += r.stars; }
    f"{reviews.len()} reviews, {stars} stars: " + (texts |> std::string::join(" | "))
}
`,
    args: [[{ stars: 5, text: "  Fast   AND reliable " }, { stars: 4, text: "Great docs" }]],
  },
  {
    name: "RAG endpoint (offline stand-ins)",
    run: "handle_rag",
    source: `import std::ai::embeddings::OllamaEmbedder;
import std::ai::llm::OpenAIClient;
import std::db::pgvector::PgVectorClient;
import std::http::Status;

let embedder = OllamaEmbedder::new("nomic-embed-text");
let vector_db = PgVectorClient::connect("postgresql://localhost:5432/vectors");
let llm = OpenAIClient::new("gpt-4o-mini");

struct QueryPayload { query: String, top_k: i32 }
struct RagResponse { query: String, retrieved_context: Vector, answer: String }

fn generate_embedding(text: String) -> Tensor<f32, [768]> {
    text |> std::text::clean_whitespace() |> std::text::truncate(512) |> embedder.embed()
}

@endpoint(path = "/v1/rag/query", method = "POST")
pub fn handle_rag(payload: QueryPayload) -> Result<RagResponse, Status> {
    if payload.query.is_empty() {
        return Err(Status::BadRequest("Query string cannot be empty."));
    }
    let docs = vector_db.search_collection("knowledge_base")
        |> vector_db.with_vector(generate_embedding(payload.query))
        |> vector_db.limit(payload.top_k)
        |> vector_db.execute();
    let answer = f"Answering: {payload.query}" |> llm.complete() |> std::text::trim();
    let context = docs |> std::iter::map(|d| d.text) |> std::collect();
    return Ok(RagResponse { query: payload.query, retrieved_context: context, answer: answer });
}
`,
    args: [{ query: "How are tensor shapes checked?", top_k: 2 }],
  },
];

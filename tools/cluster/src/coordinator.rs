//! The coordinator side: spreads `std.dist.map` over a set of workers.
//!
//! Each worker is driven by as many threads as it has CPUs; threads pull items from one shared
//! queue, so fast workers take more work. A task that fails because a worker is unreachable (or
//! lost its program) goes back on the queue for another worker; a call that fails because the
//! Telvax code itself failed (division by zero, bad data) aborts the whole map, as a local call would.

use std::collections::VecDeque;
use std::sync::Mutex;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::time::Duration;

use serde_json::{Value, json};
use telvax_modules::Image;
use telvax_runtime::Executor;

/// How many times one item may be tried on workers that fail before the map gives up.
const MAX_ATTEMPTS: u32 = 4;

#[derive(Debug)]
pub struct ClusterExecutor {
    workers: Vec<String>,
    image: Value,
    token: Option<String>,
}

#[derive(Debug)]
enum Failure {
    /// The worker is unreachable / lost state: try the task elsewhere.
    Worker(String),
    /// The program failed: no other worker would do better.
    Program(String),
}

fn agent() -> ureq::Agent {
    ureq::Agent::config_builder()
        .timeout_global(Some(Duration::from_secs(300)))
        .timeout_connect(Some(Duration::from_secs(5)))
        .http_status_as_error(false)
        .build()
        .into()
}

/// Accepts `host:port` or a full URL.
fn base_url(worker: &str) -> String {
    let w = worker.trim().trim_end_matches('/');
    if w.starts_with("http://") || w.starts_with("https://") {
        w.to_string()
    } else {
        format!("http://{w}")
    }
}

impl ClusterExecutor {
    pub fn new(
        workers: Vec<String>,
        image: &Image,
        token: Option<String>,
    ) -> Result<ClusterExecutor, String> {
        if workers.is_empty() {
            return Err("no workers given".into());
        }
        Ok(ClusterExecutor {
            workers: workers.iter().map(|w| base_url(w)).collect(),
            image: serde_json::to_value(image).map_err(|e| e.to_string())?,
            token,
        })
    }

    fn post(&self, agent: &ureq::Agent, url: &str, body: &Value) -> Result<(u16, Value), Failure> {
        let mut req = agent.post(url);
        if let Some(t) = &self.token {
            req = req.header("Authorization", &format!("Bearer {t}"));
        }
        let mut resp = req
            .send_json(body)
            .map_err(|e| Failure::Worker(format!("{url}: {e}")))?;
        let status = resp.status().as_u16();
        let reply: Value = resp.body_mut().read_json().unwrap_or(Value::Null);
        Ok((status, reply))
    }

    /// Sends the program to `worker` and returns its id there.
    fn load(&self, agent: &ureq::Agent, worker: &str) -> Result<String, Failure> {
        let (status, reply) = self.post(agent, &format!("{worker}/load"), &self.image)?;
        match status {
            200 => reply["program"]
                .as_str()
                .map(str::to_string)
                .ok_or_else(|| Failure::Worker(format!("{worker}: no program id in the reply"))),
            401 => Err(Failure::Program(format!("{worker} rejected the token"))),
            422 => Err(Failure::Program(format!(
                "{worker}: {}",
                reply["error"].as_str().unwrap_or("the program was refused")
            ))),
            code => Err(Failure::Worker(format!(
                "{worker} answered HTTP {code} to /load"
            ))),
        }
    }

    fn call(
        &self,
        agent: &ureq::Agent,
        worker: &str,
        program: &str,
        function: &str,
        item: &Value,
    ) -> Result<Value, Failure> {
        let body = json!({ "program": program, "function": function, "args": [item] });
        let (status, reply) = self.post(agent, &format!("{worker}/call"), &body)?;
        match status {
            200 => Ok(reply["result"].clone()),
            401 => Err(Failure::Program(format!("{worker} rejected the token"))),
            422 => Err(Failure::Program(
                reply["error"]
                    .as_str()
                    .unwrap_or("the call failed")
                    .to_string(),
            )),
            404 => Err(Failure::Worker(format!(
                "{worker} no longer has the program"
            ))),
            code => Err(Failure::Worker(format!("{worker} answered HTTP {code}"))),
        }
    }

    fn cpus(&self, agent: &ureq::Agent, worker: &str) -> usize {
        agent
            .get(&format!("{worker}/health"))
            .call()
            .ok()
            .and_then(|mut r| r.body_mut().read_json::<Value>().ok())
            .and_then(|v| v["cpus"].as_u64())
            .map_or(1, |n| n.clamp(1, 64) as usize)
    }
}

struct Task {
    index: usize,
    item: Value,
    attempts: u32,
}

impl Executor for ClusterExecutor {
    fn map(&self, function: &str, items: Vec<Value>) -> Result<Vec<Value>, String> {
        let total = items.len();
        let queue: Mutex<VecDeque<Task>> = Mutex::new(
            items
                .into_iter()
                .enumerate()
                .map(|(index, item)| Task {
                    index,
                    item,
                    attempts: 0,
                })
                .collect(),
        );
        let results: Mutex<Vec<Option<Value>>> = Mutex::new(vec![None; total]);
        let done = AtomicUsize::new(0);
        let abort: Mutex<Option<String>> = Mutex::new(None);
        let stop = AtomicBool::new(false);
        let alive = AtomicUsize::new(0);
        let notes: Mutex<Vec<String>> = Mutex::new(Vec::new());

        std::thread::scope(|scope| {
            for worker in &self.workers {
                let agent = agent();
                let slots = self.cpus(&agent, worker);
                for _ in 0..slots {
                    alive.fetch_add(1, Ordering::SeqCst);
                    let (agent, queue, results, done, abort, stop, alive, notes) = (
                        agent.clone(),
                        &queue,
                        &results,
                        &done,
                        &abort,
                        &stop,
                        &alive,
                        &notes,
                    );
                    scope.spawn(move || {
                        let fail = |msg: String| {
                            let mut a = abort.lock().unwrap_or_else(|e| e.into_inner());
                            a.get_or_insert(msg);
                            stop.store(true, Ordering::SeqCst);
                        };
                        let note = |msg: String| notes.lock().unwrap_or_else(|e| e.into_inner()).push(msg);
                        let mut program = match self.load(&agent, worker) {
                            Ok(p) => p,
                            Err(Failure::Program(m)) => return fail(m),
                            Err(Failure::Worker(m)) => {
                                note(m);
                                alive.fetch_sub(1, Ordering::SeqCst);
                                return;
                            }
                        };
                        let mut consecutive_failures = 0;
                        loop {
                            if stop.load(Ordering::SeqCst) || done.load(Ordering::SeqCst) == total {
                                break;
                            }
                            let task = queue.lock().unwrap_or_else(|e| e.into_inner()).pop_front();
                            let Some(mut task) = task else {
                                // Other workers still hold tasks that may bounce back; stay available.
                                std::thread::sleep(Duration::from_millis(5));
                                continue;
                            };
                            match self.call(&agent, worker, &program, function, &task.item) {
                                Ok(v) => {
                                    consecutive_failures = 0;
                                    results.lock().unwrap_or_else(|e| e.into_inner())[task.index] = Some(v);
                                    done.fetch_add(1, Ordering::SeqCst);
                                }
                                Err(Failure::Program(m)) => return fail(m),
                                Err(Failure::Worker(m)) => {
                                    note(m);
                                    task.attempts += 1;
                                    if task.attempts >= MAX_ATTEMPTS {
                                        return fail(format!("item {} failed on {MAX_ATTEMPTS} workers; last problem: {}", task.index, notes.lock().unwrap_or_else(|e| e.into_inner()).last().cloned().unwrap_or_default()));
                                    }
                                    queue.lock().unwrap_or_else(|e| e.into_inner()).push_back(task);
                                    consecutive_failures += 1;
                                    // A worker that lost the program gets it again; one that keeps failing is dropped.
                                    if let Ok(p) = self.load(&agent, worker) {
                                        program = p;
                                    }
                                    if consecutive_failures >= 3 {
                                        alive.fetch_sub(1, Ordering::SeqCst);
                                        return;
                                    }
                                }
                            }
                        }
                    });
                }
            }
            // If every worker thread has dropped out while work remains, stop waiting.
            scope.spawn(|| {
                loop {
                    if stop.load(Ordering::SeqCst) || done.load(Ordering::SeqCst) == total {
                        break;
                    }
                    if alive.load(Ordering::SeqCst) == 0 {
                        let mut a = abort.lock().unwrap_or_else(|e| e.into_inner());
                        a.get_or_insert_with(|| {
                            format!(
                                "no worker is available ({})",
                                notes
                                    .lock()
                                    .unwrap_or_else(|e| e.into_inner())
                                    .last()
                                    .cloned()
                                    .unwrap_or_else(|| "all unreachable".into())
                            )
                        });
                        stop.store(true, Ordering::SeqCst);
                        break;
                    }
                    std::thread::sleep(Duration::from_millis(10));
                }
            });
        });

        if let Some(msg) = abort.into_inner().unwrap_or_else(|e| e.into_inner()) {
            return Err(msg);
        }
        results
            .into_inner()
            .unwrap_or_else(|e| e.into_inner())
            .into_iter()
            .map(|r| r.ok_or_else(|| "a result is missing".to_string()))
            .collect()
    }
}

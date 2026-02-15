use std::collections::HashMap;
use std::f64::consts::PI;
use std::fs;
use std::io::{self, Write};

const N_EMBD: usize = 16;
const N_HEAD: usize = 4;
const N_LAYER: usize = 1;
const BLOCK_SIZE: usize = 16;
const NUM_STEPS: usize = 1000;
const HEAD_DIM: usize = N_EMBD / N_HEAD;

struct Rng {
    state: u64,
}

impl Rng {
    fn new(seed: u64) -> Self {
        let state = if seed == 0 { 0x9E3779B97F4A7C15 } else { seed };
        Self { state }
    }

    fn next_u64(&mut self) -> u64 {
        let mut x = self.state;
        x ^= x >> 12;
        x ^= x << 25;
        x ^= x >> 27;
        self.state = x;
        x.wrapping_mul(0x2545_F491_4F6C_DD1D)
    }

    fn next_f64(&mut self) -> f64 {
        let r = self.next_u64() >> 11;
        r as f64 * (1.0 / 9007199254740992.0)
    }

    fn gauss(&mut self, mean: f64, std: f64) -> f64 {
        let u1 = (1.0 - self.next_f64()).max(f64::MIN_POSITIVE);
        let u2 = self.next_f64();
        let z = (-2.0 * u1.ln()).sqrt() * (2.0 * PI * u2).cos();
        mean + std * z
    }

    fn shuffle<T>(&mut self, arr: &mut [T]) {
        if arr.len() <= 1 {
            return;
        }
        for i in (1..arr.len()).rev() {
            let j = (self.next_u64() % ((i + 1) as u64)) as usize;
            arr.swap(i, j);
        }
    }

    fn choices(&mut self, weights: &[f64]) -> usize {
        assert!(!weights.is_empty(), "choices() received empty weights");
        let total: f64 = weights.iter().sum();
        let mut r = self.next_f64() * total;
        for (i, w) in weights.iter().enumerate() {
            r -= *w;
            if r <= 0.0 {
                return i;
            }
        }
        weights.len() - 1
    }
}

struct Tape {
    data: Vec<f64>,
    grad: Vec<f64>,
    edge_start: Vec<usize>,
    edge_count: Vec<usize>,
    visit_id: Vec<u32>,
    edge_to: Vec<usize>,
    edge_w: Vec<f64>,
    node_size: usize,
    edge_size: usize,
    visit_counter: u32,
}

impl Tape {
    fn new() -> Self {
        Self {
            data: Vec::new(),
            grad: Vec::new(),
            edge_start: Vec::new(),
            edge_count: Vec::new(),
            visit_id: Vec::new(),
            edge_to: Vec::new(),
            edge_w: Vec::new(),
            node_size: 0,
            edge_size: 0,
            visit_counter: 0,
        }
    }

    fn val(&mut self, data: f64) -> usize {
        let idx = self.node_size;
        self.node_size += 1;
        if idx < self.data.len() {
            self.data[idx] = data;
            self.grad[idx] = 0.0;
            self.edge_start[idx] = self.edge_size;
            self.edge_count[idx] = 0;
            self.visit_id[idx] = 0;
        } else {
            self.data.push(data);
            self.grad.push(0.0);
            self.edge_start.push(self.edge_size);
            self.edge_count.push(0);
            self.visit_id.push(0);
        }
        idx
    }

    fn add_child(&mut self, parent: usize, child: usize, weight: f64) {
        if self.edge_size < self.edge_to.len() {
            self.edge_to[self.edge_size] = child;
            self.edge_w[self.edge_size] = weight;
        } else {
            self.edge_to.push(child);
            self.edge_w.push(weight);
        }
        self.edge_size += 1;
        self.edge_count[parent] += 1;
    }

    fn reset(&mut self, base_nodes: usize) {
        self.node_size = base_nodes;
        self.edge_size = 0;
        for i in 0..base_nodes {
            self.grad[i] = 0.0;
            self.edge_start[i] = 0;
            self.edge_count[i] = 0;
            self.visit_id[i] = 0;
        }
    }
}

fn add(tape: &mut Tape, lhs: usize, rhs: usize) -> usize {
    let out = tape.val(tape.data[lhs] + tape.data[rhs]);
    tape.add_child(out, lhs, 1.0);
    tape.add_child(out, rhs, 1.0);
    out
}

fn mul(tape: &mut Tape, lhs: usize, rhs: usize) -> usize {
    let out = tape.val(tape.data[lhs] * tape.data[rhs]);
    tape.add_child(out, lhs, tape.data[rhs]);
    tape.add_child(out, rhs, tape.data[lhs]);
    out
}

fn pow_op(tape: &mut Tape, base: usize, exponent: f64) -> usize {
    let base_data = tape.data[base];
    let out = tape.val(base_data.powf(exponent));
    tape.add_child(out, base, exponent * base_data.powf(exponent - 1.0));
    out
}

fn relu_op(tape: &mut Tape, v: usize) -> usize {
    let d = tape.data[v];
    let out = tape.val(d.max(0.0));
    tape.add_child(out, v, if d > 0.0 { 1.0 } else { 0.0 });
    out
}

fn log_op(tape: &mut Tape, v: usize) -> usize {
    let out = tape.val(tape.data[v].ln());
    tape.add_child(out, v, 1.0 / tape.data[v]);
    out
}

fn exp_op(tape: &mut Tape, v: usize) -> usize {
    let d = tape.data[v].exp();
    let out = tape.val(d);
    tape.add_child(out, v, d);
    out
}

fn sub(tape: &mut Tape, lhs: usize, rhs: usize) -> usize {
    let out = tape.val(tape.data[lhs] - tape.data[rhs]);
    tape.add_child(out, lhs, 1.0);
    tape.add_child(out, rhs, -1.0);
    out
}

fn div(tape: &mut Tape, lhs: usize, rhs: usize) -> usize {
    let inv = pow_op(tape, rhs, -1.0);
    mul(tape, lhs, inv)
}

fn div_f(tape: &mut Tape, lhs: usize, rhs: f64) -> usize {
    let inv = tape.val(1.0 / rhs);
    mul(tape, lhs, inv)
}

fn build_topo(tape: &mut Tape, v: usize, topo: &mut Vec<usize>, visit_id: u32) {
    if tape.visit_id[v] == visit_id {
        return;
    }
    tape.visit_id[v] = visit_id;
    let start = tape.edge_start[v];
    let count = tape.edge_count[v];
    for i in 0..count {
        build_topo(tape, tape.edge_to[start + i], topo, visit_id);
    }
    topo.push(v);
}

fn backward(tape: &mut Tape, root: usize) {
    tape.visit_counter = tape.visit_counter.wrapping_add(1);
    if tape.visit_counter == 0 {
        tape.visit_counter = 1;
    }
    let visit_id = tape.visit_counter;
    let mut topo = Vec::new();
    build_topo(tape, root, &mut topo, visit_id);

    tape.grad[root] = 1.0;
    for &v in topo.iter().rev() {
        let v_grad = tape.grad[v];
        let start = tape.edge_start[v];
        let count = tape.edge_count[v];
        for i in 0..count {
            let edge_idx = start + i;
            let child = tape.edge_to[edge_idx];
            tape.grad[child] += tape.edge_w[edge_idx] * v_grad;
        }
    }
}

struct Matrix {
    rows: usize,
    cols: usize,
    data: Vec<usize>,
}

impl Matrix {
    fn new(rows: usize, cols: usize) -> Self {
        Self {
            rows,
            cols,
            data: vec![usize::MAX; rows * cols],
        }
    }
}

struct Layer {
    attn_wq: Matrix,
    attn_wk: Matrix,
    attn_wv: Matrix,
    attn_wo: Matrix,
    mlp_fc1: Matrix,
    mlp_fc2: Matrix,
}

struct StateDict {
    wte: Matrix,
    wpe: Matrix,
    lm_head: Matrix,
    layers: Vec<Layer>,
}

fn matrix(rows: usize, cols: usize, rng: &mut Rng, tape: &mut Tape, params: &mut Vec<usize>) -> Matrix {
    let mut m = Matrix::new(rows, cols);
    for i in 0..rows {
        for j in 0..cols {
            let p = tape.val(rng.gauss(0.0, 0.08));
            m.data[i * cols + j] = p;
            params.push(p);
        }
    }
    m
}

fn linear(tape: &mut Tape, x: &[usize], w: &Matrix) -> Vec<usize> {
    assert_eq!(x.len(), w.cols, "linear shape mismatch");
    let mut out = Vec::with_capacity(w.rows);
    for i in 0..w.rows {
        let mut sum_v = tape.val(0.0);
        let row_start = i * w.cols;
        for j in 0..w.cols {
            let prod = mul(tape, w.data[row_start + j], x[j]);
            sum_v = add(tape, sum_v, prod);
        }
        out.push(sum_v);
    }
    out
}

fn softmax(tape: &mut Tape, logits: &[usize]) -> Vec<usize> {
    assert!(!logits.is_empty(), "softmax() received empty logits");
    let mut max_val = f64::NEG_INFINITY;
    for &idx in logits {
        max_val = max_val.max(tape.data[idx]);
    }

    let max_node = tape.val(max_val);
    let mut exps = Vec::with_capacity(logits.len());
    let mut total = tape.val(0.0);
    for &idx in logits {
        let diff = sub(tape, idx, max_node);
        let e = exp_op(tape, diff);
        exps.push(e);
        total = add(tape, total, e);
    }

    exps.into_iter().map(|e| div(tape, e, total)).collect()
}

fn rmsnorm(tape: &mut Tape, x: &[usize]) -> Vec<usize> {
    let mut ss = tape.val(0.0);
    for &xi in x {
        let sq = mul(tape, xi, xi);
        ss = add(tape, ss, sq);
    }
    let inv_n = tape.val(1.0 / x.len() as f64);
    let mean = mul(tape, ss, inv_n);
    let eps = tape.val(1e-5);
    let mean_eps = add(tape, mean, eps);
    let scale_v = pow_op(tape, mean_eps, -0.5);
    x.iter().map(|&xi| mul(tape, xi, scale_v)).collect()
}

type Sequence = Vec<Vec<usize>>;
type KVCache = Vec<Sequence>;

fn gpt(
    tape: &mut Tape,
    token_id: usize,
    pos_id: usize,
    state_dict: &StateDict,
    keys: &mut KVCache,
    values: &mut KVCache,
) -> Vec<usize> {
    let tok_start = token_id * N_EMBD;
    let pos_start = pos_id * N_EMBD;
    let mut x: Vec<usize> = (0..N_EMBD)
        .map(|i| add(tape, state_dict.wte.data[tok_start + i], state_dict.wpe.data[pos_start + i]))
        .collect();
    x = rmsnorm(tape, &x);

    for (li, layer) in state_dict.layers.iter().enumerate() {
        let x_residual = x.clone();
        x = rmsnorm(tape, &x);

        let q = linear(tape, &x, &layer.attn_wq);
        let k = linear(tape, &x, &layer.attn_wk);
        let v = linear(tape, &x, &layer.attn_wv);
        keys[li].push(k.clone());
        values[li].push(v.clone());

        let mut x_attn = Vec::with_capacity(N_EMBD);
        for h in 0..N_HEAD {
            let hs = h * HEAD_DIM;
            let attn_logits: Vec<usize> = keys[li]
                .iter()
                .map(|kt| {
                    let mut dot = tape.val(0.0);
                    for j in 0..HEAD_DIM {
                        let qk = mul(tape, q[hs + j], kt[hs + j]);
                        dot = add(tape, dot, qk);
                    }
                    div_f(tape, dot, (HEAD_DIM as f64).sqrt())
                })
                .collect();

            let attn_weights = softmax(tape, &attn_logits);
            for j in 0..HEAD_DIM {
                let mut out_j = tape.val(0.0);
                for (t, vt) in values[li].iter().enumerate() {
                    let av = mul(tape, attn_weights[t], vt[hs + j]);
                    out_j = add(tape, out_j, av);
                }
                x_attn.push(out_j);
            }
        }

        x = linear(tape, &x_attn, &layer.attn_wo)
            .iter()
            .zip(x_residual.iter())
            .map(|(&a, &b)| add(tape, a, b))
            .collect();

        let x_residual = x.clone();
        x = rmsnorm(tape, &x);
        x = linear(tape, &x, &layer.mlp_fc1);
        for xi in &mut x {
            *xi = relu_op(tape, *xi);
        }
        x = linear(tape, &x, &layer.mlp_fc2)
            .iter()
            .zip(x_residual.iter())
            .map(|(&a, &b)| add(tape, a, b))
            .collect();
    }

    linear(tape, &x, &state_dict.lm_head)
}

fn main() {
    assert_eq!(N_EMBD % N_HEAD, 0, "N_EMBD must be divisible by N_HEAD");
    assert_eq!(HEAD_DIM, N_EMBD / N_HEAD, "HEAD_DIM must match N_EMBD / N_HEAD");

    let mut rng = Rng::new(42);
    let content = fs::read_to_string("input.txt").expect("failed to read input.txt");
    let mut docs: Vec<String> = content
        .lines()
        .map(str::trim)
        .filter(|l| !l.is_empty())
        .map(str::to_string)
        .collect();
    if docs.is_empty() {
        eprintln!("input.txt has no non-empty lines");
        return;
    }
    rng.shuffle(&mut docs);
    println!("num docs: {}", docs.len());

    let mut uchars: Vec<char> = docs.iter().flat_map(|d| d.chars()).collect();
    uchars.sort_unstable();
    uchars.dedup();

    let bos = uchars.len();
    let vocab_size = uchars.len() + 1;
    println!("vocab size: {}", vocab_size);

    let mut stoi = HashMap::new();
    for (i, ch) in uchars.iter().copied().enumerate() {
        stoi.insert(ch, i);
    }

    let mut tape = Tape::new();
    let mut params = Vec::new();

    let mut layers = Vec::new();
    for _ in 0..N_LAYER {
        layers.push(Layer {
            attn_wq: matrix(N_EMBD, N_EMBD, &mut rng, &mut tape, &mut params),
            attn_wk: matrix(N_EMBD, N_EMBD, &mut rng, &mut tape, &mut params),
            attn_wv: matrix(N_EMBD, N_EMBD, &mut rng, &mut tape, &mut params),
            attn_wo: matrix(N_EMBD, N_EMBD, &mut rng, &mut tape, &mut params),
            mlp_fc1: matrix(4 * N_EMBD, N_EMBD, &mut rng, &mut tape, &mut params),
            mlp_fc2: matrix(N_EMBD, 4 * N_EMBD, &mut rng, &mut tape, &mut params),
        });
    }

    let state_dict = StateDict {
        wte: matrix(vocab_size, N_EMBD, &mut rng, &mut tape, &mut params),
        wpe: matrix(BLOCK_SIZE, N_EMBD, &mut rng, &mut tape, &mut params),
        lm_head: matrix(vocab_size, N_EMBD, &mut rng, &mut tape, &mut params),
        layers,
    };

    let num_params = params.len();
    println!("num params: {}", num_params);

    let learning_rate = 0.01;
    let beta1 = 0.85;
    let beta2 = 0.99;
    let eps_adam = 1e-8;
    let mut m = vec![0.0; num_params];
    let mut v = vec![0.0; num_params];

    for step in 0..NUM_STEPS {
        tape.reset(num_params);
        let doc = &docs[step % docs.len()];

        let mut tokens = vec![bos];
        for ch in doc.chars() {
            tokens.push(*stoi.get(&ch).expect("unknown character in doc"));
        }
        tokens.push(bos);

        let n = BLOCK_SIZE.min(tokens.len() - 1);
        let mut keys: KVCache = vec![Vec::new(); N_LAYER];
        let mut values: KVCache = vec![Vec::new(); N_LAYER];

        let mut losses = Vec::with_capacity(n);
        for pos_id in 0..n {
            let token_id = tokens[pos_id];
            let target_id = tokens[pos_id + 1];
            let logits = gpt(
                &mut tape,
                token_id,
                pos_id,
                &state_dict,
                &mut keys,
                &mut values,
            );
            let probs = softmax(&mut tape, &logits);
            let log_p = log_op(&mut tape, probs[target_id]);
            let neg_one = tape.val(-1.0);
            let loss_t = mul(&mut tape, log_p, neg_one);
            losses.push(loss_t);
        }

        let mut loss = tape.val(0.0);
        for &loss_t in &losses {
            loss = add(&mut tape, loss, loss_t);
        }
        let inv_n = tape.val(1.0 / n as f64);
        loss = mul(&mut tape, loss, inv_n);
        let loss_data = tape.data[loss];

        backward(&mut tape, loss);

        let lr_t = learning_rate * (1.0 - step as f64 / NUM_STEPS as f64);
        let t = (step + 1) as i32;
        for (i, &p) in params.iter().enumerate() {
            let g = tape.grad[p];
            m[i] = beta1 * m[i] + (1.0 - beta1) * g;
            v[i] = beta2 * v[i] + (1.0 - beta2) * g * g;
            let m_hat = m[i] / (1.0 - beta1.powi(t));
            let v_hat = v[i] / (1.0 - beta2.powi(t));
            tape.data[p] -= lr_t * m_hat / (v_hat.sqrt() + eps_adam);
            tape.grad[p] = 0.0;
        }

        print!(
            "\rstep {:4} / {:4} | loss {:.4}",
            step + 1,
            NUM_STEPS,
            loss_data
        );
        io::stdout().flush().expect("failed to flush stdout");
        if step + 1 == NUM_STEPS {
            println!();
        }
    }

    println!("\n--- inference (new, hallucinated names) ---");
    let temperature = 0.5;
    for sample_idx in 0..20 {
        tape.reset(num_params);
        let mut keys: KVCache = vec![Vec::new(); N_LAYER];
        let mut values: KVCache = vec![Vec::new(); N_LAYER];
        let mut token_id = bos;
        let mut sample = String::new();

        for pos_id in 0..BLOCK_SIZE {
            let logits = gpt(
                &mut tape,
                token_id,
                pos_id,
                &state_dict,
                &mut keys,
                &mut values,
            );
            let scaled_logits: Vec<usize> = logits
                .iter()
                .map(|&l| div_f(&mut tape, l, temperature))
                .collect();
            let probs = softmax(&mut tape, &scaled_logits);
            let weights: Vec<f64> = probs.iter().map(|&i| tape.data[i]).collect();
            token_id = rng.choices(&weights);

            if token_id == bos {
                break;
            }
            sample.push(uchars[token_id]);
        }

        println!("sample {:2}: {}", sample_idx + 1, sample);
    }
}

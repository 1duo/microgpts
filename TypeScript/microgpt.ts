import { readFileSync } from "node:fs";

type Matrix = Value[][];
type Layer = {
  attn_wq: Matrix;
  attn_wk: Matrix;
  attn_wv: Matrix;
  attn_wo: Matrix;
  mlp_fc1: Matrix;
  mlp_fc2: Matrix;
};
type StateDict = {
  wte: Matrix;
  wpe: Matrix;
  lm_head: Matrix;
  layers: Layer[];
};

const n_embd = 16;
const n_head = 4;
const n_layer = 1;
const block_size = 16;
const num_steps = 1000;
const head_dim = n_embd / n_head;

class RNG {
  private state: number;
  private spare: number | null = null;

  constructor(seed: number) {
    this.state = seed | 0;
    if (this.state === 0) this.state = 0x6d2b79f5;
  }

  private nextU32(): number {
    let x = this.state | 0;
    x ^= x << 13;
    x ^= x >>> 17;
    x ^= x << 5;
    this.state = x | 0;
    return this.state >>> 0;
  }

  random(): number {
    return this.nextU32() / 4294967296;
  }

  gauss(mean: number, std: number): number {
    if (this.spare !== null) {
      const z = this.spare;
      this.spare = null;
      return mean + std * z;
    }
    const u1 = Math.max(this.random(), Number.MIN_VALUE);
    const u2 = this.random();
    const mag = Math.sqrt(-2.0 * Math.log(u1));
    const z0 = mag * Math.cos(2.0 * Math.PI * u2);
    const z1 = mag * Math.sin(2.0 * Math.PI * u2);
    this.spare = z1;
    return mean + std * z0;
  }

  shuffle<T>(arr: T[]): void {
    for (let i = arr.length - 1; i > 0; i--) {
      const j = Math.floor(this.random() * (i + 1));
      [arr[i], arr[j]] = [arr[j], arr[i]];
    }
  }

  choices(weights: number[]): number {
    let total = 0;
    for (const w of weights) total += w;
    let r = this.random() * total;
    for (let i = 0; i < weights.length; i++) {
      r -= weights[i];
      if (r <= 0) return i;
    }
    return weights.length - 1;
  }
}

class Value {
  data: number;
  grad = 0;
  private children: [Value | null, Value | null] = [null, null];
  private local_grads: [number, number] = [0, 0];
  private n_children = 0;

  constructor(data: number, c0?: Value, g0?: number, c1?: Value, g1?: number) {
    this.data = data;
    if (c0 !== undefined && g0 !== undefined) {
      this.children[0] = c0;
      this.local_grads[0] = g0;
      this.n_children = 1;
    }
    if (c1 !== undefined && g1 !== undefined) {
      this.children[1] = c1;
      this.local_grads[1] = g1;
      this.n_children = 2;
    }
  }

  add(other: Value | number): Value {
    const o = other instanceof Value ? other : new Value(other);
    return new Value(this.data + o.data, this, 1, o, 1);
  }

  mul(other: Value | number): Value {
    const o = other instanceof Value ? other : new Value(other);
    return new Value(this.data * o.data, this, o.data, o, this.data);
  }

  pow(other: number): Value {
    return new Value(this.data ** other, this, other * this.data ** (other - 1));
  }

  log(): Value {
    return new Value(Math.log(this.data), this, 1 / this.data);
  }

  exp(): Value {
    const e = Math.exp(this.data);
    return new Value(e, this, e);
  }

  relu(): Value {
    return new Value(Math.max(0, this.data), this, this.data > 0 ? 1 : 0);
  }

  neg(): Value {
    return this.mul(-1);
  }

  sub(other: Value | number): Value {
    return this.add(other instanceof Value ? other.neg() : -other);
  }

  div(other: Value | number): Value {
    return other instanceof Value ? this.mul(other.pow(-1)) : this.mul(other ** -1);
  }

  backward(): void {
    const topo: Value[] = [];
    const visited = new Set<Value>();

    const buildTopo = (v: Value): void => {
      if (visited.has(v)) return;
      visited.add(v);
      for (let i = 0; i < v.n_children; i++) {
        buildTopo(v.children[i]!);
      }
      topo.push(v);
    };

    buildTopo(this);
    this.grad = 1;
    for (let i = topo.length - 1; i >= 0; i--) {
      const v = topo[i];
      for (let j = 0; j < v.n_children; j++) {
        v.children[j]!.grad += v.local_grads[j] * v.grad;
      }
    }
  }
}

const matrix = (nout: number, nin: number, rng: RNG, std = 0.08): Matrix =>
  Array.from({ length: nout }, () =>
    Array.from({ length: nin }, () => new Value(rng.gauss(0, std)))
  );

const linear = (x: Value[], w: Matrix): Value[] =>
  w.map((wo) => wo.reduce((acc, wi, i) => acc.add(wi.mul(x[i])), new Value(0)));

const softmax = (logits: Value[]): Value[] => {
  let maxVal = logits[0].data;
  for (let i = 1; i < logits.length; i++) {
    if (logits[i].data > maxVal) maxVal = logits[i].data;
  }
  const exps = logits.map((v) => v.sub(maxVal).exp());
  const total = exps.reduce((acc, e) => acc.add(e), new Value(0));
  return exps.map((e) => e.div(total));
};

const rmsnorm = (x: Value[]): Value[] => {
  const ms = x.reduce((acc, xi) => acc.add(xi.mul(xi)), new Value(0)).div(x.length);
  const scale = ms.add(1e-5).pow(-0.5);
  return x.map((xi) => xi.mul(scale));
};

const params = (state_dict: StateDict): Value[] => {
  const out: Value[] = [];
  const append = (m: Matrix): void => {
    for (const row of m) out.push(...row);
  };
  append(state_dict.wte);
  append(state_dict.wpe);
  append(state_dict.lm_head);
  for (const layer of state_dict.layers) {
    append(layer.attn_wq);
    append(layer.attn_wk);
    append(layer.attn_wv);
    append(layer.attn_wo);
    append(layer.mlp_fc1);
    append(layer.mlp_fc2);
  }
  return out;
};

const gpt = (
  token_id: number,
  pos_id: number,
  keys: Value[][][],
  values: Value[][][],
  state_dict: StateDict
): Value[] => {
  const tok_emb = state_dict.wte[token_id];
  const pos_emb = state_dict.wpe[pos_id];

  let x = tok_emb.map((t, i) => t.add(pos_emb[i]));
  x = rmsnorm(x);

  for (let li = 0; li < n_layer; li++) {
    const x_residual = x;
    x = rmsnorm(x);

    const layer = state_dict.layers[li];
    const q = linear(x, layer.attn_wq);
    const k = linear(x, layer.attn_wk);
    const v = linear(x, layer.attn_wv);

    keys[li].push(k);
    values[li].push(v);

    const x_attn: Value[] = [];
    for (let h = 0; h < n_head; h++) {
      const hs = h * head_dim;

      const attn_logits = keys[li].map((kt) => {
        const dot = Array.from({ length: head_dim }, (_, j) => q[hs + j].mul(kt[hs + j])).reduce(
          (acc, vj) => acc.add(vj),
          new Value(0)
        );
        return dot.div(Math.sqrt(head_dim));
      });

      const attn_weights = softmax(attn_logits);
      for (let j = 0; j < head_dim; j++) {
        const out_j = values[li].reduce(
          (acc, vt, t) => acc.add(attn_weights[t].mul(vt[hs + j])),
          new Value(0)
        );
        x_attn.push(out_j);
      }
    }

    x = linear(x_attn, layer.attn_wo).map((a, i) => a.add(x_residual[i]));

    const x_mlp_residual = x;
    x = rmsnorm(x);
    x = linear(x, layer.mlp_fc1).map((xi) => xi.relu());
    x = linear(x, layer.mlp_fc2).map((a, i) => a.add(x_mlp_residual[i]));
  }

  return linear(x, state_dict.lm_head);
};

const main = (): void => {
  const rng = new RNG(42);
  const docs = readFileSync("input.txt", "utf8")
    .split("\n")
    .map((l) => l.trim())
    .filter((l) => l.length > 0);
  rng.shuffle(docs);
  console.log(`num docs: ${docs.length}`);

  const uchars = Array.from(new Set(docs.join("").split("").sort()));
  const char_to_token = new Map<string, number>(uchars.map((ch, i) => [ch, i]));
  const BOS = uchars.length;
  const vocab_size = BOS + 1;
  console.log(`vocab size: ${vocab_size}`);

  const state_dict: StateDict = {
    wte: matrix(vocab_size, n_embd, rng),
    wpe: matrix(block_size, n_embd, rng),
    lm_head: matrix(vocab_size, n_embd, rng),
    layers: Array.from({ length: n_layer }, () => ({
      attn_wq: matrix(n_embd, n_embd, rng),
      attn_wk: matrix(n_embd, n_embd, rng),
      attn_wv: matrix(n_embd, n_embd, rng),
      attn_wo: matrix(n_embd, n_embd, rng),
      mlp_fc1: matrix(4 * n_embd, n_embd, rng),
      mlp_fc2: matrix(n_embd, 4 * n_embd, rng),
    })),
  };

  const all_params = params(state_dict);
  console.log(`num params: ${all_params.length}`);

  const learning_rate = 0.01;
  const beta1 = 0.85;
  const beta2 = 0.99;
  const eps_adam = 1e-8;
  const m = Array(all_params.length).fill(0);
  const v = Array(all_params.length).fill(0);

  for (let step = 0; step < num_steps; step++) {
    const doc = docs[step % docs.length];
    const tokens = [BOS, ...doc.split("").map((ch) => char_to_token.get(ch)!), BOS];

    const n = Math.min(block_size, tokens.length - 1);
    const keys: Value[][][] = Array.from({ length: n_layer }, () => []);
    const values: Value[][][] = Array.from({ length: n_layer }, () => []);

    const losses: Value[] = [];
    for (let pos_id = 0; pos_id < n; pos_id++) {
      const token_id = tokens[pos_id];
      const target_id = tokens[pos_id + 1];
      const logits = gpt(token_id, pos_id, keys, values, state_dict);
      const probs = softmax(logits);
      losses.push(probs[target_id].log().neg());
    }

    const loss = losses.reduce((acc, l) => acc.add(l), new Value(0)).mul(1 / n);
    loss.backward();

    const lr_t = learning_rate * (1 - step / num_steps);
    for (let i = 0; i < all_params.length; i++) {
      const p = all_params[i];
      m[i] = beta1 * m[i] + (1 - beta1) * p.grad;
      v[i] = beta2 * v[i] + (1 - beta2) * p.grad ** 2;
      const m_hat = m[i] / (1 - beta1 ** (step + 1));
      const v_hat = v[i] / (1 - beta2 ** (step + 1));
      p.data -= lr_t * m_hat / (Math.sqrt(v_hat) + eps_adam);
      p.grad = 0;
    }

    process.stdout.write(
      `\rstep ${String(step + 1).padStart(4)} / ${String(num_steps).padStart(4)} | loss ${loss.data.toFixed(4)}`
    );
    if (step + 1 === num_steps) process.stdout.write("\n");
  }

  const temperature = 0.5;
  console.log("\n--- inference (new, hallucinated names) ---");
  for (let sample_idx = 0; sample_idx < 20; sample_idx++) {
    const keys: Value[][][] = Array.from({ length: n_layer }, () => []);
    const values: Value[][][] = Array.from({ length: n_layer }, () => []);

    let token_id = BOS;
    const sample: string[] = [];

    for (let pos_id = 0; pos_id < block_size; pos_id++) {
      const logits = gpt(token_id, pos_id, keys, values, state_dict);
      const probs = softmax(logits.map((l) => l.div(temperature)));
      token_id = rng.choices(probs.map((p) => p.data));
      if (token_id === BOS) break;
      sample.push(uchars[token_id]);
    }

    console.log(`sample ${String(sample_idx + 1).padStart(2)}: ${sample.join("")}`);
  }
};

main();

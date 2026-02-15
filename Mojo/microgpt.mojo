from math import sqrt, exp, log, cos
from pathlib import Path


struct RNG:
    var state: Int

    fn __init__(inout self, seed: Int):
        self.state = seed if seed != 0 else 88172645463393265

    fn next_u64(inout self) -> Int:
        var x: Int = self.state
        x = x ^ (x << 13)
        x = x ^ (x >> 7)
        x = x ^ (x << 17)
        if x < 0:
            x = -x
        self.state = x
        return x

    fn random(inout self) -> Float64:
        return Float64(self.next_u64()) / 9223372036854775808.0

    fn gauss(inout self, mean: Float64, std: Float64) -> Float64:
        var u1: Float64 = self.random()
        if u1 < 1e-12:
            u1 = 1e-12
        var u2: Float64 = self.random()
        var z0: Float64 = sqrt(-2.0 * log(u1)) * cos(6.283185307179586 * u2)
        return mean + std * z0

    fn randint(inout self, n: Int) -> Int:
        if n <= 1:
            return 0
        return self.next_u64() % n


fn random_shuffle_docs(inout rng: RNG, inout docs: List[String]):
    var n: Int = len(docs)
    if n <= 1:
        return
    for i in range(n - 1, 0, -1):
        var j: Int = rng.randint(i + 1)
        var tmp: String = docs[i]
        docs[i] = docs[j]
        docs[j] = tmp


struct Tape:
    var data: List[Float64]
    var grad: List[Float64]
    var edge_start: List[Int]
    var edge_count: List[Int]
    var visit_id: List[Int]
    var edge_to: List[Int]
    var edge_w: List[Float64]
    var node_size: Int
    var edge_size: Int
    var visit_counter: Int

    fn __init__(inout self):
        self.data = List[Float64]()
        self.grad = List[Float64]()
        self.edge_start = List[Int]()
        self.edge_count = List[Int]()
        self.visit_id = List[Int]()
        self.edge_to = List[Int]()
        self.edge_w = List[Float64]()
        self.node_size = 0
        self.edge_size = 0
        self.visit_counter = 0

    fn val(inout self, data: Float64) -> Int:
        var idx: Int = self.node_size
        self.node_size += 1
        if idx < len(self.data):
            self.data[idx] = data
            self.grad[idx] = 0.0
            self.edge_start[idx] = self.edge_size
            self.edge_count[idx] = 0
            self.visit_id[idx] = 0
        else:
            self.data.append(data)
            self.grad.append(0.0)
            self.edge_start.append(self.edge_size)
            self.edge_count.append(0)
            self.visit_id.append(0)
        return idx

    fn add_child(inout self, parent: Int, child: Int, weight: Float64):
        if self.edge_size < len(self.edge_to):
            self.edge_to[self.edge_size] = child
            self.edge_w[self.edge_size] = weight
        else:
            self.edge_to.append(child)
            self.edge_w.append(weight)
        self.edge_size += 1
        self.edge_count[parent] += 1

    fn reset(inout self, base_nodes: Int):
        self.node_size = base_nodes
        self.edge_size = 0
        for i in range(base_nodes):
            self.grad[i] = 0.0
            self.edge_start[i] = 0
            self.edge_count[i] = 0
            self.visit_id[i] = 0


fn add(inout tape: Tape, lhs: Int, rhs: Int) -> Int:
    var out: Int = tape.val(tape.data[lhs] + tape.data[rhs])
    tape.add_child(out, lhs, 1.0)
    tape.add_child(out, rhs, 1.0)
    return out


fn mul(inout tape: Tape, lhs: Int, rhs: Int) -> Int:
    var out: Int = tape.val(tape.data[lhs] * tape.data[rhs])
    tape.add_child(out, lhs, tape.data[rhs])
    tape.add_child(out, rhs, tape.data[lhs])
    return out


fn pow_op(inout tape: Tape, base: Int, exponent: Float64) -> Int:
    var base_data: Float64 = tape.data[base]
    var out: Int = tape.val(base_data ** exponent)
    tape.add_child(out, base, exponent * (base_data ** (exponent - 1.0)))
    return out


fn relu_op(inout tape: Tape, v: Int) -> Int:
    var d: Float64 = tape.data[v]
    var out: Int = tape.val(d if d > 0 else 0.0)
    tape.add_child(out, v, 1.0 if d > 0 else 0.0)
    return out


fn log_op(inout tape: Tape, v: Int) -> Int:
    var out: Int = tape.val(log(tape.data[v]))
    tape.add_child(out, v, 1.0 / tape.data[v])
    return out


fn exp_op(inout tape: Tape, v: Int) -> Int:
    var d: Float64 = exp(tape.data[v])
    var out: Int = tape.val(d)
    tape.add_child(out, v, d)
    return out


fn sub(inout tape: Tape, lhs: Int, rhs: Int) -> Int:
    var out: Int = tape.val(tape.data[lhs] - tape.data[rhs])
    tape.add_child(out, lhs, 1.0)
    tape.add_child(out, rhs, -1.0)
    return out


fn div(inout tape: Tape, lhs: Int, rhs: Int) -> Int:
    return mul(tape, lhs, pow_op(tape, rhs, -1.0))


fn div_f(inout tape: Tape, lhs: Int, rhs: Float64) -> Int:
    return mul(tape, lhs, tape.val(1.0 / rhs))


fn random_choices(inout rng: RNG, tape: Tape, probs: List[Int]) -> Int:
    var total: Float64 = 0.0
    for i in range(len(probs)):
        total += tape.data[probs[i]]
    var r: Float64 = rng.random() * total
    for i in range(len(probs)):
        r -= tape.data[probs[i]]
        if r <= 0.0:
            return i
    return len(probs) - 1


fn build_topo(inout tape: Tape, v: Int, inout topo: List[Int], visit_id: Int):
    if tape.visit_id[v] == visit_id:
        return
    tape.visit_id[v] = visit_id
    var start: Int = tape.edge_start[v]
    var count: Int = tape.edge_count[v]
    for i in range(count):
        build_topo(tape, tape.edge_to[start + i], topo, visit_id)
    topo.append(v)


fn backward(inout tape: Tape, root: Int):
    tape.visit_counter += 1
    var visit_id: Int = tape.visit_counter
    var topo: List[Int] = List[Int]()
    build_topo(tape, root, topo, visit_id)

    tape.grad[root] = 1.0
    for i in range(len(topo) - 1, -1, -1):
        var v: Int = topo[i]
        var v_grad: Float64 = tape.grad[v]
        var start: Int = tape.edge_start[v]
        var count: Int = tape.edge_count[v]
        for j in range(count):
            var edge_idx: Int = start + j
            var child: Int = tape.edge_to[edge_idx]
            tape.grad[child] += tape.edge_w[edge_idx] * v_grad


struct Mat:
    var rows: Int
    var cols: Int
    var data: List[Int]

    fn __init__(inout self, rows: Int, cols: Int):
        self.rows = rows
        self.cols = cols
        self.data = List[Int]()
        for _ in range(rows * cols):
            self.data.append(-1)

    fn __copyinit__(inout self, existing: Self):
        self.rows = existing.rows
        self.cols = existing.cols
        self.data = List[Int]()
        for i in range(len(existing.data)):
            self.data.append(existing.data[i])

    fn set(inout self, r: Int, c: Int, v: Int):
        self.data[r * self.cols + c] = v


fn linear(inout tape: Tape, x: List[Int], w: Mat) -> List[Int]:
    var out: List[Int] = List[Int]()
    for i in range(w.rows):
        var sum_v: Int = tape.val(0.0)
        var row_start: Int = i * w.cols
        for j in range(w.cols):
            sum_v = add(tape, sum_v, mul(tape, w.data[row_start + j], x[j]))
        out.append(sum_v)
    return out


fn softmax(inout tape: Tape, logits: List[Int]) -> List[Int]:
    var max_val: Float64 = -1e9
    for i in range(len(logits)):
        var v: Float64 = tape.data[logits[i]]
        if v > max_val:
            max_val = v

    var max_node: Int = tape.val(max_val)
    var exps: List[Int] = List[Int]()
    var total: Int = tape.val(0.0)
    for i in range(len(logits)):
        var e: Int = exp_op(tape, sub(tape, logits[i], max_node))
        exps.append(e)
        total = add(tape, total, e)

    var out: List[Int] = List[Int]()
    for i in range(len(exps)):
        out.append(div(tape, exps[i], total))
    return out


fn rmsnorm(inout tape: Tape, x: List[Int]) -> List[Int]:
    var ss: Int = tape.val(0.0)
    for i in range(len(x)):
        var sq: Int = mul(tape, x[i], x[i])
        ss = add(tape, ss, sq)
    var mean: Int = mul(tape, ss, tape.val(1.0 / Float64(len(x))))
    var scale_v: Int = pow_op(tape, add(tape, mean, tape.val(1e-5)), -0.5)

    var out: List[Int] = List[Int]()
    for i in range(len(x)):
        out.append(mul(tape, x[i], scale_v))
    return out


fn matrix(rows: Int, cols: Int, inout rng: RNG, inout tape: Tape, inout params: List[Int]) -> Mat:
    var m: Mat = Mat(rows, cols)
    for i in range(rows):
        for j in range(cols):
            var p: Int = tape.val(rng.gauss(0.0, 0.08))
            m.set(i, j, p)
            params.append(p)
    return m


struct StateDict:
    var wte: Mat
    var wpe: Mat
    var lm_head: Mat
    var attn_wq: Mat
    var attn_wk: Mat
    var attn_wv: Mat
    var attn_wo: Mat
    var mlp_fc1: Mat
    var mlp_fc2: Mat

    fn __init__(inout self):
        self.wte = Mat(0, 0)
        self.wpe = Mat(0, 0)
        self.lm_head = Mat(0, 0)
        self.attn_wq = Mat(0, 0)
        self.attn_wk = Mat(0, 0)
        self.attn_wv = Mat(0, 0)
        self.attn_wo = Mat(0, 0)
        self.mlp_fc1 = Mat(0, 0)
        self.mlp_fc2 = Mat(0, 0)


alias n_head = 4
alias n_embd = 16
alias n_layer = 1
alias block_size = 16


fn gpt(
    inout tape: Tape,
    token_id: Int,
    pos_id: Int,
    state_dict: StateDict,
    inout key_cache: List[Int],
    inout value_cache: List[Int]
) -> List[Int]:
    var x: List[Int] = List[Int]()
    var tok_start: Int = token_id * n_embd
    var pos_start: Int = pos_id * n_embd
    for i in range(n_embd):
        x.append(add(tape, state_dict.wte.data[tok_start + i], state_dict.wpe.data[pos_start + i]))

    var x_residual: List[Int] = rmsnorm(tape, x)
    var x_norm: List[Int] = rmsnorm(tape, x_residual)

    var q: List[Int] = linear(tape, x_norm, state_dict.attn_wq)
    var k: List[Int] = linear(tape, x_norm, state_dict.attn_wk)
    var v: List[Int] = linear(tape, x_norm, state_dict.attn_wv)

    for i in range(len(k)):
        key_cache.append(k[i])
    for i in range(len(v)):
        value_cache.append(v[i])

    var x_attn: List[Int] = List[Int]()
    var head_dim: Int = n_embd // n_head
    var scale: Int = tape.val(sqrt(Float64(head_dim)))
    var num_tokens: Int = len(key_cache) // n_embd

    for h in range(n_head):
        var hs: Int = h * head_dim
        var attn_logits: List[Int] = List[Int]()
        for t in range(num_tokens):
            var k_offset: Int = t * n_embd + hs
            var dot: Int = tape.val(0.0)
            for i in range(head_dim):
                dot = add(tape, dot, mul(tape, q[hs + i], key_cache[k_offset + i]))
            attn_logits.append(div(tape, dot, scale))

        var attn_weights: List[Int] = softmax(tape, attn_logits)
        for i in range(head_dim):
            var sum_v: Int = tape.val(0.0)
            for t in range(num_tokens):
                var v_offset: Int = t * n_embd + hs
                sum_v = add(tape, sum_v, mul(tape, attn_weights[t], value_cache[v_offset + i]))
            x_attn.append(sum_v)

    var attn_proj: List[Int] = linear(tape, x_attn, state_dict.attn_wo)
    var x_next: List[Int] = List[Int]()
    for i in range(n_embd):
        x_next.append(add(tape, attn_proj[i], x_residual[i]))

    var x_mlp_norm: List[Int] = rmsnorm(tape, x_next)
    var hidden: List[Int] = linear(tape, x_mlp_norm, state_dict.mlp_fc1)
    for i in range(len(hidden)):
        hidden[i] = relu_op(tape, hidden[i])
    var mlp_out: List[Int] = linear(tape, hidden, state_dict.mlp_fc2)

    var x_out: List[Int] = List[Int]()
    for i in range(n_embd):
        x_out.append(add(tape, mlp_out[i], x_next[i]))

    return linear(tape, x_out, state_dict.lm_head)


fn main() raises:
    var text: String = Path("input.txt").read_text()
    var lines: List[String] = text.split("\n")
    var docs: List[String] = List[String]()
    for i in range(len(lines)):
        var line: String = lines[i].strip()
        if len(line) > 0:
            docs.append(line)

    var rng: RNG = RNG(42)
    random_shuffle_docs(rng, docs)
    var num_docs: Int = len(docs)
    print("num docs:", num_docs)

    var present: List[Bool] = List[Bool]()
    for _ in range(256):
        present.append(False)
    for i in range(num_docs):
        var doc: String = docs[i]
        for j in range(len(doc)):
            present[ord(doc[j])] = True

    var uchars: List[String] = List[String]()
    for i in range(256):
        if present[i]:
            uchars.append(chr(i))

    var char_to_token: List[Int] = List[Int]()
    for _ in range(256):
        char_to_token.append(-1)
    for i in range(len(uchars)):
        char_to_token[ord(uchars[i])] = i

    var BOS: Int = len(uchars)
    var vocab_size: Int = BOS + 1
    print("vocab size:", vocab_size)

    var tape: Tape = Tape()
    var params: List[Int] = List[Int]()
    var state_dict: StateDict = StateDict()
    state_dict.wte = matrix(vocab_size, n_embd, rng, tape, params)
    state_dict.wpe = matrix(block_size, n_embd, rng, tape, params)
    state_dict.lm_head = matrix(vocab_size, n_embd, rng, tape, params)
    state_dict.attn_wq = matrix(n_embd, n_embd, rng, tape, params)
    state_dict.attn_wk = matrix(n_embd, n_embd, rng, tape, params)
    state_dict.attn_wv = matrix(n_embd, n_embd, rng, tape, params)
    state_dict.attn_wo = matrix(n_embd, n_embd, rng, tape, params)
    state_dict.mlp_fc1 = matrix(4 * n_embd, n_embd, rng, tape, params)
    state_dict.mlp_fc2 = matrix(n_embd, 4 * n_embd, rng, tape, params)

    var num_params: Int = len(params)
    print("num params:", num_params)

    var learning_rate: Float64 = 0.01
    var beta1: Float64 = 0.85
    var beta2: Float64 = 0.99
    var eps: Float64 = 1e-8
    var m_vec: List[Float64] = List[Float64]()
    var v_vec: List[Float64] = List[Float64]()
    for _ in range(num_params):
        m_vec.append(0.0)
        v_vec.append(0.0)

    var num_steps: Int = 1000
    for step in range(num_steps):
        tape.reset(num_params)

        var doc: String = docs[step % num_docs]
        var token_ids: List[Int] = List[Int]()
        token_ids.append(BOS)
        for i in range(len(doc)):
            token_ids.append(char_to_token[ord(doc[i])])
        token_ids.append(BOS)

        var key_cache: List[Int] = List[Int]()
        var value_cache: List[Int] = List[Int]()

        var n: Int = len(token_ids) - 1
        if n > block_size:
            n = block_size

        var losses: List[Int] = List[Int]()
        for pos_id in range(n):
            var token_id: Int = token_ids[pos_id]
            var target_id: Int = token_ids[pos_id + 1]
            var logits: List[Int] = gpt(tape, token_id, pos_id, state_dict, key_cache, value_cache)
            var probs: List[Int] = softmax(tape, logits)
            losses.append(mul(tape, log_op(tape, probs[target_id]), tape.val(-1.0)))

        var loss: Int = tape.val(0.0)
        for i in range(len(losses)):
            loss = add(tape, loss, losses[i])
        loss = mul(tape, loss, tape.val(1.0 / Float64(n)))

        backward(tape, loss)

        var lr_t: Float64 = learning_rate * (1.0 - Float64(step) / Float64(num_steps))
        for i in range(num_params):
            var p: Int = params[i]
            var g: Float64 = tape.grad[p]
            m_vec[i] = beta1 * m_vec[i] + (1.0 - beta1) * g
            v_vec[i] = beta2 * v_vec[i] + (1.0 - beta2) * (g * g)
            var m_hat: Float64 = m_vec[i] / (1.0 - beta1 ** (step + 1))
            var v_hat: Float64 = v_vec[i] / (1.0 - beta2 ** (step + 1))
            tape.data[p] -= lr_t * m_hat / (sqrt(v_hat) + eps)
            tape.grad[p] = 0.0

        print("\rstep", step + 1, "/", num_steps, "| loss", tape.data[loss], end="", flush=True)
        if step + 1 == num_steps:
            print()

    print("\n--- inference (new, hallucinated names) ---")
    var temperature: Float64 = 0.5
    for sample_idx in range(20):
        tape.reset(num_params)
        var key_cache: List[Int] = List[Int]()
        var value_cache: List[Int] = List[Int]()
        var token_id: Int = BOS
        var sample: String = String("")

        for pos_id in range(block_size):
            var logits: List[Int] = gpt(tape, token_id, pos_id, state_dict, key_cache, value_cache)
            var scaled_logits: List[Int] = List[Int]()
            for i in range(len(logits)):
                scaled_logits.append(div_f(tape, logits[i], temperature))
            var probs: List[Int] = softmax(tape, scaled_logits)
            var next_id: Int = random_choices(rng, tape, probs)

            if next_id == BOS:
                break

            sample += uchars[next_id]
            token_id = next_id

        print("sample", sample_idx + 1, ":", sample)

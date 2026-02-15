import Foundation

func random_seed(_ seed: Int) {
    srand48(seed)
}

func random_gauss(mean: Double = 0.0, std: Double = 1.0) -> Double {
    // Box-Muller transform
    let u1 = max(1e-12, drand48())
    let u2 = drand48()
    let z0 = sqrt(-2.0 * log(u1)) * cos(2.0 * Double.pi * u2)
    return mean + std * z0
}

func random_shuffle<T>(_ array: inout [T]) {
    guard array.count > 1 else { return }
    for i in stride(from: array.count - 1, through: 1, by: -1) {
        let j = Int(drand48() * Double(i + 1))
        array.swapAt(i, j)
    }
}

func random_choices(weights: [Double]) -> Int {
    let total = weights.reduce(0.0, +)
    let r = drand48() * total
    var cumsum = 0.0
    for (i, w) in weights.enumerated() {
        cumsum += w
        if r <= cumsum {
            return i
        }
    }
    return weights.count - 1
}

random_seed(42)

let input_path = "input.txt"
var docs = try! String(contentsOfFile: input_path, encoding: .utf8)
    .split(separator: "\n")
    .map { String($0).trimmingCharacters(in: .whitespaces) }
    .filter { !$0.isEmpty }
random_shuffle(&docs)
print("num docs: \(docs.count)")

let uchars = Array(Set(docs.joined())).sorted()
let char_to_token = Dictionary(uniqueKeysWithValues: uchars.enumerated().map { ($0.element, $0.offset) })
let BOS = uchars.count
let vocab_size = uchars.count + 1
print("vocab size: \(vocab_size)")

class Value {
    var data: Double
    var grad: Double
    var _children: [Value]
    var _local_grads: [Double]

    init(_ data: Double, children: [Value] = [], local_grads: [Double] = []) {
        self.data = data
        self.grad = 0.0
        self._children = children
        self._local_grads = local_grads
    }

    static func + (lhs: Value, rhs: Value) -> Value {
        return Value(lhs.data + rhs.data, children: [lhs, rhs], local_grads: [1.0, 1.0])
    }

    static func + (lhs: Value, rhs: Double) -> Value {
        return lhs + Value(rhs)
    }

    static func + (lhs: Double, rhs: Value) -> Value {
        return Value(lhs) + rhs
    }

    static func * (lhs: Value, rhs: Value) -> Value {
        return Value(lhs.data * rhs.data, children: [lhs, rhs], local_grads: [rhs.data, lhs.data])
    }

    static func * (lhs: Value, rhs: Double) -> Value {
        return lhs * Value(rhs)
    }

    static func * (lhs: Double, rhs: Value) -> Value {
        return Value(lhs) * rhs
    }

    func pow(_ other: Double) -> Value {
        return Value(
            Foundation.pow(data, other), children: [self],
            local_grads: [other * Foundation.pow(data, other - 1.0)])
    }

    func log() -> Value {
        return Value(Foundation.log(data), children: [self], local_grads: [1.0 / data])
    }

    func exp() -> Value {
        return Value(Foundation.exp(data), children: [self], local_grads: [Foundation.exp(data)])
    }

    func relu() -> Value {
        return Value(max(0.0, data), children: [self], local_grads: [data > 0 ? 1.0 : 0.0])
    }

    static prefix func - (x: Value) -> Value {
        return x * -1.0
    }

    static func - (lhs: Value, rhs: Value) -> Value {
        return lhs + (-rhs)
    }

    static func - (lhs: Double, rhs: Value) -> Value {
        return Value(lhs) + (-rhs)
    }

    static func - (lhs: Value, rhs: Double) -> Value {
        return lhs + (-Value(rhs))
    }

    static func / (lhs: Value, rhs: Value) -> Value {
        return lhs * rhs.pow(-1.0)
    }

    static func / (lhs: Value, rhs: Double) -> Value {
        return lhs * Value(rhs).pow(-1.0)
    }

    static func / (lhs: Double, rhs: Value) -> Value {
        return Value(lhs) * rhs.pow(-1.0)
    }

    func backward() {
        var topo: [Value] = []
        var visited = Set<ObjectIdentifier>()
        func build_topo(_ v: Value) {
            let id = ObjectIdentifier(v)
            if !visited.contains(id) {
                visited.insert(id)
                for child in v._children {
                    build_topo(child)
                }
                topo.append(v)
            }
        }
        build_topo(self)
        self.grad = 1.0
        for v in topo.reversed() {
            for (child, local_grad) in zip(v._children, v._local_grads) {
                child.grad += local_grad * v.grad
            }
        }
    }
}

let n_embd = 16
let n_head = 4
let n_layer = 1
let block_size = 16
let head_dim = n_embd / n_head

func matrix(_ nout: Int, _ nin: Int, std: Double = 0.08) -> [[Value]] {
    return (0..<nout).map { _ in
        (0..<nin).map { _ in Value(random_gauss(mean: 0.0, std: std)) }
    }
}

var state_dict: [String: [[Value]]] = [
    "wte": matrix(vocab_size, n_embd),
    "wpe": matrix(block_size, n_embd),
    "lm_head": matrix(vocab_size, n_embd),
]
for i in 0..<n_layer {
    state_dict["layer\(i).attn_wq"] = matrix(n_embd, n_embd)
    state_dict["layer\(i).attn_wk"] = matrix(n_embd, n_embd)
    state_dict["layer\(i).attn_wv"] = matrix(n_embd, n_embd)
    state_dict["layer\(i).attn_wo"] = matrix(n_embd, n_embd)
    state_dict["layer\(i).mlp_fc1"] = matrix(4 * n_embd, n_embd)
    state_dict["layer\(i).mlp_fc2"] = matrix(n_embd, 4 * n_embd)
}
var params: [Value] = []
func append_params(_ m: [[Value]]) {
    for row in m {
        params.append(contentsOf: row)
    }
}
append_params(state_dict["wte"]!)
append_params(state_dict["wpe"]!)
append_params(state_dict["lm_head"]!)
for i in 0..<n_layer {
    append_params(state_dict["layer\(i).attn_wq"]!)
    append_params(state_dict["layer\(i).attn_wk"]!)
    append_params(state_dict["layer\(i).attn_wv"]!)
    append_params(state_dict["layer\(i).attn_wo"]!)
    append_params(state_dict["layer\(i).mlp_fc1"]!)
    append_params(state_dict["layer\(i).mlp_fc2"]!)
}
print("num params: \(params.count)")

func linear(_ x: [Value], _ w: [[Value]]) -> [Value] {
    return w.map { wo in
        zip(wo, x).map(*).reduce(Value(0.0), +)
    }
}

func softmax(_ logits: [Value]) -> [Value] {
    let max_val = logits.map { $0.data }.max()!
    let exps = logits.map { ($0 - max_val).exp() }
    let total = exps.reduce(Value(0.0), +)
    return exps.map { $0 / total }
}

func rmsnorm(_ x: [Value]) -> [Value] {
    let ms = x.map { $0 * $0 }.reduce(Value(0.0), +) / Double(x.count)
    let scale = (ms + 1e-5).pow(-0.5)
    return x.map { $0 * scale }
}

func gpt(_ token_id: Int, _ pos_id: Int, _ keys: inout [[[Value]]], _ values: inout [[[Value]]])
    -> [Value]
{
    let tok_emb = state_dict["wte"]![token_id]
    let pos_emb = state_dict["wpe"]![pos_id]
    var x = zip(tok_emb, pos_emb).map(+)
    x = rmsnorm(x)

    for li in 0..<n_layer {
        // 1) Multi-head attention block
        let x_residual = x
        x = rmsnorm(x)
        let q = linear(x, state_dict["layer\(li).attn_wq"]!)
        let k = linear(x, state_dict["layer\(li).attn_wk"]!)
        let v = linear(x, state_dict["layer\(li).attn_wv"]!)
        keys[li].append(k)
        values[li].append(v)
        var x_attn: [Value] = []
        for h in 0..<n_head {
            let hs = h * head_dim
            var attn_logits: [Value] = []
            for t in 0..<keys[li].count {
                var dot = Value(0.0)
                for j in 0..<head_dim {
                    dot = dot + q[hs + j] * keys[li][t][hs + j]
                }
                attn_logits.append(dot / Foundation.pow(Double(head_dim), 0.5))
            }
            let attn_weights = softmax(attn_logits)
            let head_out = (0..<head_dim).map { j -> Value in
                var out = Value(0.0)
                for t in 0..<values[li].count {
                    out = out + attn_weights[t] * values[li][t][hs + j]
                }
                return out
            }
            x_attn.append(contentsOf: head_out)
        }
        x = linear(x_attn, state_dict["layer\(li).attn_wo"]!)
        x = zip(x, x_residual).map(+)
        // 2) MLP block
        let x_residual2 = x
        x = rmsnorm(x)
        x = linear(x, state_dict["layer\(li).mlp_fc1"]!)
        x = x.map { $0.relu() }
        x = linear(x, state_dict["layer\(li).mlp_fc2"]!)
        x = zip(x, x_residual2).map(+)
    }

    let logits = linear(x, state_dict["lm_head"]!)
    return logits
}

let learning_rate = 0.01
let beta1 = 0.85
let beta2 = 0.99
let eps_adam = 1e-8
var m = Array(repeating: 0.0, count: params.count)
var v = Array(repeating: 0.0, count: params.count)

let num_steps = 1000
for step in 0..<num_steps {

    let doc = docs[step % docs.count]
    let tokens = [BOS] + doc.map { char_to_token[$0]! } + [BOS]
    let n = min(block_size, tokens.count - 1)

    var keys = Array(repeating: [[Value]](), count: n_layer)
    var values = Array(repeating: [[Value]](), count: n_layer)
    var losses: [Value] = []
    for pos_id in 0..<n {
        let token_id = tokens[pos_id]
        let target_id = tokens[pos_id + 1]
        let logits = gpt(token_id, pos_id, &keys, &values)
        let probs = softmax(logits)
        let loss_t = -probs[target_id].log()
        losses.append(loss_t)
    }
    let loss = (1.0 / Double(n)) * losses.reduce(Value(0.0), +)

    loss.backward()

    let lr_t = learning_rate * (1.0 - Double(step) / Double(num_steps))
    for i in 0..<params.count {
        let p = params[i]
        m[i] = beta1 * m[i] + (1.0 - beta1) * p.grad
        v[i] = beta2 * v[i] + (1.0 - beta2) * p.grad * p.grad
        let m_hat = m[i] / (1.0 - Foundation.pow(beta1, Double(step + 1)))
        let v_hat = v[i] / (1.0 - Foundation.pow(beta2, Double(step + 1)))
        p.data -= lr_t * m_hat / (Foundation.pow(v_hat, 0.5) + eps_adam)
        p.grad = 0.0
    }

    fputs(String(format: "\rstep %4d / %4d | loss %.4f", step + 1, num_steps, loss.data), stdout)
    fflush(stdout)
    if step + 1 == num_steps { fputs("\n", stdout) }
}

let temperature = 0.5
print("\n--- inference (new, hallucinated names) ---")
for sample_idx in 0..<20 {
    var keys = Array(repeating: [[Value]](), count: n_layer)
    var values = Array(repeating: [[Value]](), count: n_layer)
    var token_id = BOS
    var sample: [Character] = []
    for pos_id in 0..<block_size {
        let logits = gpt(token_id, pos_id, &keys, &values)
        let probs = softmax(logits.map { $0 / temperature })
        token_id = random_choices(weights: probs.map { $0.data })
        if token_id == BOS {
            break
        }
        sample.append(uchars[token_id])
    }
    print(String(format: "sample %2d: %@", sample_idx + 1, String(sample)))
}

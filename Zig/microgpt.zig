const std = @import("std");
const math = std.math;
const Allocator = std.mem.Allocator;

const n_embd: usize = 16;
const n_head: usize = 4;
const n_layer: usize = 1;
const block_size: usize = 16;
const head_dim: usize = n_embd / n_head;

fn print(comptime fmt: []const u8, args: anytype) void {
    var buf: [4096]u8 = undefined;
    const s = std.fmt.bufPrint(&buf, fmt, args) catch return;
    _ = std.posix.write(std.posix.STDOUT_FILENO, s) catch return;
}

var rng: std.Random.DefaultPrng = undefined;

fn randomInit(seed: u64) void {
    rng = std.Random.DefaultPrng.init(seed);
}

fn randomFloat() f64 {
    return rng.random().float(f64);
}

fn randomGauss(mean: f64, stddev: f64) f64 {
    const r1 = @max(1e-12, randomFloat());
    const r2 = randomFloat();
    const z0 = @sqrt(-2.0 * @log(r1)) * @cos(2.0 * math.pi * r2);
    return mean + stddev * z0;
}

fn randomShuffle(comptime T: type, arr: []T) void {
    if (arr.len <= 1) return;
    var i = arr.len - 1;
    while (i >= 1) : (i -= 1) {
        const j: usize = @intFromFloat(randomFloat() * @as(f64, @floatFromInt(i + 1)));
        const tmp = arr[i];
        arr[i] = arr[j];
        arr[j] = tmp;
    }
}

fn randomChoices(weights: []const f64) usize {
    var total: f64 = 0;
    for (weights) |w| total += w;
    const r = randomFloat() * total;
    var cumsum: f64 = 0;
    for (weights, 0..) |w, i| {
        cumsum += w;
        if (r <= cumsum) return i;
    }
    return weights.len - 1;
}

const Value = struct {
    data: f64,
    grad: f64,
    _children: [2]?*Value,
    _local_grads: [2]f64,
    n_children: u8,

    fn init(alloc: Allocator, data: f64) !*Value {
        const v = try alloc.create(Value);
        v.* = .{
            .data = data,
            .grad = 0,
            ._children = .{ null, null },
            ._local_grads = .{ 0, 0 },
            .n_children = 0,
        };
        return v;
    }

    fn withChildren1(alloc: Allocator, data: f64, c0: *Value, g0: f64) !*Value {
        const v = try alloc.create(Value);
        v.* = .{
            .data = data,
            .grad = 0,
            ._children = .{ c0, null },
            ._local_grads = .{ g0, 0 },
            .n_children = 1,
        };
        return v;
    }

    fn withChildren2(alloc: Allocator, data: f64, c0: *Value, g0: f64, c1: *Value, g1: f64) !*Value {
        const v = try alloc.create(Value);
        v.* = .{
            .data = data,
            .grad = 0,
            ._children = .{ c0, c1 },
            ._local_grads = .{ g0, g1 },
            .n_children = 2,
        };
        return v;
    }

    fn add(self: *Value, alloc: Allocator, other: *Value) !*Value {
        return withChildren2(alloc, self.data + other.data, self, 1.0, other, 1.0);
    }

    fn mul(self: *Value, alloc: Allocator, other: *Value) !*Value {
        return withChildren2(alloc, self.data * other.data, self, other.data, other, self.data);
    }

    fn mulf(self: *Value, alloc: Allocator, scalar: f64) !*Value {
        const other = try init(alloc, scalar);
        return self.mul(alloc, other);
    }

    fn sub(self: *Value, alloc: Allocator, other: *Value) !*Value {
        const negated = try other.mulf(alloc, -1.0);
        return self.add(alloc, negated);
    }

    fn subf(self: *Value, alloc: Allocator, scalar: f64) !*Value {
        const other = try init(alloc, scalar);
        return self.sub(alloc, other);
    }

    fn divv(self: *Value, alloc: Allocator, other: *Value) !*Value {
        const inv = try other.powf(alloc, -1.0);
        return self.mul(alloc, inv);
    }

    fn divf(self: *Value, alloc: Allocator, scalar: f64) !*Value {
        const other = try init(alloc, scalar);
        return self.divv(alloc, other);
    }

    fn powf(self: *Value, alloc: Allocator, exponent: f64) !*Value {
        return withChildren1(
            alloc,
            math.pow(f64, self.data, exponent),
            self,
            exponent * math.pow(f64, self.data, exponent - 1.0),
        );
    }

    fn log(self: *Value, alloc: Allocator) !*Value {
        return withChildren1(alloc, @log(self.data), self, 1.0 / self.data);
    }

    fn exp(self: *Value, alloc: Allocator) !*Value {
        return withChildren1(alloc, @exp(self.data), self, @exp(self.data));
    }

    fn relu(self: *Value, alloc: Allocator) !*Value {
        return withChildren1(
            alloc,
            @max(0.0, self.data),
            self,
            if (self.data > 0) 1.0 else 0.0,
        );
    }

    fn negv(self: *Value, alloc: Allocator) !*Value {
        return self.mulf(alloc, -1.0);
    }

    fn buildTopo(v: *Value, visited: *std.AutoHashMap(*Value, void), topo: *std.ArrayList(*Value), scratch: Allocator) !void {
        if (visited.get(v) != null) return;
        try visited.put(v, {});
        var ci: u8 = 0;
        while (ci < v.n_children) : (ci += 1) {
            try buildTopo(v._children[ci].?, visited, topo, scratch);
        }
        try topo.append(scratch, v);
    }

    fn backward(self: *Value, scratch: Allocator) !void {
        var topo = std.ArrayList(*Value).empty;
        var visited = std.AutoHashMap(*Value, void).init(scratch);
        try buildTopo(self, &visited, &topo, scratch);
        self.grad = 1.0;
        var i = topo.items.len;
        while (i > 0) {
            i -= 1;
            const v = topo.items[i];
            var ci: u8 = 0;
            while (ci < v.n_children) : (ci += 1) {
                v._children[ci].?.grad += v._local_grads[ci] * v.grad;
            }
        }
    }
};

fn linear(alloc: Allocator, x: []*Value, w: [][]*Value) ![]*Value {
    const result = try alloc.alloc(*Value, w.len);
    for (w, 0..) |wo, i| {
        var sum = try Value.init(alloc, 0);
        for (wo, x) |wi, xi| {
            const prod = try wi.mul(alloc, xi);
            sum = try sum.add(alloc, prod);
        }
        result[i] = sum;
    }
    return result;
}

fn softmax(alloc: Allocator, logits: []*Value) ![]*Value {
    var max_val: f64 = -math.inf(f64);
    for (logits) |v| {
        if (v.data > max_val) max_val = v.data;
    }
    const exps = try alloc.alloc(*Value, logits.len);
    for (logits, 0..) |v, i| {
        const shifted = try v.subf(alloc, max_val);
        exps[i] = try shifted.exp(alloc);
    }
    var total = try Value.init(alloc, 0);
    for (exps) |e| total = try total.add(alloc, e);
    const result = try alloc.alloc(*Value, logits.len);
    for (exps, 0..) |e, i| result[i] = try e.divv(alloc, total);
    return result;
}

fn rmsnorm(alloc: Allocator, x: []*Value) ![]*Value {
    var ms = try Value.init(alloc, 0);
    for (x) |xi| {
        const sq = try xi.mul(alloc, xi);
        ms = try ms.add(alloc, sq);
    }
    ms = try ms.divf(alloc, @floatFromInt(x.len));
    const eps = try Value.init(alloc, 1e-5);
    const ms_eps = try ms.add(alloc, eps);
    const scale = try ms_eps.powf(alloc, -0.5);
    const result = try alloc.alloc(*Value, x.len);
    for (x, 0..) |xi, i| result[i] = try xi.mul(alloc, scale);
    return result;
}

const Layer = struct {
    attn_wq: [][]*Value,
    attn_wk: [][]*Value,
    attn_wv: [][]*Value,
    attn_wo: [][]*Value,
    mlp_fc1: [][]*Value,
    mlp_fc2: [][]*Value,
};

const StateDict = struct {
    wte: [][]*Value,
    wpe: [][]*Value,
    lm_head: [][]*Value,
    layers: []Layer,
};

fn matrix(alloc: Allocator, params: *std.ArrayList(*Value), nout: usize, nin: usize) ![][]*Value {
    const rows = try alloc.alloc([]*Value, nout);
    for (0..nout) |r| {
        rows[r] = try alloc.alloc(*Value, nin);
        for (0..nin) |c| {
            const v = try Value.init(alloc, randomGauss(0, 0.08));
            rows[r][c] = v;
            try params.append(alloc, v);
        }
    }
    return rows;
}

fn gpt(
    alloc: Allocator,
    token_id: usize,
    pos_id: usize,
    keys: []std.ArrayList([]*Value),
    values: []std.ArrayList([]*Value),
    state_dict: *const StateDict,
) ![]*Value {
    const tok_emb = state_dict.wte[token_id];
    const pos_emb = state_dict.wpe[pos_id];
    var x = try alloc.alloc(*Value, n_embd);
    for (0..n_embd) |i| x[i] = try tok_emb[i].add(alloc, pos_emb[i]);
    x = try rmsnorm(alloc, x);

    for (0..n_layer) |li| {
        const x_residual = x;
        x = try rmsnorm(alloc, x);
        const layer = state_dict.layers[li];

        const q = try linear(alloc, x, layer.attn_wq);
        const k = try linear(alloc, x, layer.attn_wk);
        const v = try linear(alloc, x, layer.attn_wv);
        try keys[li].append(alloc, k);
        try values[li].append(alloc, v);

        var x_attn = std.ArrayList(*Value).empty;
        for (0..n_head) |h| {
            const hs = h * head_dim;
            const q_h = q[hs .. hs + head_dim];
            const k_h_list = keys[li].items;
            const v_h_list = values[li].items;

            const attn_logits = try alloc.alloc(*Value, k_h_list.len);
            for (k_h_list, 0..) |ki, t| {
                var dot = try Value.init(alloc, 0);
                for (0..head_dim) |j| {
                    const prod = try q_h[j].mul(alloc, ki[hs + j]);
                    dot = try dot.add(alloc, prod);
                }
                attn_logits[t] = try dot.divf(alloc, @sqrt(@as(f64, @floatFromInt(head_dim))));
            }
            const attn_weights = try softmax(alloc, attn_logits);

            for (0..head_dim) |j| {
                var weighted_sum = try Value.init(alloc, 0);
                for (v_h_list, 0..) |vi, t| {
                    const prod = try attn_weights[t].mul(alloc, vi[hs + j]);
                    weighted_sum = try weighted_sum.add(alloc, prod);
                }
                try x_attn.append(alloc, weighted_sum);
            }
        }

        x = try linear(alloc, x_attn.items, layer.attn_wo);
        for (0..n_embd) |i| x[i] = try x[i].add(alloc, x_residual[i]);

        const x_residual2 = x;
        x = try rmsnorm(alloc, x);
        x = try linear(alloc, x, layer.mlp_fc1);
        const relu_x = try alloc.alloc(*Value, x.len);
        for (x, 0..) |xi, i| relu_x[i] = try xi.relu(alloc);
        x = try linear(alloc, relu_x, layer.mlp_fc2);
        for (0..n_embd) |i| x[i] = try x[i].add(alloc, x_residual2[i]);
    }

    return try linear(alloc, x, state_dict.lm_head);
}

pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const alloc = gpa.allocator();

    randomInit(42);

    const input_path = "input.txt";
    const content = try std.fs.cwd().readFileAlloc(alloc, input_path, 1024 * 1024);

    var doc_list = std.ArrayList([]const u8).empty;
    var lines = std.mem.splitScalar(u8, content, '\n');
    while (lines.next()) |line| {
        const trimmed = std.mem.trim(u8, line, &[_]u8{ ' ', '\t', '\r' });
        if (trimmed.len > 0) try doc_list.append(alloc, trimmed);
    }
    const docs = doc_list.items;
    randomShuffle([]const u8, docs);
    print("num docs: {d}\n", .{docs.len});

    var char_set = std.AutoHashMap(u8, void).init(alloc);
    for (docs) |doc| {
        for (doc) |ch| try char_set.put(ch, {});
    }
    var uchars = std.ArrayList(u8).empty;
    var it = char_set.keyIterator();
    while (it.next()) |key| try uchars.append(alloc, key.*);
    std.mem.sort(u8, uchars.items, {}, std.sort.asc(u8));
    const BOS = uchars.items.len;
    const vocab_size = uchars.items.len + 1;
    var char_to_token: [256]usize = [_]usize{0} ** 256;
    for (uchars.items, 0..) |ch, i| char_to_token[ch] = i;
    print("vocab size: {d}\n", .{vocab_size});

    var params_list = std.ArrayList(*Value).empty;
    const wte = try matrix(alloc, &params_list, vocab_size, n_embd);
    const wpe = try matrix(alloc, &params_list, block_size, n_embd);
    const lm_head = try matrix(alloc, &params_list, vocab_size, n_embd);
    const layers = try alloc.alloc(Layer, n_layer);
    for (0..n_layer) |i| {
        layers[i] = .{
            .attn_wq = try matrix(alloc, &params_list, n_embd, n_embd),
            .attn_wk = try matrix(alloc, &params_list, n_embd, n_embd),
            .attn_wv = try matrix(alloc, &params_list, n_embd, n_embd),
            .attn_wo = try matrix(alloc, &params_list, n_embd, n_embd),
            .mlp_fc1 = try matrix(alloc, &params_list, 4 * n_embd, n_embd),
            .mlp_fc2 = try matrix(alloc, &params_list, n_embd, 4 * n_embd),
        };
    }
    const state_dict = StateDict{
        .wte = wte,
        .wpe = wpe,
        .lm_head = lm_head,
        .layers = layers,
    };
    const params = params_list.items;
    print("num params: {d}\n", .{params.len});

    const learning_rate: f64 = 0.01;
    const beta1: f64 = 0.85;
    const beta2: f64 = 0.99;
    const eps_adam: f64 = 1e-8;
    const m = try alloc.alloc(f64, params.len);
    const v = try alloc.alloc(f64, params.len);
    @memset(m, 0);
    @memset(v, 0);

    const num_steps: usize = 1000;
    for (0..num_steps) |step| {
        const doc = docs[step % docs.len];
        const tokens = try alloc.alloc(usize, doc.len + 2);
        defer alloc.free(tokens);
        tokens[0] = BOS;
        for (doc, 0..) |ch, i| {
            tokens[i + 1] = char_to_token[ch];
        }
        tokens[doc.len + 1] = BOS;
        const n = @min(block_size, tokens.len - 1);

        var arena = std.heap.ArenaAllocator.init(std.heap.page_allocator);
        defer arena.deinit();
        const arena_alloc = arena.allocator();

        const keys = try arena_alloc.alloc(std.ArrayList([]*Value), n_layer);
        const values = try arena_alloc.alloc(std.ArrayList([]*Value), n_layer);
        for (0..n_layer) |li| {
            keys[li] = std.ArrayList([]*Value).empty;
            values[li] = std.ArrayList([]*Value).empty;
        }

        var losses = std.ArrayList(*Value).empty;
        for (0..n) |pos_id| {
            const token_id = tokens[pos_id];
            const target_id = tokens[pos_id + 1];
            const logits = try gpt(arena_alloc, token_id, pos_id, keys, values, &state_dict);
            const probs = try softmax(arena_alloc, logits);
            const log_prob = try probs[target_id].log(arena_alloc);
            const loss_t = try log_prob.negv(arena_alloc);
            try losses.append(arena_alloc, loss_t);
        }

        var total_loss = try Value.init(arena_alloc, 0);
        for (losses.items) |l| total_loss = try total_loss.add(arena_alloc, l);
        const loss = try total_loss.mulf(arena_alloc, 1.0 / @as(f64, @floatFromInt(n)));

        try loss.backward(arena_alloc);

        const lr_t = learning_rate * (1.0 - @as(f64, @floatFromInt(step)) / @as(f64, @floatFromInt(num_steps)));
        for (params, 0..) |p, i| {
            m[i] = beta1 * m[i] + (1.0 - beta1) * p.grad;
            v[i] = beta2 * v[i] + (1.0 - beta2) * p.grad * p.grad;
            const m_hat = m[i] / (1.0 - math.pow(f64, beta1, @as(f64, @floatFromInt(step + 1))));
            const v_hat = v[i] / (1.0 - math.pow(f64, beta2, @as(f64, @floatFromInt(step + 1))));
            p.data -= lr_t * m_hat / (@sqrt(v_hat) + eps_adam);
            p.grad = 0;
        }

        print("\rstep {d:4} / {d:4} | loss {d:.4}", .{ step + 1, num_steps, loss.data });
        if (step + 1 == num_steps) print("\n", .{});
    }

    const temperature: f64 = 0.5;
    print("\n--- inference (new, hallucinated names) ---\n", .{});
    for (0..20) |sample_idx| {
        var arena = std.heap.ArenaAllocator.init(std.heap.page_allocator);
        defer arena.deinit();
        const arena_alloc = arena.allocator();

        const keys = try arena_alloc.alloc(std.ArrayList([]*Value), n_layer);
        const values = try arena_alloc.alloc(std.ArrayList([]*Value), n_layer);
        for (0..n_layer) |li| {
            keys[li] = std.ArrayList([]*Value).empty;
            values[li] = std.ArrayList([]*Value).empty;
        }

        var token_id: usize = BOS;
        var sample = std.ArrayList(u8).empty;
        defer sample.deinit(alloc);
        for (0..block_size) |pos_id| {
            const logits = try gpt(arena_alloc, token_id, pos_id, keys, values, &state_dict);
            const scaled = try arena_alloc.alloc(*Value, logits.len);
            for (logits, 0..) |l, i| scaled[i] = try l.divf(arena_alloc, temperature);
            const probs = try softmax(arena_alloc, scaled);
            const weights = try arena_alloc.alloc(f64, probs.len);
            for (probs, 0..) |p, i| weights[i] = p.data;
            token_id = randomChoices(weights);
            if (token_id == BOS) break;
            try sample.append(alloc, uchars.items[token_id]);
        }
        print("sample {d:2}: {s}\n", .{ sample_idx + 1, sample.items });
    }
}

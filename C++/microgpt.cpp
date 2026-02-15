#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <memory>
#include <random>
#include <ranges>
#include <span>
#include <stdexcept>
#include <string>
#include <unordered_set>
#include <vector>

struct Value;
using V = std::shared_ptr<Value>;
using Sequence = std::vector<std::vector<V>>;
using KVCache = std::vector<Sequence>;
using Matrix = std::vector<std::vector<V>>;

constexpr int n_embd = 16;
constexpr int n_head = 4;
constexpr int n_layer = 1;
constexpr int block_size = 16;
constexpr int head_dim = n_embd / n_head;
constexpr int num_steps = 1000;

struct Value : public std::enable_shared_from_this<Value> {
    double data;
    double grad;
    std::array<V, 2> _children;
    std::array<double, 2> _local_grads;
    std::size_t _n_children;

private:
    explicit Value(double d) : data(d), grad(0.0), _children{}, _local_grads{}, _n_children(0) {}
    Value(double d, const V& c0, double g0)
        : data(d), grad(0.0), _children{c0, nullptr}, _local_grads{g0, 0.0}, _n_children(1) {}
    Value(double d, const V& c0, double g0, const V& c1, double g1)
        : data(d), grad(0.0), _children{c0, c1}, _local_grads{g0, g1}, _n_children(2) {}

public:
    static V create(double d) { return V(new Value(d)); }
    static V create(double d, const V& c0, double g0) { return V(new Value(d, c0, g0)); }
    static V create(double d, const V& c0, double g0, const V& c1, double g1) { return V(new Value(d, c0, g0, c1, g1)); }

    V pow(double other) { return Value::create(std::pow(data, other), shared_from_this(), other * std::pow(data, other - 1.0)); }
    V log() { return Value::create(std::log(data), shared_from_this(), 1.0 / data); }
    V exp() {
        double ex = std::exp(data);
        return Value::create(ex, shared_from_this(), ex);
    }
    V relu() { return Value::create(std::max(0.0, data), shared_from_this(), data > 0.0 ? 1.0 : 0.0); }

    void backward() {
        std::vector<V> topo;
        std::unordered_set<Value*> visited;
        auto build_topo = [&](auto&& self, const V& v) -> void {
            if (visited.insert(v.get()).second) {
                for (std::size_t i = 0; i < v->_n_children; ++i) self(self, v->_children[i]);
                topo.push_back(v);
            }
        };
        build_topo(build_topo, shared_from_this());
        grad = 1.0;

        for (auto it = topo.rbegin(); it != topo.rend(); ++it) {
            const auto& v = *it;
            for (std::size_t i = 0; i < v->_n_children; ++i) v->_children[i]->grad += v->_local_grads[i] * v->grad;
        }
    }
};

static inline V val(double x) { return Value::create(x); }

static inline V operator+(const V& lhs, const V& rhs) { return Value::create(lhs->data + rhs->data, lhs, 1.0, rhs, 1.0); }
static inline V operator*(const V& lhs, const V& rhs) { return Value::create(lhs->data * rhs->data, lhs, rhs->data, rhs, lhs->data); }
static inline V operator+(const V& lhs, double rhs) { return lhs + val(rhs); }
static inline V operator*(const V& lhs, double rhs) { return lhs * val(rhs); }
static inline V operator*(double lhs, const V& rhs) { return val(lhs) * rhs; }
static inline V operator-(const V& x) { return x * -1.0; }
static inline V operator-(const V& lhs, double rhs) { return lhs + (-val(rhs)); }
static inline V operator/(const V& lhs, const V& rhs) { return lhs * rhs->pow(-1.0); }
static inline V operator/(const V& lhs, double rhs) { return lhs * std::pow(rhs, -1.0); }

struct Layer { Matrix attn_wq, attn_wk, attn_wv, attn_wo, mlp_fc1, mlp_fc2; };
struct StateDict { Matrix wte, wpe, lm_head; std::vector<Layer> layers; };

static Matrix matrix(std::size_t nout, std::size_t nin, std::mt19937& rng, double std = 0.08) {
    std::normal_distribution<double> gauss(0.0, std);
    Matrix out(nout, std::vector<V>(nin));
    for (std::size_t i = 0; i < nout; ++i) for (std::size_t j = 0; j < nin; ++j) out[i][j] = val(gauss(rng));
    return out;
}

static std::vector<V> linear(std::span<const V> x, const Matrix& w) {
    std::vector<V> out(w.size());
    for (size_t i = 0; i < w.size(); i++) {
        V sum = val(0.0);
        for (size_t j = 0; j < w[i].size(); j++) {
            sum = sum + (w[i][j] * x[j]);
        }
        out[i] = sum;
    }
    return out;
}

static std::vector<V> softmax(std::span<const V> logits) {
    if (logits.empty()) throw std::runtime_error("softmax() received empty logits");
    double max_val = logits[0]->data;
    for (const auto& v : logits) if (v->data > max_val) max_val = v->data;
    std::vector<V> exps(logits.size());
    V total = val(0.0);
    for (size_t i = 0; i < logits.size(); i++) {
        exps[i] = (logits[i] - max_val)->exp();
        total = total + exps[i];
    }
    std::vector<V> out(logits.size());
    for (size_t i = 0; i < logits.size(); i++) out[i] = exps[i] / total;
    return out;
}

static std::vector<V> rmsnorm(std::span<const V> x) {
    V ms = val(0.0);
    for (const auto& xi : x) ms = ms + (xi * xi);
    ms = ms / static_cast<double>(x.size());
    V scale = (ms + 1e-5)->pow(-0.5);
    std::vector<V> out(x.size());
    for (size_t i = 0; i < x.size(); i++) out[i] = x[i] * scale;
    return out;
}

static std::vector<V> gpt(int token_id, int pos_id, KVCache& keys, KVCache& values, const StateDict& state_dict) {
    const auto& tok_emb = state_dict.wte[static_cast<std::size_t>(token_id)];
    const auto& pos_emb = state_dict.wpe[static_cast<std::size_t>(pos_id)];

    std::vector<V> x(tok_emb.size());
    for (std::size_t i = 0; i < tok_emb.size(); ++i) x[i] = tok_emb[i] + pos_emb[i];
    x = rmsnorm(x);

    for (std::size_t li = 0; li < static_cast<std::size_t>(n_layer); ++li) {
        std::vector<V> x_residual = x;
        x = rmsnorm(x);

        const auto& layer = state_dict.layers[li];
        std::vector<V> q = linear(x, layer.attn_wq);
        std::vector<V> k = linear(x, layer.attn_wk);
        std::vector<V> v = linear(x, layer.attn_wv);

        keys[li].push_back(k);
        values[li].push_back(v);

        std::vector<V> x_attn;
        x_attn.reserve(static_cast<size_t>(n_embd));
        double attn_scale = 1.0 / std::sqrt(static_cast<double>(head_dim));

        for (std::size_t h = 0; h < static_cast<std::size_t>(n_head); ++h) {
            std::size_t hs = h * static_cast<std::size_t>(head_dim);
            std::vector<V> attn_logits(keys[li].size());
            for (std::size_t t = 0; t < keys[li].size(); ++t) {
                V dot = val(0.0);
                for (std::size_t j = 0; j < static_cast<std::size_t>(head_dim); ++j) dot = dot + (q[hs + j] * keys[li][t][hs + j]);
                attn_logits[t] = dot * attn_scale;
            }

            std::vector<V> attn_weights = softmax(attn_logits);
            for (std::size_t j = 0; j < static_cast<std::size_t>(head_dim); ++j) {
                V out_j = val(0.0);
                for (std::size_t t = 0; t < values[li].size(); ++t) out_j = out_j + (attn_weights[t] * values[li][t][hs + j]);
                x_attn.push_back(out_j);
            }
        }

        x = linear(x_attn, layer.attn_wo);
        for (std::size_t i = 0; i < x.size(); ++i) x[i] = x[i] + x_residual[i];

        x_residual = x;
        x = rmsnorm(x);
        x = linear(x, layer.mlp_fc1);
        for (auto& xi : x) xi = xi->relu();
        x = linear(x, layer.mlp_fc2);
        for (std::size_t i = 0; i < x.size(); ++i) x[i] = x[i] + x_residual[i];
    }

    return linear(x, state_dict.lm_head);
}

int main() {
    std::mt19937 rng(42);

    std::ifstream in("input.txt");
    if (!in) {
        std::cerr << "failed to read input.txt\n";
        return 1;
    }

    std::vector<std::string> docs;
    std::string line;
    while (std::getline(in, line)) {
        auto b = line.find_first_not_of(" \t\r\n");
        if (b == std::string::npos) continue;
        docs.push_back(line.substr(b, line.find_last_not_of(" \t\r\n") - b + 1));
    }
    if (docs.empty()) {
        std::cerr << "input.txt has no non-empty lines\n";
        return 1;
    }
    std::ranges::shuffle(docs, rng);
    std::cout << "num docs: " << docs.size() << "\n";

    bool present[256] = {false};
    for (const auto& doc : docs) {
        for (char ch : doc) present[static_cast<unsigned char>(ch)] = true;
    }
    std::vector<char> uchars;
    for (int i = 0; i < 256; i++) if (present[i]) uchars.push_back(static_cast<char>(i));

    std::array<int, 256> char_to_token;
    char_to_token.fill(-1);
    for (size_t i = 0; i < uchars.size(); i++) char_to_token[static_cast<unsigned char>(uchars[i])] = static_cast<int>(i);

    int BOS = static_cast<int>(uchars.size());
    int vocab_size = BOS + 1;
    std::cout << "vocab size: " << vocab_size << "\n";

    StateDict state_dict;
    std::vector<V> params;

    state_dict.wte = matrix(static_cast<std::size_t>(vocab_size), static_cast<std::size_t>(n_embd), rng);
    state_dict.wpe = matrix(static_cast<std::size_t>(block_size), static_cast<std::size_t>(n_embd), rng);
    state_dict.lm_head = matrix(static_cast<std::size_t>(vocab_size), static_cast<std::size_t>(n_embd), rng);

    auto flatten = [&](const Matrix& m) { for (const auto& row : m) for (const auto& p : row) params.push_back(p); };

    flatten(state_dict.wte);
    flatten(state_dict.wpe);
    flatten(state_dict.lm_head);

    state_dict.layers.resize(static_cast<std::size_t>(n_layer));
    for (std::size_t li = 0; li < static_cast<std::size_t>(n_layer); ++li) {
        auto& layer = state_dict.layers[li];
        layer.attn_wq = matrix(static_cast<std::size_t>(n_embd), static_cast<std::size_t>(n_embd), rng);
        layer.attn_wk = matrix(static_cast<std::size_t>(n_embd), static_cast<std::size_t>(n_embd), rng);
        layer.attn_wv = matrix(static_cast<std::size_t>(n_embd), static_cast<std::size_t>(n_embd), rng);
        layer.attn_wo = matrix(static_cast<std::size_t>(n_embd), static_cast<std::size_t>(n_embd), rng);
        layer.mlp_fc1 = matrix(static_cast<std::size_t>(4 * n_embd), static_cast<std::size_t>(n_embd), rng);
        layer.mlp_fc2 = matrix(static_cast<std::size_t>(n_embd), static_cast<std::size_t>(4 * n_embd), rng);

        flatten(layer.attn_wq);
        flatten(layer.attn_wk);
        flatten(layer.attn_wv);
        flatten(layer.attn_wo);
        flatten(layer.mlp_fc1);
        flatten(layer.mlp_fc2);
    }
    std::cout << "num params: " << params.size() << "\n";

    double learning_rate = 0.01, beta1 = 0.85, beta2 = 0.99, eps_adam = 1e-8;

    std::vector<double> m(params.size(), 0.0);
    std::vector<double> v(params.size(), 0.0);

    for (int step = 0; step < num_steps; step++) {
        std::size_t doc_idx = static_cast<std::size_t>(step) % docs.size();
        const std::string& doc = docs[doc_idx];

        std::vector<int> tokens;
        tokens.reserve(doc.size() + 2);
        tokens.push_back(BOS);
        for (char ch : doc) tokens.push_back(char_to_token[static_cast<unsigned char>(ch)]);
        tokens.push_back(BOS);

        int n = std::min(block_size, static_cast<int>(tokens.size()) - 1);

        KVCache keys(static_cast<std::size_t>(n_layer));
        KVCache values(static_cast<std::size_t>(n_layer));
        std::vector<V> losses;
        losses.reserve(static_cast<std::size_t>(n));

        for (int pos_id = 0; pos_id < n; pos_id++) {
            int token_id = tokens[static_cast<std::size_t>(pos_id)];
            int target_id = tokens[static_cast<std::size_t>(pos_id + 1)];

            std::vector<V> logits = gpt(token_id, pos_id, keys, values, state_dict);
            std::vector<V> probs = softmax(logits);
            V loss_t = -(probs[static_cast<std::size_t>(target_id)]->log());
            losses.push_back(loss_t);
        }

        V loss = val(0.0);
        for (const auto& loss_t : losses) loss = loss + loss_t;
        loss = (1.0 / n) * loss;

        loss->backward();

        double lr_t = learning_rate * (1.0 - static_cast<double>(step) / num_steps);
        for (size_t i = 0; i < params.size(); i++) {
            V p = params[i];
            m[i] = beta1 * m[i] + (1.0 - beta1) * p->grad;
            v[i] = beta2 * v[i] + (1.0 - beta2) * p->grad * p->grad;
            double m_hat = m[i] / (1.0 - std::pow(beta1, step + 1));
            double v_hat = v[i] / (1.0 - std::pow(beta2, step + 1));
            p->data -= lr_t * m_hat / (std::sqrt(v_hat) + eps_adam);
            p->grad = 0.0;
        }

        std::cout << "\rstep " << std::setw(4) << step + 1 << " / " << std::setw(4) << num_steps
                  << " | loss " << std::fixed << std::setprecision(4) << loss->data << std::flush;
        if (step + 1 == num_steps) std::cout << "\n";
    }

    double temperature = 0.5;
    std::cout << "\n--- inference (new, hallucinated names) ---\n";
    for (int sample_idx = 0; sample_idx < 20; sample_idx++) {
        KVCache keys(static_cast<std::size_t>(n_layer));
        KVCache values(static_cast<std::size_t>(n_layer));

        int token_id = BOS;
        std::string sample;

        for (int pos_id = 0; pos_id < block_size; pos_id++) {
            std::vector<V> logits = gpt(token_id, pos_id, keys, values, state_dict);
            std::vector<V> scaled_logits;
            scaled_logits.reserve(logits.size());
            for (const auto& l : logits) scaled_logits.push_back(l / temperature);
            std::vector<V> probs = softmax(scaled_logits);

            std::vector<double> weights;
            weights.reserve(probs.size());
            for (const auto& p : probs) weights.push_back(p->data);

            std::discrete_distribution<int> dist(weights.begin(), weights.end());
            token_id = dist(rng);
            if (token_id == BOS) break;
            sample.push_back(uchars[static_cast<std::size_t>(token_id)]);
        }

        std::cout << "sample " << std::setw(2) << sample_idx + 1 << ": " << sample << "\n";
    }

    return 0;
}

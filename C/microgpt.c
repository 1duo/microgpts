#include <ctype.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define PI 3.14159265358979323846
#define ARENA_CAPACITY (128ULL * 1024ULL * 1024ULL)
#define TOPO_CAPACITY 1000000
#define N_EMBD 16
#define N_HEAD 4
#define N_LAYER 1
#define BLOCK_SIZE 16
#define NUM_STEPS 1000
#define HEAD_DIM (N_EMBD / N_HEAD)
_Static_assert(N_EMBD % N_HEAD == 0, "N_EMBD must be divisible by N_HEAD");

typedef struct {
    unsigned char *memory;
    size_t offset;
    size_t capacity;
} Arena;

typedef struct Value {
    double data;
    double grad;
    struct Value *children[2];
    double local_grads[2];
    int n_children;
    int seen_id;
} Value;

typedef struct {
    int rows;
    int cols;
    Value **data;
} Matrix;

typedef struct {
    Matrix wte;
    Matrix wpe;
    Matrix lm_head;
    Matrix *attn_wq;
    Matrix *attn_wk;
    Matrix *attn_wv;
    Matrix *attn_wo;
    Matrix *mlp_fc1;
    Matrix *mlp_fc2;
    Value **params;
    int n_params;
} StateDict;

static uint64_t rng_state = 42;
static int visit_id = 1;

static void random_seed(uint64_t seed) { rng_state = seed; }

static void *xmalloc(size_t n) {
    void *p = malloc(n);
    if (!p) {
        fprintf(stderr, "allocation failed\n");
        exit(1);
    }
    return p;
}

static void *xcalloc(size_t n, size_t size) {
    void *p = calloc(n, size);
    if (!p) {
        fprintf(stderr, "allocation failed\n");
        exit(1);
    }
    return p;
}

static void *xrealloc(void *ptr, size_t n) {
    void *p = realloc(ptr, n);
    if (!p) {
        fprintf(stderr, "allocation failed\n");
        exit(1);
    }
    return p;
}

static double random_f64(void) {
    uint64_t x = rng_state;
    x ^= x >> 12;
    x ^= x << 25;
    x ^= x >> 27;
    rng_state = x;
    return (double)(x * 0x2545F4914F6CDD1DULL) * (1.0 / 18446744073709551616.0);
}

static double random_gauss(double mean, double std) {
    double u1 = random_f64();
    double u2 = random_f64();
    if (u1 < 1e-12) u1 = 1e-12;
    double z0 = sqrt(-2.0 * log(u1)) * cos(2.0 * PI * u2);
    return mean + std * z0;
}

static void random_shuffle(char **arr, int n) {
    for (int i = n - 1; i > 0; i--) {
        int j = (int)(random_f64() * (i + 1));
        char *tmp = arr[i];
        arr[i] = arr[j];
        arr[j] = tmp;
    }
}

static int random_choices(int n, const double *weights) {
    double total = 0.0;
    for (int i = 0; i < n; i++) total += weights[i];
    double r = random_f64() * total;
    for (int i = 0; i < n; i++) {
        r -= weights[i];
        if (r <= 0.0) return i;
    }
    return n - 1;
}

static Arena arena_init(size_t capacity) {
    Arena arena;
    arena.memory = (unsigned char *)xmalloc(capacity);
    arena.offset = 0;
    arena.capacity = capacity;
    return arena;
}

static void arena_reset(Arena *arena) { arena->offset = 0; }

static void *arena_alloc(Arena *arena, size_t size) {
    size_t aligned = (size + 7) & ~((size_t)7);
    if (arena->offset + aligned > arena->capacity) {
        fprintf(stderr, "arena out of memory\n");
        exit(1);
    }
    void *ptr = arena->memory + arena->offset;
    arena->offset += aligned;
    return ptr;
}

static Value *value_new(Arena *arena, double data) {
    Value *v = (Value *)arena_alloc(arena, sizeof(Value));
    v->data = data;
    v->grad = 0.0;
    v->children[0] = NULL;
    v->children[1] = NULL;
    v->local_grads[0] = 0.0;
    v->local_grads[1] = 0.0;
    v->n_children = 0;
    v->seen_id = 0;
    return v;
}

static Value *param_new(double data) {
    Value *v = (Value *)xmalloc(sizeof(Value));
    v->data = data;
    v->grad = 0.0;
    v->children[0] = NULL;
    v->children[1] = NULL;
    v->local_grads[0] = 0.0;
    v->local_grads[1] = 0.0;
    v->n_children = 0;
    v->seen_id = 0;
    return v;
}

static Value *add(Arena *arena, Value *lhs, Value *rhs) {
    Value *out = value_new(arena, lhs->data + rhs->data);
    out->children[0] = lhs;
    out->children[1] = rhs;
    out->local_grads[0] = 1.0;
    out->local_grads[1] = 1.0;
    out->n_children = 2;
    return out;
}

static Value *mul(Arena *arena, Value *lhs, Value *rhs) {
    Value *out = value_new(arena, lhs->data * rhs->data);
    out->children[0] = lhs;
    out->children[1] = rhs;
    out->local_grads[0] = rhs->data;
    out->local_grads[1] = lhs->data;
    out->n_children = 2;
    return out;
}

static Value *pow_op(Arena *arena, Value *base, double exponent) {
    Value *out = value_new(arena, pow(base->data, exponent));
    out->children[0] = base;
    out->local_grads[0] = exponent * pow(base->data, exponent - 1.0);
    out->n_children = 1;
    return out;
}

static Value *log_op(Arena *arena, Value *x) {
    Value *out = value_new(arena, log(x->data));
    out->children[0] = x;
    out->local_grads[0] = 1.0 / x->data;
    out->n_children = 1;
    return out;
}

static Value *exp_op(Arena *arena, Value *x) {
    double ex = exp(x->data);
    Value *out = value_new(arena, ex);
    out->children[0] = x;
    out->local_grads[0] = ex;
    out->n_children = 1;
    return out;
}

static Value *relu(Arena *arena, Value *x) {
    Value *out = value_new(arena, x->data > 0.0 ? x->data : 0.0);
    out->children[0] = x;
    out->local_grads[0] = x->data > 0.0 ? 1.0 : 0.0;
    out->n_children = 1;
    return out;
}

static Value *neg(Arena *arena, Value *x) { return mul(arena, x, value_new(arena, -1.0)); }
static Value *sub(Arena *arena, Value *lhs, Value *rhs) { return add(arena, lhs, neg(arena, rhs)); }
static Value *div_op(Arena *arena, Value *lhs, Value *rhs) { return mul(arena, lhs, pow_op(arena, rhs, -1.0)); }
static Value *div_f(Arena *arena, Value *lhs, double rhs) { return mul(arena, lhs, value_new(arena, 1.0 / rhs)); }

static void build_topo(Value *v, Value **topo, int *n, int seen_id) {
    if (v->seen_id == seen_id) return;
    v->seen_id = seen_id;
    for (int i = 0; i < v->n_children; i++) {
        build_topo(v->children[i], topo, n, seen_id);
    }
    if (*n >= TOPO_CAPACITY) {
        fprintf(stderr, "topo capacity exceeded\n");
        exit(1);
    }
    topo[(*n)++] = v;
}

static void backward(Value *root) {
    static Value *topo[TOPO_CAPACITY];
    int topo_n = 0;
    int seen_id = visit_id++;
    build_topo(root, topo, &topo_n, seen_id);
    root->grad = 1.0;
    for (int i = topo_n - 1; i >= 0; i--) {
        Value *v = topo[i];
        for (int j = 0; j < v->n_children; j++) {
            v->children[j]->grad += v->local_grads[j] * v->grad;
        }
    }
}

static Matrix matrix(int nout, int nin, double std) {
    Matrix m;
    m.rows = nout;
    m.cols = nin;
    m.data = (Value **)xmalloc(sizeof(Value *) * (size_t)nout * (size_t)nin);
    for (int i = 0; i < nout * nin; i++) {
        m.data[i] = param_new(random_gauss(0.0, std));
    }
    return m;
}

static void append_params(StateDict *state_dict, Matrix *m) {
    int count = m->rows * m->cols;
    state_dict->params = (Value **)xrealloc(
        state_dict->params, sizeof(Value *) * (size_t)(state_dict->n_params + count)
    );
    memcpy(&state_dict->params[state_dict->n_params], m->data, sizeof(Value *) * (size_t)count);
    state_dict->n_params += count;
}

static Value **linear(Arena *arena, Value **x, Matrix *w) {
    int nout = w->rows;
    int nin = w->cols;
    Value **out = (Value **)arena_alloc(arena, sizeof(Value *) * (size_t)nout);
    for (int i = 0; i < nout; i++) {
        Value *acc = value_new(arena, 0.0);
        for (int j = 0; j < nin; j++) {
            acc = add(arena, acc, mul(arena, w->data[i * nin + j], x[j]));
        }
        out[i] = acc;
    }
    return out;
}

static Value **softmax(Arena *arena, Value **logits, int n) {
    double max_val = logits[0]->data;
    for (int i = 1; i < n; i++) {
        if (logits[i]->data > max_val) max_val = logits[i]->data;
    }
    Value **exps = (Value **)arena_alloc(arena, sizeof(Value *) * (size_t)n);
    Value *total = value_new(arena, 0.0);
    for (int i = 0; i < n; i++) {
        exps[i] = exp_op(arena, sub(arena, logits[i], value_new(arena, max_val)));
        total = add(arena, total, exps[i]);
    }
    Value **out = (Value **)arena_alloc(arena, sizeof(Value *) * (size_t)n);
    for (int i = 0; i < n; i++) out[i] = div_op(arena, exps[i], total);
    return out;
}

static Value **rmsnorm(Arena *arena, Value **x, int n) {
    Value *ms = value_new(arena, 0.0);
    for (int i = 0; i < n; i++) ms = add(arena, ms, mul(arena, x[i], x[i]));
    ms = div_f(arena, ms, (double)n);
    Value *scale = pow_op(arena, add(arena, ms, value_new(arena, 1e-5)), -0.5);
    Value **out = (Value **)arena_alloc(arena, sizeof(Value *) * (size_t)n);
    for (int i = 0; i < n; i++) out[i] = mul(arena, x[i], scale);
    return out;
}

static Value **gpt(
    Arena *arena,
    StateDict *state_dict,
    int token_id,
    int pos_id,
    Value ****keys,
    Value ****values
) {
    Value **tok_emb = &state_dict->wte.data[token_id * N_EMBD];
    Value **pos_emb = &state_dict->wpe.data[pos_id * N_EMBD];
    Value **x = (Value **)arena_alloc(arena, sizeof(Value *) * (size_t)N_EMBD);
    for (int i = 0; i < N_EMBD; i++) x[i] = add(arena, tok_emb[i], pos_emb[i]);
    x = rmsnorm(arena, x, N_EMBD);

    for (int li = 0; li < N_LAYER; li++) {
        Value **x_residual = x;
        x = rmsnorm(arena, x, N_EMBD);

        Value **q = linear(arena, x, &state_dict->attn_wq[li]);
        Value **k = linear(arena, x, &state_dict->attn_wk[li]);
        Value **v = linear(arena, x, &state_dict->attn_wv[li]);
        keys[li][pos_id] = k;
        values[li][pos_id] = v;

        int seq_len = pos_id + 1;
        Value **x_attn = (Value **)arena_alloc(arena, sizeof(Value *) * (size_t)N_EMBD);

        for (int h = 0; h < N_HEAD; h++) {
            int hs = h * HEAD_DIM;
            Value **attn_logits = (Value **)arena_alloc(arena, sizeof(Value *) * (size_t)seq_len);
            for (int t = 0; t < seq_len; t++) {
                Value *dot = value_new(arena, 0.0);
                for (int j = 0; j < HEAD_DIM; j++) {
                    dot = add(arena, dot, mul(arena, q[hs + j], keys[li][t][hs + j]));
                }
                attn_logits[t] = div_f(arena, dot, sqrt((double)HEAD_DIM));
            }
            Value **attn_weights = softmax(arena, attn_logits, seq_len);
            for (int j = 0; j < HEAD_DIM; j++) {
                Value *head_out = value_new(arena, 0.0);
                for (int t = 0; t < seq_len; t++) {
                    head_out = add(arena, head_out, mul(arena, attn_weights[t], values[li][t][hs + j]));
                }
                x_attn[hs + j] = head_out;
            }
        }

        x = linear(arena, x_attn, &state_dict->attn_wo[li]);
        for (int i = 0; i < N_EMBD; i++) x[i] = add(arena, x[i], x_residual[i]);

        x_residual = x;
        x = rmsnorm(arena, x, N_EMBD);
        x = linear(arena, x, &state_dict->mlp_fc1[li]);
        for (int i = 0; i < 4 * N_EMBD; i++) x[i] = relu(arena, x[i]);
        x = linear(arena, x, &state_dict->mlp_fc2[li]);
        for (int i = 0; i < N_EMBD; i++) x[i] = add(arena, x[i], x_residual[i]);
    }

    return linear(arena, x, &state_dict->lm_head);
}

static char *read_file(const char *path) {
    FILE *f = fopen(path, "rb");
    if (!f) return NULL;
    if (fseek(f, 0, SEEK_END) != 0) {
        fclose(f);
        return NULL;
    }
    long sz = ftell(f);
    if (sz < 0 || fseek(f, 0, SEEK_SET) != 0) {
        fclose(f);
        return NULL;
    }
    char *buf = (char *)xmalloc((size_t)sz + 1);
    if (fread(buf, 1, (size_t)sz, f) != (size_t)sz) {
        fclose(f);
        free(buf);
        return NULL;
    }
    buf[sz] = '\0';
    fclose(f);
    return buf;
}

static char *trim(char *s) {
    while (*s && isspace((unsigned char)*s)) s++;
    char *end = s + strlen(s);
    while (end > s && isspace((unsigned char)end[-1])) end--;
    *end = '\0';
    return s;
}

int main(void) {
    random_seed(42);

    char *content = read_file("input.txt");
    if (!content) {
        fprintf(stderr, "failed to read input.txt\n");
        return 1;
    }

    int docs_cap = 32768;
    char **docs = (char **)xmalloc(sizeof(char *) * (size_t)docs_cap);
    int n_docs = 0;
    char *line = strtok(content, "\n");
    while (line) {
        char *doc = trim(line);
        if (*doc) {
            if (n_docs == docs_cap) {
                docs_cap *= 2;
                docs = (char **)xrealloc(docs, sizeof(char *) * (size_t)docs_cap);
            }
            docs[n_docs++] = doc;
        }
        line = strtok(NULL, "\n");
    }
    if (n_docs == 0) {
        fprintf(stderr, "input.txt has no non-empty lines\n");
        return 1;
    }
    random_shuffle(docs, n_docs);
    printf("num docs: %d\n", n_docs);

    int present[256] = {0};
    for (int i = 0; i < n_docs; i++) {
        for (char *p = docs[i]; *p; p++) present[(unsigned char)*p] = 1;
    }
    char uchars[256];
    int n_uchars = 0;
    for (int i = 0; i < 256; i++) {
        if (present[i]) uchars[n_uchars++] = (char)i;
    }
    int BOS = n_uchars;
    int vocab_size = n_uchars + 1;
    printf("vocab size: %d\n", vocab_size);

    int char_to_token[256];
    for (int i = 0; i < 256; i++) char_to_token[i] = -1;
    for (int i = 0; i < n_uchars; i++) char_to_token[(unsigned char)uchars[i]] = i;

    StateDict state_dict = {0};
    state_dict.wte = matrix(vocab_size, N_EMBD, 0.08);
    state_dict.wpe = matrix(BLOCK_SIZE, N_EMBD, 0.08);
    state_dict.lm_head = matrix(vocab_size, N_EMBD, 0.08);
    append_params(&state_dict, &state_dict.wte);
    append_params(&state_dict, &state_dict.wpe);
    append_params(&state_dict, &state_dict.lm_head);

    state_dict.attn_wq = (Matrix *)xmalloc(sizeof(Matrix) * (size_t)N_LAYER);
    state_dict.attn_wk = (Matrix *)xmalloc(sizeof(Matrix) * (size_t)N_LAYER);
    state_dict.attn_wv = (Matrix *)xmalloc(sizeof(Matrix) * (size_t)N_LAYER);
    state_dict.attn_wo = (Matrix *)xmalloc(sizeof(Matrix) * (size_t)N_LAYER);
    state_dict.mlp_fc1 = (Matrix *)xmalloc(sizeof(Matrix) * (size_t)N_LAYER);
    state_dict.mlp_fc2 = (Matrix *)xmalloc(sizeof(Matrix) * (size_t)N_LAYER);

    for (int i = 0; i < N_LAYER; i++) {
        state_dict.attn_wq[i] = matrix(N_EMBD, N_EMBD, 0.08);
        state_dict.attn_wk[i] = matrix(N_EMBD, N_EMBD, 0.08);
        state_dict.attn_wv[i] = matrix(N_EMBD, N_EMBD, 0.08);
        state_dict.attn_wo[i] = matrix(N_EMBD, N_EMBD, 0.08);
        state_dict.mlp_fc1[i] = matrix(4 * N_EMBD, N_EMBD, 0.08);
        state_dict.mlp_fc2[i] = matrix(N_EMBD, 4 * N_EMBD, 0.08);
        append_params(&state_dict, &state_dict.attn_wq[i]);
        append_params(&state_dict, &state_dict.attn_wk[i]);
        append_params(&state_dict, &state_dict.attn_wv[i]);
        append_params(&state_dict, &state_dict.attn_wo[i]);
        append_params(&state_dict, &state_dict.mlp_fc1[i]);
        append_params(&state_dict, &state_dict.mlp_fc2[i]);
    }
    printf("num params: %d\n", state_dict.n_params);

    double learning_rate = 0.01;
    double beta1 = 0.85;
    double beta2 = 0.99;
    double eps_adam = 1e-8;
    double *m = (double *)xcalloc((size_t)state_dict.n_params, sizeof(double));
    double *v = (double *)xcalloc((size_t)state_dict.n_params, sizeof(double));

    Value ****keys = (Value ****)xmalloc(sizeof(Value ***) * (size_t)N_LAYER);
    Value ****values = (Value ****)xmalloc(sizeof(Value ***) * (size_t)N_LAYER);
    for (int li = 0; li < N_LAYER; li++) {
        keys[li] = (Value ***)xmalloc(sizeof(Value **) * (size_t)BLOCK_SIZE);
        values[li] = (Value ***)xmalloc(sizeof(Value **) * (size_t)BLOCK_SIZE);
    }

    Arena arena = arena_init(ARENA_CAPACITY);

    for (int step = 0; step < NUM_STEPS; step++) {
        arena_reset(&arena);

        char *doc = docs[step % n_docs];
        int doc_len = (int)strlen(doc);
        int *tokens = (int *)arena_alloc(&arena, sizeof(int) * (size_t)(doc_len + 2));
        tokens[0] = BOS;
        for (int i = 0; i < doc_len; i++) tokens[i + 1] = char_to_token[(unsigned char)doc[i]];
        tokens[doc_len + 1] = BOS;

        int n = doc_len + 1;
        if (n > BLOCK_SIZE) n = BLOCK_SIZE;

        Value **losses = (Value **)arena_alloc(&arena, sizeof(Value *) * (size_t)n);
        for (int pos_id = 0; pos_id < n; pos_id++) {
            int token_id = tokens[pos_id];
            int target_id = tokens[pos_id + 1];
            Value **logits = gpt(&arena, &state_dict, token_id, pos_id, keys, values);
            Value **probs = softmax(&arena, logits, vocab_size);
            losses[pos_id] = neg(&arena, log_op(&arena, probs[target_id]));
        }

        Value *loss = value_new(&arena, 0.0);
        for (int i = 0; i < n; i++) loss = add(&arena, loss, losses[i]);
        loss = mul(&arena, value_new(&arena, 1.0 / n), loss);

        backward(loss);

        double lr_t = learning_rate * (1.0 - (double)step / (double)NUM_STEPS);
        for (int i = 0; i < state_dict.n_params; i++) {
            Value *p = state_dict.params[i];
            m[i] = beta1 * m[i] + (1.0 - beta1) * p->grad;
            v[i] = beta2 * v[i] + (1.0 - beta2) * p->grad * p->grad;
            double m_hat = m[i] / (1.0 - pow(beta1, step + 1));
            double v_hat = v[i] / (1.0 - pow(beta2, step + 1));
            p->data -= lr_t * m_hat / (sqrt(v_hat) + eps_adam);
            p->grad = 0.0;
        }

        printf("\rstep %4d / %4d | loss %.4f", step + 1, NUM_STEPS, loss->data);
        fflush(stdout);
        if (step + 1 == NUM_STEPS) {
            printf("\n");
        }
    }

    double temperature = 0.5;
    printf("\n--- inference (new, hallucinated names) ---\n");
    for (int sample_idx = 0; sample_idx < 20; sample_idx++) {
        arena_reset(&arena);
        int token_id = BOS;
        char sample[BLOCK_SIZE + 1];
        int sample_len = 0;

        for (int pos_id = 0; pos_id < BLOCK_SIZE; pos_id++) {
            Value **logits = gpt(&arena, &state_dict, token_id, pos_id, keys, values);
            Value **scaled_logits = (Value **)arena_alloc(&arena, sizeof(Value *) * (size_t)vocab_size);
            for (int i = 0; i < vocab_size; i++) scaled_logits[i] = div_f(&arena, logits[i], temperature);
            Value **probs = softmax(&arena, scaled_logits, vocab_size);

            double *weights = (double *)arena_alloc(&arena, sizeof(double) * (size_t)vocab_size);
            for (int i = 0; i < vocab_size; i++) weights[i] = probs[i]->data;
            token_id = random_choices(vocab_size, weights);
            if (token_id == BOS) break;
            sample[sample_len++] = uchars[token_id];
        }

        sample[sample_len] = '\0';
        printf("sample %2d: %s\n", sample_idx + 1, sample);
    }

    return 0;
}

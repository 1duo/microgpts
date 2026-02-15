import java.io.File
import java.util.Random
import kotlin.math.exp
import kotlin.math.ln
import kotlin.math.max
import kotlin.math.pow
import kotlin.math.sqrt

const val N_EMBD = 16
const val N_HEAD = 4
const val N_LAYER = 1
const val BLOCK_SIZE = 16
const val NUM_STEPS = 1000
const val HEAD_DIM = N_EMBD / N_HEAD

class Value {
    var data: Double
    var grad = 0.0
    private val children = arrayOfNulls<Value>(2)
    private val localGrads = DoubleArray(2)
    private var nChildren = 0

    constructor(data: Double) {
        this.data = data
    }

    constructor(data: Double, c0: Value, g0: Double) {
        this.data = data
        children[0] = c0
        localGrads[0] = g0
        nChildren = 1
    }

    constructor(data: Double, c0: Value, g0: Double, c1: Value, g1: Double) {
        this.data = data
        children[0] = c0
        children[1] = c1
        localGrads[0] = g0
        localGrads[1] = g1
        nChildren = 2
    }

    operator fun plus(other: Value): Value = Value(data + other.data, this, 1.0, other, 1.0)
    operator fun plus(other: Double): Value = this + Value(other)

    operator fun times(other: Value): Value = Value(data * other.data, this, other.data, other, data)
    operator fun times(other: Double): Value = this * Value(other)

    fun pow(other: Double): Value = Value(data.pow(other), this, other * data.pow(other - 1.0))
    fun log(): Value = Value(ln(data), this, 1.0 / data)

    fun exp(): Value {
        val ex = exp(data)
        return Value(ex, this, ex)
    }

    fun relu(): Value = Value(max(0.0, data), this, if (data > 0.0) 1.0 else 0.0)

    operator fun unaryMinus(): Value = this * -1.0
    operator fun minus(other: Value): Value = this + (-other)
    operator fun minus(other: Double): Value = this + (-other)
    operator fun div(other: Value): Value = this * other.pow(-1.0)
    operator fun div(other: Double): Value = this * other.pow(-1.0)

    fun backward() {
        val topo = mutableListOf<Value>()
        val visited = HashSet<Value>()

        fun buildTopo(v: Value) {
            if (visited.add(v)) {
                for (i in 0 until v.nChildren) {
                    buildTopo(v.children[i]!!)
                }
                topo.add(v)
            }
        }

        buildTopo(this)
        grad = 1.0

        for (i in topo.indices.reversed()) {
            val v = topo[i]
            for (j in 0 until v.nChildren) {
                v.children[j]!!.grad += v.localGrads[j] * v.grad
            }
        }
    }
}

typealias Matrix = MutableList<MutableList<Value>>

data class Layer(
    val attnWq: Matrix,
    val attnWk: Matrix,
    val attnWv: Matrix,
    val attnWo: Matrix,
    val mlpFc1: Matrix,
    val mlpFc2: Matrix
)

data class StateDict(
    val wte: Matrix,
    val wpe: Matrix,
    val lmHead: Matrix,
    val layers: MutableList<Layer>
)

fun matrix(nout: Int, nin: Int, rng: Random, std: Double = 0.08): Matrix {
    return MutableList(nout) { MutableList(nin) { Value(rng.nextGaussian() * std) } }
}

fun linear(x: List<Value>, w: Matrix): MutableList<Value> {
    return w.map { row -> row.zip(x).fold(Value(0.0)) { acc, (wi, xi) -> acc + (wi * xi) } }.toMutableList()
}

fun softmax(logits: List<Value>): MutableList<Value> {
    val maxVal = logits.maxOf { it.data }
    val exps = logits.map { (it - maxVal).exp() }
    val total = exps.fold(Value(0.0)) { acc, e -> acc + e }
    return exps.map { it / total }.toMutableList()
}

fun rmsnorm(x: List<Value>): MutableList<Value> {
    val ms = x.fold(Value(0.0)) { acc, xi -> acc + (xi * xi) } / x.size.toDouble()
    val scale = (ms + 1e-5).pow(-0.5)
    return x.map { it * scale }.toMutableList()
}

fun randomChoices(rng: Random, weights: DoubleArray): Int {
    var total = 0.0
    for (w in weights) total += w
    var r = rng.nextDouble() * total
    for (i in weights.indices) {
        r -= weights[i]
        if (r <= 0.0) return i
    }
    return weights.size - 1
}

fun params(stateDict: StateDict): MutableList<Value> {
    val params = mutableListOf<Value>()
    fun appendMatrix(m: Matrix) {
        for (row in m) params.addAll(row)
    }

    appendMatrix(stateDict.wte)
    appendMatrix(stateDict.wpe)
    appendMatrix(stateDict.lmHead)
    for (layer in stateDict.layers) {
        appendMatrix(layer.attnWq)
        appendMatrix(layer.attnWk)
        appendMatrix(layer.attnWv)
        appendMatrix(layer.attnWo)
        appendMatrix(layer.mlpFc1)
        appendMatrix(layer.mlpFc2)
    }
    return params
}

fun gpt(
    tokenId: Int,
    posId: Int,
    keys: MutableList<MutableList<MutableList<Value>>>,
    values: MutableList<MutableList<MutableList<Value>>>,
    stateDict: StateDict
): MutableList<Value> {
    val tokEmb = stateDict.wte[tokenId]
    val posEmb = stateDict.wpe[posId]

    var x = MutableList(N_EMBD) { i -> tokEmb[i] + posEmb[i] }
    x = rmsnorm(x)

    for (li in 0 until N_LAYER) {
        var xResidual = x
        x = rmsnorm(x)

        val layer = stateDict.layers[li]
        val q = linear(x, layer.attnWq)
        val k = linear(x, layer.attnWk)
        val v = linear(x, layer.attnWv)

        keys[li].add(k)
        values[li].add(v)

        val xAttn = mutableListOf<Value>()
        for (h in 0 until N_HEAD) {
            val hs = h * HEAD_DIM
            val attnLogits = MutableList(keys[li].size) { Value(0.0) }
            for (t in keys[li].indices) {
                var dot = Value(0.0)
                for (j in 0 until HEAD_DIM) dot += q[hs + j] * keys[li][t][hs + j]
                attnLogits[t] = dot / sqrt(HEAD_DIM.toDouble())
            }

            val attnWeights = softmax(attnLogits)
            for (j in 0 until HEAD_DIM) {
                var outJ = Value(0.0)
                for (t in values[li].indices) outJ += attnWeights[t] * values[li][t][hs + j]
                xAttn.add(outJ)
            }
        }

        x = linear(xAttn, layer.attnWo)
        x = MutableList(N_EMBD) { i -> x[i] + xResidual[i] }

        xResidual = x
        x = rmsnorm(x)
        x = linear(x, layer.mlpFc1)
        x = x.map { it.relu() }.toMutableList()
        x = linear(x, layer.mlpFc2)
        x = MutableList(N_EMBD) { i -> x[i] + xResidual[i] }
    }

    return linear(x, stateDict.lmHead)
}

fun main() {
    val rng = Random(42)
    val docs = File("input.txt").readLines().map { it.trim() }.filter { it.isNotEmpty() }.toMutableList()
    docs.shuffle(rng)
    println("num docs: ${docs.size}")

    val uchars = docs.joinToString("").toSet().toList().sorted()
    val charToToken = uchars.withIndex().associate { it.value to it.index }
    val BOS = uchars.size
    val vocabSize = BOS + 1
    println("vocab size: $vocabSize")

    val layers = MutableList(N_LAYER) {
        Layer(
            matrix(N_EMBD, N_EMBD, rng),
            matrix(N_EMBD, N_EMBD, rng),
            matrix(N_EMBD, N_EMBD, rng),
            matrix(N_EMBD, N_EMBD, rng),
            matrix(4 * N_EMBD, N_EMBD, rng),
            matrix(N_EMBD, 4 * N_EMBD, rng)
        )
    }
    val stateDict = StateDict(
        matrix(vocabSize, N_EMBD, rng),
        matrix(BLOCK_SIZE, N_EMBD, rng),
        matrix(vocabSize, N_EMBD, rng),
        layers
    )

    val params = params(stateDict)
    println("num params: ${params.size}")

    val learningRate = 0.01
    val beta1 = 0.85
    val beta2 = 0.99
    val epsAdam = 1e-8
    val m = DoubleArray(params.size)
    val v = DoubleArray(params.size)

    for (step in 0 until NUM_STEPS) {
        val doc = docs[step % docs.size]
        val tokens = mutableListOf(BOS)
        for (ch in doc) tokens.add(charToToken[ch]!!)
        tokens.add(BOS)

        val n = minOf(BLOCK_SIZE, tokens.size - 1)
        val keys = MutableList(N_LAYER) { mutableListOf<MutableList<Value>>() }
        val values = MutableList(N_LAYER) { mutableListOf<MutableList<Value>>() }
        val losses = mutableListOf<Value>()

        for (posId in 0 until n) {
            val tokenId = tokens[posId]
            val targetId = tokens[posId + 1]
            val logits = gpt(tokenId, posId, keys, values, stateDict)
            val probs = softmax(logits)
            losses.add(-probs[targetId].log())
        }

        var loss = losses.fold(Value(0.0)) { acc, l -> acc + l }
        loss *= 1.0 / n.toDouble()
        loss.backward()

        val lrT = learningRate * (1.0 - step.toDouble() / NUM_STEPS)
        for (i in params.indices) {
            val p = params[i]
            m[i] = beta1 * m[i] + (1.0 - beta1) * p.grad
            v[i] = beta2 * v[i] + (1.0 - beta2) * p.grad.pow(2.0)
            val mHat = m[i] / (1.0 - beta1.pow(step + 1.0))
            val vHat = v[i] / (1.0 - beta2.pow(step + 1.0))
            p.data -= lrT * mHat / (sqrt(vHat) + epsAdam)
            p.grad = 0.0
        }

        print("\rstep %4d / %4d | loss %.4f".format(step + 1, NUM_STEPS, loss.data))
        System.out.flush()
        if (step + 1 == NUM_STEPS) println()
    }

    val temperature = 0.5
    println("\n--- inference (new, hallucinated names) ---")
    for (sampleIdx in 0 until 20) {
        val keys = MutableList(N_LAYER) { mutableListOf<MutableList<Value>>() }
        val values = MutableList(N_LAYER) { mutableListOf<MutableList<Value>>() }

        var tokenId = BOS
        val sample = StringBuilder()
        for (posId in 0 until BLOCK_SIZE) {
            val logits = gpt(tokenId, posId, keys, values, stateDict)
            val probs = softmax(logits.map { it / temperature })
            val weights = DoubleArray(vocabSize) { i -> probs[i].data }
            tokenId = randomChoices(rng, weights)
            if (tokenId == BOS) break
            sample.append(uchars[tokenId])
        }

        println("sample %2d: %s".format(sampleIdx + 1, sample.toString()))
    }
}

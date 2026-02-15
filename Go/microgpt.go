package main

import (
	"fmt"
	"math"
	"math/rand"
	"os"
	"sort"
	"strings"
)

const (
	nEmbd     = 16
	nHead     = 4
	nLayer    = 1
	blockSize = 16
	numSteps  = 1000
	headDim   = nEmbd / nHead
)

type Value struct {
	data       float64
	grad       float64
	children   [2]*Value
	localGrads [2]float64
	nChildren  int
}

func newValue(data float64) *Value { return &Value{data: data} }

func newUnary(data float64, c0 *Value, g0 float64) *Value {
	out := newValue(data)
	out.children[0] = c0
	out.localGrads[0] = g0
	out.nChildren = 1
	return out
}

func newBinary(data float64, c0 *Value, g0 float64, c1 *Value, g1 float64) *Value {
	out := newValue(data)
	out.children[0], out.children[1] = c0, c1
	out.localGrads[0], out.localGrads[1] = g0, g1
	out.nChildren = 2
	return out
}

func (v *Value) add(other *Value) *Value { return newBinary(v.data+other.data, v, 1.0, other, 1.0) }
func (v *Value) mul(other *Value) *Value {
	return newBinary(v.data*other.data, v, other.data, other, v.data)
}
func (v *Value) pow(exponent float64) *Value {
	return newUnary(math.Pow(v.data, exponent), v, exponent*math.Pow(v.data, exponent-1.0))
}
func (v *Value) log() *Value { return newUnary(math.Log(v.data), v, 1.0/v.data) }
func (v *Value) exp() *Value {
	ex := math.Exp(v.data)
	return newUnary(ex, v, ex)
}
func (v *Value) relu() *Value {
	if v.data > 0.0 {
		return newUnary(v.data, v, 1.0)
	}
	return newUnary(0.0, v, 0.0)
}
func (v *Value) neg() *Value             { return v.mul(newValue(-1.0)) }
func (v *Value) div(other *Value) *Value { return v.mul(other.pow(-1.0)) }

func (v *Value) backward() {
	topo := make([]*Value, 0)
	visited := make(map[*Value]struct{})
	var buildTopo func(*Value)
	buildTopo = func(node *Value) {
		if _, ok := visited[node]; ok {
			return
		}
		visited[node] = struct{}{}
		for i := 0; i < node.nChildren; i++ {
			buildTopo(node.children[i])
		}
		topo = append(topo, node)
	}
	buildTopo(v)
	v.grad = 1.0
	for i := len(topo) - 1; i >= 0; i-- {
		node := topo[i]
		for j := 0; j < node.nChildren; j++ {
			node.children[j].grad += node.localGrads[j] * node.grad
		}
	}
}

type Matrix [][]*Value

type Layer struct {
	attnWQ Matrix
	attnWK Matrix
	attnWV Matrix
	attnWO Matrix
	mlpFC1 Matrix
	mlpFC2 Matrix
}

type StateDict struct {
	wte    Matrix
	wpe    Matrix
	lmHead Matrix
	layers []Layer
}

func params(stateDict *StateDict) []*Value {
	params := make([]*Value, 0)
	appendMatrix := func(m Matrix) {
		for _, row := range m {
			params = append(params, row...)
		}
	}
	appendMatrix(stateDict.wte)
	appendMatrix(stateDict.wpe)
	appendMatrix(stateDict.lmHead)
	for _, layer := range stateDict.layers {
		appendMatrix(layer.attnWQ)
		appendMatrix(layer.attnWK)
		appendMatrix(layer.attnWV)
		appendMatrix(layer.attnWO)
		appendMatrix(layer.mlpFC1)
		appendMatrix(layer.mlpFC2)
	}
	return params
}

func randomChoices(rng *rand.Rand, weights []float64) int {
	total := 0.0
	for _, w := range weights {
		total += w
	}
	r := rng.Float64() * total
	for i, w := range weights {
		r -= w
		if r <= 0.0 {
			return i
		}
	}
	return len(weights) - 1
}

func matrix(rows, cols int, rng *rand.Rand) Matrix {
	m := make(Matrix, rows)
	for i := range m {
		m[i] = make([]*Value, cols)
		for j := range m[i] {
			m[i][j] = newValue(rng.NormFloat64() * 0.08)
		}
	}
	return m
}

func linear(x []*Value, w Matrix) []*Value {
	out := make([]*Value, len(w))
	for i, row := range w {
		sum := newValue(0.0)
		for j, wi := range row {
			sum = sum.add(wi.mul(x[j]))
		}
		out[i] = sum
	}
	return out
}

func softmax(logits []*Value) []*Value {
	maxVal := logits[0].data
	for _, v := range logits[1:] {
		if v.data > maxVal {
			maxVal = v.data
		}
	}
	exps := make([]*Value, len(logits))
	total := newValue(0.0)
	for i, l := range logits {
		exps[i] = l.add(newValue(-maxVal)).exp()
		total = total.add(exps[i])
	}
	out := make([]*Value, len(logits))
	for i, e := range exps {
		out[i] = e.div(total)
	}
	return out
}

func rmsnorm(x []*Value) []*Value {
	ms := newValue(0.0)
	for _, xi := range x {
		ms = ms.add(xi.mul(xi))
	}
	ms = ms.div(newValue(float64(len(x))))
	scale := ms.add(newValue(1e-5)).pow(-0.5)
	out := make([]*Value, len(x))
	for i, xi := range x {
		out[i] = xi.mul(scale)
	}
	return out
}

func gpt(tokenID, posID int, keys, values [][][]*Value, stateDict *StateDict) []*Value {
	tokEmb := stateDict.wte[tokenID]
	posEmb := stateDict.wpe[posID]
	x := make([]*Value, len(tokEmb))
	for i := range x {
		x[i] = tokEmb[i].add(posEmb[i])
	}
	x = rmsnorm(x)

	for li, layer := range stateDict.layers {
		xResidual := x
		x = rmsnorm(x)
		q := linear(x, layer.attnWQ)
		k := linear(x, layer.attnWK)
		v := linear(x, layer.attnWV)
		keys[li] = append(keys[li], k)
		values[li] = append(values[li], v)

		xAttn := make([]*Value, 0, nEmbd)
		attnScale := newValue(1.0 / math.Sqrt(float64(headDim)))
		for h := 0; h < nHead; h++ {
			hs := h * headDim
			attnLogits := make([]*Value, len(keys[li]))
			for t := range keys[li] {
				dot := newValue(0.0)
				for j := 0; j < headDim; j++ {
					dot = dot.add(q[hs+j].mul(keys[li][t][hs+j]))
				}
				attnLogits[t] = dot.mul(attnScale)
			}
			attnWeights := softmax(attnLogits)
			for j := 0; j < headDim; j++ {
				outJ := newValue(0.0)
				for t := range values[li] {
					outJ = outJ.add(attnWeights[t].mul(values[li][t][hs+j]))
				}
				xAttn = append(xAttn, outJ)
			}
		}

		x = linear(xAttn, layer.attnWO)
		for i := range x {
			x[i] = x[i].add(xResidual[i])
		}
		xResidual = x
		x = rmsnorm(x)
		x = linear(x, layer.mlpFC1)
		for i := range x {
			x[i] = x[i].relu()
		}
		x = linear(x, layer.mlpFC2)
		for i := range x {
			x[i] = x[i].add(xResidual[i])
		}
	}

	return linear(x, stateDict.lmHead)
}

func main() {
	rng := rand.New(rand.NewSource(42))
	contentBytes, err := os.ReadFile("input.txt")
	if err != nil {
		panic(err)
	}

	docs := make([]string, 0)
	for _, line := range strings.Split(string(contentBytes), "\n") {
		line = strings.TrimSpace(line)
		if line != "" {
			docs = append(docs, line)
		}
	}
	rng.Shuffle(len(docs), func(i, j int) { docs[i], docs[j] = docs[j], docs[i] })
	fmt.Printf("num docs: %d\n", len(docs))

	charSet := make(map[rune]struct{})
	for _, doc := range docs {
		for _, ch := range doc {
			charSet[ch] = struct{}{}
		}
	}
	uchars := make([]rune, 0, len(charSet))
	for ch := range charSet {
		uchars = append(uchars, ch)
	}
	sort.Slice(uchars, func(i, j int) bool { return uchars[i] < uchars[j] })
	charToToken := make(map[rune]int, len(uchars))
	for i, ch := range uchars {
		charToToken[ch] = i
	}

	BOS := len(uchars)
	vocabSize := BOS + 1
	fmt.Printf("vocab size: %d\n", vocabSize)

	layers := make([]Layer, nLayer)
	for i := range layers {
		layers[i] = Layer{
			attnWQ: matrix(nEmbd, nEmbd, rng),
			attnWK: matrix(nEmbd, nEmbd, rng),
			attnWV: matrix(nEmbd, nEmbd, rng),
			attnWO: matrix(nEmbd, nEmbd, rng),
			mlpFC1: matrix(4*nEmbd, nEmbd, rng),
			mlpFC2: matrix(nEmbd, 4*nEmbd, rng),
		}
	}
	stateDict := &StateDict{
		wte:    matrix(vocabSize, nEmbd, rng),
		wpe:    matrix(blockSize, nEmbd, rng),
		lmHead: matrix(vocabSize, nEmbd, rng),
		layers: layers,
	}
	allParams := params(stateDict)
	fmt.Printf("num params: %d\n", len(allParams))

	learningRate, beta1, beta2, epsAdam := 0.01, 0.85, 0.99, 1e-8
	m := make([]float64, len(allParams))
	v := make([]float64, len(allParams))
	for step := 0; step < numSteps; step++ {
		doc := docs[step%len(docs)]
		tokens := make([]int, 0, len(doc)+2)
		tokens = append(tokens, BOS)
		for _, ch := range doc {
			tokens = append(tokens, charToToken[ch])
		}
		tokens = append(tokens, BOS)

		n := len(tokens) - 1
		if n > blockSize {
			n = blockSize
		}
		keys := make([][][]*Value, nLayer)
		values := make([][][]*Value, nLayer)
		losses := make([]*Value, 0, n)
		for posID := 0; posID < n; posID++ {
			tokenID, targetID := tokens[posID], tokens[posID+1]
			logits := gpt(tokenID, posID, keys, values, stateDict)
			probs := softmax(logits)
			losses = append(losses, probs[targetID].log().neg())
		}

		loss := newValue(0.0)
		for _, l := range losses {
			loss = loss.add(l)
		}
		loss = loss.mul(newValue(1.0 / float64(n)))
		loss.backward()

		lrT := learningRate * (1.0 - float64(step)/numSteps)
		for i, p := range allParams {
			m[i] = beta1*m[i] + (1.0-beta1)*p.grad
			v[i] = beta2*v[i] + (1.0-beta2)*p.grad*p.grad
			mHat := m[i] / (1.0 - math.Pow(beta1, float64(step+1)))
			vHat := v[i] / (1.0 - math.Pow(beta2, float64(step+1)))
			p.data -= lrT * mHat / (math.Sqrt(vHat) + epsAdam)
			p.grad = 0.0
		}

		fmt.Printf("\rstep %4d / %4d | loss %.4f", step+1, numSteps, loss.data)
		if step+1 == numSteps {
			fmt.Print("\n")
		}
	}

	temperature := 0.5
	fmt.Println("\n--- inference (new, hallucinated names) ---")
	for sampleIdx := 0; sampleIdx < 20; sampleIdx++ {
		keys := make([][][]*Value, nLayer)
		values := make([][][]*Value, nLayer)
		tokenID := BOS
		var sample strings.Builder

		for posID := 0; posID < blockSize; posID++ {
			logits := gpt(tokenID, posID, keys, values, stateDict)
			scaledLogits := make([]*Value, len(logits))
			for i, l := range logits {
				scaledLogits[i] = l.div(newValue(temperature))
			}
			probs := softmax(scaledLogits)
			weights := make([]float64, len(probs))
			for i, p := range probs {
				weights[i] = p.data
			}
			tokenID = randomChoices(rng, weights)
			if tokenID == BOS {
				break
			}
			sample.WriteRune(uchars[tokenID])
		}
		fmt.Printf("sample %2d: %s\n", sampleIdx+1, sample.String())
	}
}

import Foundation
import MLX
import MLXNN
import MLXRandom

// MARK: - GPT-2 Transformer in MLX Swift

/// A GPT-2 style causal transformer built with MLX Swift.
/// Supports pre-norm architecture with GELU FFN, tied embeddings, and causal masking.
public class GPT2Block: Module {
    @ModuleInfo(key: "ln1") var attnNorm: LayerNorm
    @ModuleInfo(key: "ln2") var ffnNorm: LayerNorm
    @ModuleInfo(key: "wQ") var wQ: Linear
    @ModuleInfo(key: "wK") var wK: Linear
    @ModuleInfo(key: "wV") var wV: Linear
    @ModuleInfo(key: "wO") var wO: Linear
    @ModuleInfo(key: "fc1") var fc1: Linear
    @ModuleInfo(key: "fc2") var fc2: Linear
    @ModuleInfo(key: "attn_drop") var attnDrop: Dropout
    @ModuleInfo(key: "resid_drop") var residDrop: Dropout

    let nHeads: Int
    let headDim: Int

    public init(config: TransformerConfig, layerIndex: Int) {
        let d = config.dModel
        self.nHeads = config.nHeads
        self.headDim = config.headDim

        self._attnNorm = ModuleInfo(wrappedValue: LayerNorm(dimensions: d, eps: 1e-5))
        self._ffnNorm = ModuleInfo(wrappedValue: LayerNorm(dimensions: d, eps: 1e-5))

        self._wQ = ModuleInfo(wrappedValue: Linear(d, d))
        self._wK = ModuleInfo(wrappedValue: Linear(d, d))
        self._wV = ModuleInfo(wrappedValue: Linear(d, d))
        self._wO = ModuleInfo(wrappedValue: Linear(d, d))

        self._fc1 = ModuleInfo(wrappedValue: Linear(d, config.ffnHidden))
        self._fc2 = ModuleInfo(wrappedValue: Linear(config.ffnHidden, d))

        self._attnDrop = ModuleInfo(wrappedValue: Dropout(p: config.dropout))
        self._residDrop = ModuleInfo(wrappedValue: Dropout(p: config.dropout))
    }

    public func callAsFunction(_ x: MLXArray, mask: MLXArray? = nil) -> MLXArray {
        // Pre-norm attention
        let normed = attnNorm(x)
        let attnOut = attention(normed, mask: mask)
        let postAttn = x + residDrop(attnOut)

        // Pre-norm FFN
        let normedFFN = ffnNorm(postAttn)
        let ffnOut = feedForward(normedFFN)
        return postAttn + residDrop(ffnOut)
    }

    private func attention(_ x: MLXArray, mask: MLXArray?) -> MLXArray {
        let B = x.dim(0)
        let S = x.dim(1)

        // Project Q, K, V
        var q = wQ(x)
        var k = wK(x)
        var v = wV(x)

        // Reshape to [B, S, nHeads, headDim] then transpose to [B, nHeads, S, headDim]
        q = q.reshaped(B, S, nHeads, headDim).transposed(0, 2, 1, 3)
        k = k.reshaped(B, S, nHeads, headDim).transposed(0, 2, 1, 3)
        v = v.reshaped(B, S, nHeads, headDim).transposed(0, 2, 1, 3)

        // Scaled dot-product attention
        let scale = MLXArray(Float(1.0 / sqrt(Float(headDim))))
        var scores = matmul(q, k.transposed(0, 1, 3, 2)) * scale

        if let mask = mask {
            scores = scores + mask
        }

        var weights = softmax(scores, axis: -1)
        weights = attnDrop(weights)

        // Weighted sum
        var attnOut = matmul(weights, v)

        // Transpose back and flatten: [B, nHeads, S, headDim] -> [B, S, dModel]
        attnOut = attnOut.transposed(0, 2, 1, 3).reshaped(B, S, nHeads * headDim)

        return wO(attnOut)
    }

    private func feedForward(_ x: MLXArray) -> MLXArray {
        // GPT-2 FFN: fc1 -> GELU -> fc2
        var h = fc1(x)
        h = geluApproximate(h)
        return fc2(h)
    }
}

// MARK: - Full GPT-2 Model

public class GPT2Model: Module {
    @ModuleInfo(key: "embedding") var embedding: Embedding
    @ParameterInfo(key: "pos_embedding") var posEmbedding: MLXArray
    @ModuleInfo(key: "blocks") var blocks: [GPT2Block]
    @ModuleInfo(key: "final_ln") var finalNorm: LayerNorm
    @ModuleInfo(key: "emb_drop") var embDrop: Dropout

    public let config: TransformerConfig

    public init(config: TransformerConfig) {
        self.config = config

        self._embedding = ModuleInfo(wrappedValue: Embedding(embeddingCount: config.vocabSize, dimensions: config.dModel))

        // Learned positional embeddings initialized from N(0, 0.01)
        self._posEmbedding = ParameterInfo(wrappedValue: MLXRandom.normal([config.seqLen, config.dModel]) * 0.01)

        var layerBlocks: [GPT2Block] = []
        for i in 0..<config.nLayers {
            layerBlocks.append(GPT2Block(config: config, layerIndex: i))
        }
        self._blocks = ModuleInfo(wrappedValue: layerBlocks)

        self._finalNorm = ModuleInfo(wrappedValue: LayerNorm(dimensions: config.dModel, eps: 1e-5))
        self._embDrop = ModuleInfo(wrappedValue: Dropout(p: config.dropout))
    }

    public func callAsFunction(_ tokens: MLXArray) -> MLXArray {
        let S = tokens.dim(1)

        // Token embedding + positional embedding
        var x = embedding(tokens)
        let posSlice = posEmbedding[0..<S]
        x = embDrop(x + posSlice)

        // Build causal mask: upper-triangular with -inf
        let mask = createCausalMask(seqLen: S)

        // Transformer blocks
        for block in blocks {
            x = block(x, mask: mask)
        }

        // Final layer norm
        x = finalNorm(x)

        // Tied output projection: x @ embedding.weight^T -> [B, S, vocabSize]
        let logits = embedding.asLinear(x)

        return logits
    }

    /// Forward through only the front half: embedding + blocks[0..<splitAt].
    /// Returns the intermediate activation tensor [B, S, dModel].
    public func forwardFrontHalf(_ tokens: MLXArray, splitAt: Int) -> MLXArray {
        let S = tokens.dim(1)

        var x = embedding(tokens)
        let posSlice = posEmbedding[0..<S]
        x = embDrop(x + posSlice)

        let mask = createCausalMask(seqLen: S)

        for i in 0..<splitAt {
            x = blocks[i](x, mask: mask)
        }

        return x
    }

    /// Forward through a middle slice: blocks[fromLayer..<toLayer].
    /// No embedding, no finalNorm/LM-head.  Returns activation [B, S, dModel].
    public func forwardMiddle(_ activation: MLXArray, fromLayer: Int, toLayer: Int) -> MLXArray {
        let S = activation.dim(1)
        let mask = createCausalMask(seqLen: S)

        var x = activation
        for i in fromLayer..<toLayer {
            x = blocks[i](x, mask: mask)
        }
        return x
    }

    /// Forward through only the back half: blocks[fromLayer..<nLayers] + finalNorm + LM head.
    /// Returns logits [B, S, vocabSize].
    public func forwardBackHalf(_ activation: MLXArray, fromLayer: Int) -> MLXArray {
        let S = activation.dim(1)
        let mask = createCausalMask(seqLen: S)

        var x = activation
        for i in fromLayer..<config.nLayers {
            x = blocks[i](x, mask: mask)
        }

        x = finalNorm(x)
        let logits = embedding.asLinear(x)
        return logits
    }

    public func createCausalMask(seqLen: Int) -> MLXArray {
        // Create additive causal mask: 0 where allowed, -inf where masked
        let indices = MLXArray(Array(0..<seqLen))
        let rowIndices = indices.reshaped(seqLen, 1)
        let colIndices = indices.reshaped(1, seqLen)
        let mask = MLX.where(rowIndices .>= colIndices, MLXArray(Float(0.0)), MLXArray(Float(-1e9)))
        return mask.reshaped(1, 1, seqLen, seqLen)
    }
}

// MARK: - Surrogate Loss for Split Training

/// Surrogate loss for split-model backward pass.
/// Computes a scalar whose gradient w.r.t. model parameters equals the true gradient
/// when upstream gradients are received from a remote server.
///
/// Math: loss = sum(activation * stopGradient(upstreamGrads))
///   d(loss)/d(params) = upstreamGrads * d(activation)/d(params)
/// This is exactly the chain rule with the server's gradient injected.
public func frontSurrogateLoss(
    model: GPT2Model,
    inputs: MLXArray,
    upstreamGrads: MLXArray,
    splitAt: Int
) -> MLXArray {
    let activation = model.forwardFrontHalf(inputs, splitAt: splitAt)
    return (activation * stopGradient(upstreamGrads)).sum()
}

// MARK: - Middle Surrogate Loss for Pipeline Training

/// Surrogate loss for a middle pipeline stage.
/// Same trick as frontSurrogateLoss but applied to forwardMiddle.
public func middleSurrogateLoss(
    model: GPT2Model,
    inputActivation: MLXArray,
    upstreamGrads: MLXArray,
    fromLayer: Int,
    toLayer: Int
) -> MLXArray {
    let activation = model.forwardMiddle(inputActivation, fromLayer: fromLayer, toLayer: toLayer)
    return (activation * stopGradient(upstreamGrads)).sum()
}

// MARK: - Tail Loss for Pipeline Training

/// Loss function for the tail worker in pipeline-parallel training.
/// Runs the back half of the model (blocks[fromLayer..<N] + finalNorm + LM head)
/// and computes cross-entropy loss against targets.
public func tailLoss(
    model: GPT2Model,
    inputActivation: MLXArray,
    targets: MLXArray,
    fromLayer: Int
) -> MLXArray {
    let logits = model.forwardBackHalf(inputActivation, fromLayer: fromLayer)
    let B = logits.dim(0)
    let S = logits.dim(1)
    let V = logits.dim(2)
    let flatLogits = logits.reshaped(B * S, V)
    let flatTargets = targets.reshaped(B * S)
    return crossEntropy(logits: flatLogits, targets: flatTargets, reduction: .mean)
}

// MARK: - Loss Function

/// Cross-entropy loss for language modeling.
/// logits: [B, S, V], targets: [B, S] (Int32 token IDs)
public func gpt2Loss(model: GPT2Model, inputs: MLXArray, targets: MLXArray) -> MLXArray {
    let logits = model(inputs)
    // Reshape for crossEntropy: [B*S, V] and [B*S]
    let B = logits.dim(0)
    let S = logits.dim(1)
    let V = logits.dim(2)
    let flatLogits = logits.reshaped(B * S, V)
    let flatTargets = targets.reshaped(B * S)
    return crossEntropy(logits: flatLogits, targets: flatTargets, reduction: .mean)
}

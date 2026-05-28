import Foundation

// MARK: - Transformer Configuration

public struct TransformerConfig: Sendable {
    public var vocabSize: Int = 256       // set from dataset at runtime
    public var dModel: Int = 768          // GPT-2 small: 768
    public var nHeads: Int = 12           // GPT-2 small: 12
    public var nLayers: Int = 12          // GPT-2 small: 12
    public var ffnHiddenMul: Int = 4      // FFN hidden = dModel * 4 = 3072
    public var seqLen: Int = 128          // start conservative (GPT-2: 1024)
    public var batchSize: Int = 1         // start at 1 for memory safety
    public var learningRate: Float = 6e-4 // GPT-2 training LR
    public var weightDecay: Float = 0.1   // AdamW decoupled weight decay
    public var beta1: Float = 0.9
    public var beta2: Float = 0.95        // GPT-2 uses 0.95
    public var dropout: Float = 0.1       // GPT-2 uses 0.1

    public var headDim: Int { dModel / nHeads }
    public var ffnHidden: Int { dModel * ffnHiddenMul }

    public init(
        vocabSize: Int = 256,
        dModel: Int = 768,
        nHeads: Int = 12,
        nLayers: Int = 12,
        ffnHiddenMul: Int = 4,
        seqLen: Int = 128,
        batchSize: Int = 1,
        learningRate: Float = 6e-4,
        weightDecay: Float = 0.1,
        beta1: Float = 0.9,
        beta2: Float = 0.95,
        dropout: Float = 0.1
    ) {
        self.vocabSize = vocabSize
        self.dModel = dModel
        self.nHeads = nHeads
        self.nLayers = nLayers
        self.ffnHiddenMul = ffnHiddenMul
        self.seqLen = seqLen
        self.batchSize = batchSize
        self.learningRate = learningRate
        self.weightDecay = weightDecay
        self.beta1 = beta1
        self.beta2 = beta2
        self.dropout = dropout
    }

    /// GPT-2 small preset (BPE vocab: 50,257)
    public static var gpt2Small: TransformerConfig {
        TransformerConfig(
            vocabSize: 50257,
            dModel: 768, nHeads: 12, nLayers: 12,
            seqLen: 128, batchSize: 1,
            learningRate: 6e-4, weightDecay: 0.1,
            beta1: 0.9, beta2: 0.95
        )
    }
}

// MARK: - Training Log Entry

public struct GPTTrainingLogEntry: Identifiable, Sendable {
    public let id = UUID()
    public let timestamp: Date
    public let message: String

    public init(timestamp: Date, message: String) {
        self.timestamp = timestamp
        self.message = message
    }
}

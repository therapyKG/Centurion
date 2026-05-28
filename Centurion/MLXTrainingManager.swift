import CenturionMLX
import Foundation
import MLX
import MLXNN
import MLXOptimizers
import MLXRandom
import OSLog

// MARK: - MLX Training Manager

@MainActor
@Observable
final class MLXTrainingManager {
    // Observable state
    var status: String = "Select a dataset, configure, build, then train."
    var isBuilt: Bool = false
    var isTraining: Bool = false
    var progress: Double = 0
    var trainingLog: [GPTTrainingLogEntry] = []
    var currentLoss: Float = 0
    var currentEpoch: Int = 0
    var totalEpochs: Int = 0
    var tokensPerSecond: Double = 0
    var generatedSample: String = ""

    // Config (user can modify before building)
    var config = TransformerConfig()

    // Dataset
    var selectedDataset: BuiltinDataset = .nurseryRhymes
    var useBPETokenizer: Bool = false
    var stepsPerEpoch: Int = 50
    private(set) var dataset: TextDataset?

    // Internal
    private var model: GPT2Model?
    private var optimizer: AdamW?
    private var trainingTask: Task<Void, Never>?

    private let logger = Logger(subsystem: "Centurion", category: "MLXTraining")

    // MARK: - Load Dataset & Build Model

    func loadAndBuild() {
        let buildStart = ContinuousClock.now

        // Load dataset
        let ds = selectedDataset.load(bpe: useBPETokenizer)
        self.dataset = ds
        log(ds.textStats())

        // Set vocab size from dataset
        config.vocabSize = ds.vocabSize
        log("Vocab set to \(ds.vocabSize) unique characters")

        log("Building MLX transformer: d=\(config.dModel) h=\(config.nHeads) L=\(config.nLayers) seq=\(config.seqLen) B=\(config.batchSize) vocab=\(config.vocabSize)")

        // Build the model
        let m = GPT2Model(config: config)

        // Evaluate to materialize parameters
        MLX.eval(m)

        self.model = m
        self.optimizer = AdamW(
            learningRate: config.learningRate,
            betas: (config.beta1, config.beta2),
            eps: 1e-8,
            weightDecay: config.weightDecay
        )

        self.isBuilt = true

        let elapsed = buildStart.duration(to: .now)
        let paramCount = countParameters()
        let paramStr = paramCount >= 1_000_000
            ? String(format: "%.1fM", Double(paramCount) / 1_000_000.0)
            : String(format: "%.1fK", Double(paramCount) / 1000.0)
        let msg = String(format: "Ready — %d params (%@) — built in %.0f ms",
                         paramCount, paramStr, durationMs(elapsed))

        status = msg
        log(msg)

        // Reset metrics
        currentLoss = 0
        currentEpoch = 0
        progress = 0
        generatedSample = ""
    }

    // MARK: - Train

    func startTraining(epochs: Int) {
        guard let model, let optimizer, let dataset else {
            status = "Build the model first."
            return
        }
        guard !isTraining else { return }

        isTraining = true
        progress = 0
        currentEpoch = 0
        totalEpochs = epochs

        let stepsPerEpoch = self.stepsPerEpoch
        let config = self.config
        let tokenizer = dataset.tokenizer
        let corpusTokens = dataset.tokens
        let vocabSize = dataset.vocabSize
        let tokenCount = dataset.tokens.count

        // Create the loss+grad function
        let lossAndGrad = valueAndGrad(model: model, gpt2Loss)

        trainingTask = Task.detached(priority: .utility) { [weak self, lossAndGrad, model, optimizer, dataset, tokenizer, corpusTokens, config] in
            let batchSize = config.batchSize
            let seqLen = config.seqLen

            await self?.log("Training — \(epochs) epochs × \(stepsPerEpoch) steps, batch=\(batchSize), seq=\(seqLen)")
            await self?.log("Dataset: \(tokenCount) tokens, vocab=\(vocabSize)")

            for epoch in 0..<epochs {
                let epochStart = ContinuousClock.now
                var epochLoss: Float = 0

                for step in 0..<stepsPerEpoch {
                    // Sample a batch from the dataset
                    let (inputArray, targetArray) = dataset.sampleBatchMLX(
                        batchSize: batchSize,
                        seqLen: seqLen
                    )

                    // Forward + backward
                    let (loss, grads) = lossAndGrad(model, inputArray, targetArray)

                    // Update weights
                    optimizer.update(model: model, gradients: grads)

                    // Evaluate to materialize the computation
                    MLX.eval(model, optimizer)

                    let lossVal = loss.item(Float.self)
                    epochLoss += lossVal

                    // Yield periodically
                    if step % 5 == 4 {
                        try? await Task.sleep(for: .milliseconds(1))
                    }

                    if Task.isCancelled {
                        await self?.log("Training cancelled")
                        await self?.finishTraining()
                        return
                    }
                }

                let avgLoss = epochLoss / Float(stepsPerEpoch)
                let epochElapsed = epochStart.duration(to: .now)
                let totalTokens = Double(batchSize * seqLen * stepsPerEpoch)
                let elapsedMs = durationMs(epochElapsed)
                let tps = totalTokens / max(elapsedMs / 1000.0, 0.001)

                // Generate a sample every 5 epochs (or last epoch)
                var sampleText = ""
                if epoch % 5 == 4 || epoch == epochs - 1 {
                    sampleText = Self.generateSample(
                        model: model,
                        tokenizer: tokenizer,
                        corpusTokens: corpusTokens,
                        config: config,
                        length: 80
                    )
                }

                let msg = String(
                    format: "Epoch %d/%d — loss: %.4f — %.0f ms — %.0f tok/s",
                    epoch + 1, epochs, avgLoss, elapsedMs, tps
                )

                let capturedSampleText = sampleText
                await MainActor.run { [weak self] in
                    self?.currentEpoch = epoch + 1
                    self?.currentLoss = avgLoss
                    self?.progress = Double(epoch + 1) / Double(epochs)
                    self?.tokensPerSecond = tps
                    self?.status = msg
                    if !capturedSampleText.isEmpty {
                        self?.generatedSample = capturedSampleText
                    }
                }
                await self?.log(msg)
                if !sampleText.isEmpty {
                    let preview = String(sampleText.prefix(60)).replacingOccurrences(of: "\n", with: "\\n")
                    await self?.log("  Sample: \(preview)…")
                }

                try? await Task.sleep(for: .milliseconds(5))
            }

            await self?.finishTraining()
        }
    }

    func stopTraining() {
        trainingTask?.cancel()
        trainingTask = nil
    }

    func clearLog() {
        trainingLog.removeAll()
    }

    // MARK: - Sample Generation

    private nonisolated static func generateSample(
        model: GPT2Model,
        tokenizer: AnyTokenizer,
        corpusTokens: [Int32],
        config: TransformerConfig,
        length: Int,
        temperature: Float = 0.8,
        topK: Int = 40
    ) -> String {
        let seqLen = config.seqLen
        var tokenIDs: [Int32]

        // Seed with a random chunk from the corpus
        if corpusTokens.count > seqLen {
            let start = Int.random(in: 0...(corpusTokens.count - seqLen))
            tokenIDs = Array(corpusTokens[start..<(start + seqLen)])
        } else {
            tokenIDs = corpusTokens
            while tokenIDs.count < seqLen {
                tokenIDs.insert(0, at: 0)
            }
        }
        let seedText = tokenizer.decode(tokenIDs)

        var generated: [Int32] = []

        // Disable dropout for generation
        model.train(false)
        defer { model.train(true) }

        for _ in 0..<length {
            // Create input: [1, seqLen]
            let inputArray = MLXArray(tokenIDs).reshaped(1, seqLen)

            // Forward pass
            let logits = model(inputArray)

            // Get logits for last position: [1, seqLen, vocabSize] -> [vocabSize]
            var lastLogits = logits[0, seqLen - 1]

            // Apply temperature
            if temperature > 0 {
                lastLogits = lastLogits / MLXArray(temperature)
            }

            // Top-k filtering: zero out everything outside top-k
            if topK > 0 {
                let k = min(topK, lastLogits.dim(0))
                let topKValues = sorted(lastLogits, axis: -1)[(lastLogits.dim(0) - k)...]
                let threshold = topKValues[0]
                lastLogits = MLX.where(lastLogits .< threshold, MLXArray(Float(-1e9)), lastLogits)
            }

            // Sample from the distribution (categorical expects unnormalized logits)
            let sampled = MLXRandom.categorical(lastLogits)
            let nextToken = sampled.item(Int32.self)

            generated.append(nextToken)

            // Slide window
            tokenIDs.removeFirst()
            tokenIDs.append(nextToken)

            // Evaluate to avoid building up computation graph
            MLX.eval(logits)
        }

        let seedSuffix = String(seedText.suffix(20))
        return "…\(seedSuffix)⟩ \(tokenizer.decode(generated))"
    }

    // MARK: - Internal

    private func finishTraining() {
        isTraining = false
        progress = 1
        log("Training complete")
    }

    func log(_ message: String) {
        let entry = GPTTrainingLogEntry(timestamp: Date(), message: message)
        trainingLog.append(entry)
        logger.info("\(message, privacy: .public)")
    }

    private func countParameters() -> Int {
        guard let model else { return 0 }
        return model.parameters().flattenedValues().reduce(0) { $0 + $1.size }
    }
}

// MARK: - Duration helper

private nonisolated func durationMs(_ d: Duration) -> Double {
    let (seconds, attoseconds) = d.components
    return Double(seconds) * 1000.0 + Double(attoseconds) / 1_000_000_000_000_000.0
}

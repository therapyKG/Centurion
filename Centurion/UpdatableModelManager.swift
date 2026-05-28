import Foundation
import CoreML
import SwiftUI
import OSLog
import CoreVideo

struct TrainingLogEntry: Identifiable {
    let id = UUID()
    let timestamp: Date
    let message: String
}

@MainActor
@Observable
final class UpdatableModelManager {
    var status: String = "Add an updatable Core ML model to your bundle, set its name below, then Load."
    var isTraining: Bool = false
    var progress: Double = 0
    var modelSummary: String = ""
    var loadedModelName: String?
    var lastPredictionSummary: String = ""
    var trainingLog: [TrainingLogEntry] = []
    var selectedComputeUnits: MLComputeUnits = .all

    private let logger = Logger(subsystem: "Centurion", category: "ML")

    private(set) var model: MLModel?
    private var modelURL: URL?
    private var trainingStartTime: Date?

    // MARK: - Public API

    func unload() {
        model = nil
        modelURL = nil
        loadedModelName = nil
        modelSummary = ""
        lastPredictionSummary = ""
        status = "Unloaded model."
    }

    func loadModel(named name: String) {
        do {
            let (compiledURL, effectiveName) = try locateOrCompileModel(named: name)
            let writableURL = try ensureWritableModelCopy(compiledURL: compiledURL, modelName: effectiveName)

            let config = MLModelConfiguration()
            config.computeUnits = .cpuAndNeuralEngine

            let loaded = try MLModel(contentsOf: writableURL, configuration: config)
            self.model = loaded
            self.modelURL = writableURL
            self.loadedModelName = effectiveName
            self.modelSummary = summarize(model: loaded)
            self.status = "Loaded model: \(effectiveName)"
            logger.info("Loaded model at: \(writableURL.path, privacy: .public)")
        } catch {
            logger.error("Failed to load model: \(error.localizedDescription, privacy: .public)")
            self.status = "Failed to load model: \(error.localizedDescription)"
        }
    }

    func warmUpPrediction() async {
        guard let model else {
            status = "No model loaded."
            return
        }
        do {
            let synthetic = try syntheticInput(for: model)
            let provider = try MLDictionaryFeatureProvider(dictionary: synthetic)
            let start = ContinuousClock.now
            let output = try await model.prediction(from: provider)
            let elapsed = start.duration(to: .now)
            self.lastPredictionSummary = summarize(features: output)
            self.status = String(format: "Prediction completed in %.1f ms", elapsed.milliseconds)
        } catch {
            self.status = "Prediction failed: \(error)"
        }
    }

    func trainWithSynthetic(samples count: Int = 100, epochs: Int = 5) async {
        guard let modelURL, let model else {
            status = "No model loaded."
            return
        }
        guard model.modelDescription.isUpdatable else {
            status = "This model is not updatable."
            return
        }

        do {
            let batchStart = ContinuousClock.now
            let batch = try syntheticTrainingBatch(for: model, count: count)
            let batchElapsed = batchStart.duration(to: .now)

            log("Generated \(count) synthetic samples in \(String(format: "%.0f", batchElapsed.milliseconds)) ms")
            log("Compute units: \(computeUnitsLabel(selectedComputeUnits))")

            try await runUpdateTask(modelAt: modelURL, trainingData: batch, epochs: epochs)
        } catch {
            self.status = "Training failed: \(error.localizedDescription)"
            log("FAILED: \(error.localizedDescription)")
        }
    }

    func setComputeUnits(_ units: MLComputeUnits) {
        selectedComputeUnits = units
    }

    func clearLog() {
        trainingLog.removeAll()
    }

    // MARK: - Logging

    private func log(_ message: String) {
        let entry = TrainingLogEntry(timestamp: Date(), message: message)
        trainingLog.append(entry)
        logger.info("\(message, privacy: .public)")
    }

    func computeUnitsLabel(_ units: MLComputeUnits) -> String {
        switch units {
        case .cpuOnly: return "CPU Only"
        case .cpuAndGPU: return "CPU + GPU"
        case .cpuAndNeuralEngine: return "CPU + ANE"
        case .all: return "All (CPU + GPU + ANE)"
        @unknown default: return "Unknown"
        }
    }

    // MARK: - Model location & preparation

    private func locateOrCompileModel(named name: String) throws -> (compiledURL: URL, effectiveName: String) {
        if let compiled = Bundle.main.url(forResource: name, withExtension: "mlmodelc") {
            return (compiled, name)
        }
        if let raw = Bundle.main.url(forResource: name, withExtension: "mlmodel") {
            let compiled = try MLModel.compileModel(at: raw)
            return (compiled, name)
        }
        if let anyCompiled = Bundle.main.urls(forResourcesWithExtension: "mlmodelc", subdirectory: nil)?.first {
            let effectiveName = anyCompiled.deletingPathExtension().lastPathComponent
            return (anyCompiled, effectiveName)
        }
        if let anyRaw = Bundle.main.urls(forResourcesWithExtension: "mlmodel", subdirectory: nil)?.first {
            let compiled = try MLModel.compileModel(at: anyRaw)
            let effectiveName = anyRaw.deletingPathExtension().lastPathComponent
            return (compiled, effectiveName)
        }
        throw NSError(domain: "UpdatableModelManager", code: 1, userInfo: [NSLocalizedDescriptionKey: "No .mlmodel or .mlmodelc found in app bundle."])
    }

    private func ensureWritableModelCopy(compiledURL: URL, modelName: String) throws -> URL {
        let fm = FileManager.default
        let appSupport = try fm.url(for: .applicationSupportDirectory, in: .userDomainMask, appropriateFor: nil, create: true)
        let modelsDir = appSupport.appendingPathComponent("Models", isDirectory: true)
        if !fm.fileExists(atPath: modelsDir.path) {
            try fm.createDirectory(at: modelsDir, withIntermediateDirectories: true)
        }
        let dest = modelsDir.appendingPathComponent("\(modelName).mlmodelc", isDirectory: true)
        if fm.fileExists(atPath: dest.path) {
            try fm.removeItem(at: dest)
        }
        try fm.copyItem(at: compiledURL, to: dest)
        return dest
    }

    // MARK: - Update Task

    private func runUpdateTask(modelAt url: URL, trainingData: MLBatchProvider, epochs: Int) async throws {
        isTraining = true
        progress = 0
        status = "Starting training..."

        let totalEpochs = max(1, epochs)
        trainingStartTime = Date()

        let config = MLModelConfiguration()
        config.computeUnits = selectedComputeUnits
        config.parameters = [
            .epochs: NSNumber(value: totalEpochs),
            .shuffle: NSNumber(value: true),
            .miniBatchSize: NSNumber(value: 32)
        ]

        log("Training started — \(totalEpochs) epochs, \(trainingData.count) samples, batch size 32")

        var epochStartTime = ContinuousClock.now
        var miniBatchCount = 0

        let handlers = MLUpdateProgressHandlers(
            forEvents: [.trainingBegin, .miniBatchEnd, .epochEnd],
            progressHandler: { [weak self] context in
                Task { @MainActor [weak self] in
                    guard let self else { return }
                    switch context.event {
                    case .trainingBegin:
                        self.status = "Training began"
                        self.progress = 0
                        epochStartTime = .now
                        miniBatchCount = 0
                        self.log("Training begin event received")
                    case .miniBatchEnd:
                        miniBatchCount += 1
                        if let loss = context.metrics[.lossValue] as? Double {
                            self.status = String(format: "Mini-batch %d — loss: %.4f", miniBatchCount, loss)
                        }
                    case .epochEnd:
                        let epochElapsed = epochStartTime.duration(to: .now)
                        let completedEpochs = (context.metrics[.epochIndex] as? Int).map { $0 + 1 } ?? 0
                        self.progress = Double(completedEpochs) / Double(totalEpochs)

                        if let loss = context.metrics[.lossValue] as? Double {
                            let msg = String(
                                format: "Epoch %d/%d — loss: %.4f — %.0f ms (%d batches)",
                                completedEpochs, totalEpochs, loss,
                                epochElapsed.milliseconds, miniBatchCount
                            )
                            self.status = msg
                            self.log(msg)
                        } else {
                            let msg = String(
                                format: "Epoch %d/%d — %.0f ms",
                                completedEpochs, totalEpochs, epochElapsed.milliseconds
                            )
                            self.status = msg
                            self.log(msg)
                        }

                        epochStartTime = .now
                        miniBatchCount = 0
                    default:
                        break
                    }
                }
            },
            completionHandler: { [weak self] context in
                Task { @MainActor [weak self] in
                    guard let self else { return }
                    self.isTraining = false
                    self.progress = 1

                    if let error = context.task.error {
                        self.status = "Training error: \(error.localizedDescription)"
                        self.log("FAILED: \(error.localizedDescription)")
                        return
                    }

                    let totalElapsed = self.trainingStartTime.map { Date().timeIntervalSince($0) } ?? 0
                    self.model = context.model
                    self.modelSummary = self.summarize(model: context.model)

                    let msg = String(
                        format: "Training complete — total: %.1f s — compute: %@",
                        totalElapsed,
                        self.computeUnitsLabel(self.selectedComputeUnits)
                    )
                    self.status = msg
                    self.log(msg)
                }
            }
        )

        do {
            let task = try MLUpdateTask(forModelAt: url, trainingData: trainingData, configuration: config, progressHandlers: handlers)
            task.resume()
        } catch {
            isTraining = false
            progress = 0
            throw error
        }
    }

    // MARK: - Synthetic data

    private func syntheticInput(for model: MLModel) throws -> [String: MLFeatureValue] {
        var dict: [String: MLFeatureValue] = [:]
        for (name, desc) in model.modelDescription.inputDescriptionsByName {
            if let value = try syntheticValue(for: desc) {
                dict[name] = value
            }
        }
        return dict
    }

    private func syntheticTrainingBatch(for model: MLModel, count: Int) throws -> MLArrayBatchProvider {
        let md = model.modelDescription
        let inputDescs = md.inputDescriptionsByName
        let trainingDescs = md.trainingInputDescriptionsByName
        let targetNames = trainingDescs.keys.filter { inputDescs[$0] == nil }

        guard !targetNames.isEmpty else {
            throw NSError(domain: "UpdatableModelManager", code: 2, userInfo: [NSLocalizedDescriptionKey: "Could not infer target feature name(s)."])
        }

        var providers: [MLFeatureProvider] = []
        for _ in 0..<count {
            var features: [String: MLFeatureValue] = [:]
            for (name, desc) in inputDescs {
                if let v = try syntheticValue(for: desc) {
                    features[name] = v
                }
            }
            for name in targetNames {
                if let desc = trainingDescs[name], let v = try syntheticValue(for: desc, isTarget: true) {
                    features[name] = v
                }
            }
            let provider = try MLDictionaryFeatureProvider(dictionary: features)
            providers.append(provider)
        }
        return MLArrayBatchProvider(array: providers)
    }

    private func syntheticValue(for desc: MLFeatureDescription, isTarget: Bool = false) throws -> MLFeatureValue? {
        switch desc.type {
        case .image:
            guard let constraint = desc.imageConstraint else { return nil }
            let pb = try makeRandomPixelBuffer(width: constraint.pixelsWide, height: constraint.pixelsHigh, pixelFormat: constraint.pixelFormatType)
            return MLFeatureValue(pixelBuffer: pb)
        case .multiArray:
            guard let c = desc.multiArrayConstraint else { return nil }
            let arr = try MLMultiArray(shape: c.shape, dataType: c.dataType)
            for i in 0..<arr.count { arr[i] = NSNumber(value: Float.random(in: 0...1)) }
            return MLFeatureValue(multiArray: arr)
        case .double:
            return MLFeatureValue(double: Double.random(in: 0...1))
        case .int64:
            return MLFeatureValue(int64: Int64.random(in: 0...9))
        case .string:
            if isTarget {
                // For MNIST-style models, labels are digit strings
                return MLFeatureValue(string: String(Int.random(in: 0...9)))
            }
            return MLFeatureValue(string: UUID().uuidString.prefix(4).description)
        case .dictionary, .sequence, .invalid, .state:
            return nil
        @unknown default:
            return nil
        }
    }

    // MARK: - Summaries

    private func summarize(model: MLModel) -> String {
        let md = model.modelDescription
        var lines: [String] = []
        lines.append("Model: \(loadedModelName ?? "(unknown)")")
        lines.append("Updatable: \(md.isUpdatable ? "yes" : "no")")
        lines.append("")
        lines.append("Inputs:")
        for (name, d) in md.inputDescriptionsByName {
            lines.append("  • \(name): \(typeString(d))")
        }
        lines.append("Outputs:")
        for (name, d) in md.outputDescriptionsByName {
            lines.append("  • \(name): \(typeString(d))")
        }
        if !md.trainingInputDescriptionsByName.isEmpty {
            lines.append("")
            lines.append("Training inputs:")
            for (name, d) in md.trainingInputDescriptionsByName {
                lines.append("  • \(name): \(typeString(d))")
            }
            let inputKeys = Set(md.inputDescriptionsByName.keys)
            let targetKeys = Set(md.trainingInputDescriptionsByName.keys).subtracting(inputKeys)
            if !targetKeys.isEmpty {
                lines.append("Targets: \(Array(targetKeys).joined(separator: ", "))")
            }
        }
        return lines.joined(separator: "\n")
    }

    private func summarize(features: MLFeatureProvider) -> String {
        var out: [String] = []
        for name in features.featureNames.sorted() {
            if let v = features.featureValue(for: name) {
                out.append("  • \(name): \(summary(of: v))")
            }
        }
        return out.joined(separator: "\n")
    }

    private func typeString(_ d: MLFeatureDescription) -> String {
        switch d.type {
        case .image:
            if let c = d.imageConstraint { return "image (\(c.pixelsWide)x\(c.pixelsHigh))" } else { return "image" }
        case .multiArray:
            if let c = d.multiArrayConstraint {
                let shape = c.shape.map { $0.intValue }
                return "multiArray \(shape)"
            } else { return "multiArray" }
        case .double: return "double"
        case .int64: return "int64"
        case .string: return "string"
        case .dictionary: return "dictionary"
        case .sequence: return "sequence"
        case .state: return "state"
        case .invalid: return "invalid"
        @unknown default: return "unknown"
        }
    }

    private func summary(of v: MLFeatureValue) -> String {
        if v.isUndefined { return "undefined" }
        switch v.type {
        case .string: return v.stringValue
        case .int64: return String(v.int64Value)
        case .double: return String(format: "%.4f", v.doubleValue)
        case .multiArray:
            if let arr = v.multiArrayValue {
                let preview = (0..<min(6, arr.count)).map { i in String(format: "%.3f", arr[i].doubleValue) }.joined(separator: ", ")
                return "multiArray[count=\(arr.count)] [\(preview)]"
            }
            return "multiArray"
        case .image: return "image"
        case .dictionary: return "dictionary"
        case .sequence: return "sequence"
        case .state: return "state"
        case .invalid: return "invalid"
        @unknown default: return "unknown"
        }
    }

    // MARK: - Pixel buffer helper

    private func makeRandomPixelBuffer(width: Int, height: Int, pixelFormat: OSType) throws -> CVPixelBuffer {
        var pb: CVPixelBuffer?
        let attrs: [CFString: Any] = [
            kCVPixelBufferCGImageCompatibilityKey: true,
            kCVPixelBufferCGBitmapContextCompatibilityKey: true
        ]
        let status = CVPixelBufferCreate(kCFAllocatorDefault, width, height, pixelFormat, attrs as CFDictionary, &pb)
        guard status == kCVReturnSuccess, let buffer = pb else {
            throw NSError(domain: "UpdatableModelManager", code: 3, userInfo: [NSLocalizedDescriptionKey: "Failed to create pixel buffer (format: \(pixelFormat))"])
        }
        CVPixelBufferLockBaseAddress(buffer, [])
        defer { CVPixelBufferUnlockBaseAddress(buffer, []) }
        let bytesPerRow = CVPixelBufferGetBytesPerRow(buffer)
        let base = CVPixelBufferGetBaseAddress(buffer)!.assumingMemoryBound(to: UInt8.self)

        if pixelFormat == kCVPixelFormatType_OneComponent8 {
            // Grayscale 8-bit (1 byte per pixel)
            for y in 0..<height {
                let row = base.advanced(by: y * bytesPerRow)
                for x in 0..<width {
                    row[x] = UInt8.random(in: 0...255)
                }
            }
        } else if pixelFormat == kCVPixelFormatType_32BGRA {
            // BGRA 32-bit (4 bytes per pixel)
            for y in 0..<height {
                let row = base.advanced(by: y * bytesPerRow)
                for x in 0..<width {
                    let px = row.advanced(by: x * 4)
                    let gray = UInt8.random(in: 0...255)
                    px[0] = gray  // B
                    px[1] = gray  // G
                    px[2] = gray  // R
                    px[3] = 255   // A
                }
            }
        } else {
            // Fallback: zero-fill
            memset(base, 0, bytesPerRow * height)
        }
        return buffer
    }
}

// MARK: - Duration helper

private extension Duration {
    var milliseconds: Double {
        let (seconds, attoseconds) = components
        return Double(seconds) * 1000.0 + Double(attoseconds) / 1_000_000_000_000_000.0
    }
}

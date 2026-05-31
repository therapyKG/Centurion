import CenturionMLX
import CommonCrypto
import Foundation
import MLX
import MLXNN
import MLXOptimizers
import MLXRandom
import Network
import OSLog
#if canImport(UIKit)
import UIKit
#endif

// MARK: - Pipeline Training Manager
//
// Coordinates pipeline-parallel training with a remote server.
// Each worker owns a contiguous slice of GPT-2 layers.
// Head worker: embedding + blocks[0..<lastLayer]
// Tail worker: blocks[firstLayer..<N] + finalNorm + lm_head
//
// The server relays activations forward and gradients backward
// between workers. Micro-batching with double-buffered sends
// hides network latency behind compute.

// MARK: - Pipeline Message Types

private enum PipelineMsg {
    nonisolated static let authChallenge: UInt8 = 0x30
    nonisolated static let authResponse: UInt8 = 0x31
    nonisolated static let authResult: UInt8 = 0x32

    nonisolated static let register: UInt8 = 0x40
    nonisolated static let config: UInt8 = 0x41
    nonisolated static let configAck: UInt8 = 0x42
    nonisolated static let start: UInt8 = 0x43
    nonisolated static let stop: UInt8 = 0x44

    nonisolated static let dataBatch: UInt8 = 0x50

    nonisolated static let activation: UInt8 = 0x60
    nonisolated static let gradient: UInt8 = 0x61

    nonisolated static let syncBarrier: UInt8 = 0x70
    nonisolated static let syncAck: UInt8 = 0x71

    nonisolated static let lossReport: UInt8 = 0x80
}

/// Thrown when the server sends PIPELINE_STOP, signaling workers to exit their training loop.
private struct PipelineStopSignal: Error {}

// MARK: - Pipeline Stage Info

struct PipelineStageInfo {
    var stageIndex: Int = 0
    var totalStages: Int = 0
    var firstLayer: Int = 0
    var lastLayer: Int = 0
    var isHead: Bool = false
    var isTail: Bool = false
    var numMicroBatches: Int = 4
}

@MainActor
@Observable
final class PipelineTrainingManager {

    // MARK: - Observable State

    var status: String = "Configure and connect to pipeline server."
    var isConnected: Bool = false
    var authFailed: Bool = false
    var isTraining: Bool = false
    var progress: Double = 0
    var trainingLog: [GPTTrainingLogEntry] = []

    // Pipeline info (read-only, assigned by server)
    var stageInfo = PipelineStageInfo()
    var pipelineConfigured: Bool = false

    // Training metrics
    var currentStep: Int = 0
    var totalSteps: Int = 0
    var currentLoss: Float = 0
    var avgForwardMs: Double = 0
    var avgBackwardMs: Double = 0
    var avgSendMs: Double = 0
    var avgRecvMs: Double = 0
    var pipelineEfficiency: Double = 0

    // Model config (set by server)
    var config = TransformerConfig()

    // Server connection settings
    var serverHost: String = "34.60.122.134"
    var serverPort: UInt16 = 9998
    var serverSecret: String = ""

    // Internal
    private var model: GPT2Model?
    private var optimizer: AdamW?
    private var connection: NWConnection?
    private var trainingTask: Task<Void, Never>?

    let workerId: UInt32 = 0  // Server assigns sequential ID

    private let logger = Logger(subsystem: "Centurion", category: "PipelineTraining")

    // MARK: - Connection

    func connect() {
        disconnect()

        guard !serverSecret.isEmpty else {
            status = "Enter a server secret first."
            log("Error: server secret is empty")
            return
        }

        let endpoint = NWEndpoint.hostPort(
            host: NWEndpoint.Host(serverHost),
            port: NWEndpoint.Port(rawValue: serverPort)!
        )
        let params = NWParameters.tcp
        params.serviceClass = .responsiveData

        let conn = NWConnection(to: endpoint, using: params)
        self.connection = conn

        let secret = serverSecret
        authFailed = false
        status = "Connecting to \(serverHost):\(serverPort)..."
        log("Connecting to \(serverHost):\(serverPort)...")

        conn.stateUpdateHandler = { [weak self] state in
            Task { @MainActor [weak self] in
                switch state {
                case .ready:
                    self?.status = "Authenticating..."
                    self?.log("TCP connected, authenticating...")
                    let authOk = await self?.performAuth(connection: conn, secret: secret) ?? false
                    if authOk {
                        self?.isConnected = true
                        self?.status = "Authenticated. Registering with pipeline server..."
                        self?.log("Authenticated successfully")
                        await self?.registerAndWaitForConfig(connection: conn)
                    } else {
                        self?.isConnected = false
                        self?.authFailed = true
                        self?.status = "Authentication failed — wrong secret"
                        self?.log("Authentication failed — check secret")
                        conn.cancel()
                    }
                case .failed(let e):
                    self?.status = "Connection failed: \(e.localizedDescription)"
                    self?.log("Connection failed: \(e.localizedDescription)")
                    self?.disconnect()
                case .cancelled:
                    self?.isConnected = false
                default: break
                }
            }
        }

        conn.start(queue: DispatchQueue(label: "com.centurion.pipeline", qos: .userInitiated))
    }

    func disconnect() {
        trainingTask?.cancel()
        trainingTask = nil
        connection?.cancel()
        connection = nil
        isConnected = false
        isTraining = false
        pipelineConfigured = false
        progress = 0
        currentStep = 0
        currentLoss = 0
        avgForwardMs = 0
        avgBackwardMs = 0
        avgSendMs = 0
        avgRecvMs = 0
        pipelineEfficiency = 0
    }

    // MARK: - Authentication

    private func performAuth(connection: NWConnection, secret: String) async -> Bool {
        do {
            let challenge = try await Self.receiveFrame(connection: connection)
            guard challenge.count == 33, challenge[0] == PipelineMsg.authChallenge else {
                log("Bad auth challenge: len=\(challenge.count)")
                return false
            }
            let nonce = challenge.subdata(in: 1..<33)
            let mac = Self.hmacSHA256(key: Data(secret.utf8), data: nonce)

            var response = Data()
            response.append(PipelineMsg.authResponse)
            response.append(mac)
            try await Self.sendFrame(connection: connection, payload: response)

            let result = try await Self.receiveFrame(connection: connection)
            guard result.count == 2, result[0] == PipelineMsg.authResult else {
                log("Bad auth result: len=\(result.count)")
                return false
            }
            return result[1] == 0
        } catch {
            log("Auth error: \(error)")
            return false
        }
    }

    nonisolated static func hmacSHA256(key: Data, data: Data) -> Data {
        var mac = [UInt8](repeating: 0, count: Int(CC_SHA256_DIGEST_LENGTH))
        key.withUnsafeBytes { keyPtr in
            data.withUnsafeBytes { dataPtr in
                CCHmac(
                    CCHmacAlgorithm(kCCHmacAlgSHA256),
                    keyPtr.baseAddress, key.count,
                    dataPtr.baseAddress, data.count,
                    &mac
                )
            }
        }
        return Data(mac)
    }

    // MARK: - Registration & Config

    private func registerAndWaitForConfig(connection: NWConnection) async {
        do {
            // Send PIPELINE_REGISTER: [1B type][4B worker_id][4B device_type][4B memory_mb]
            var regMsg = Data()
            regMsg.append(PipelineMsg.register)
            appendBigEndian(&regMsg, workerId)
            #if canImport(UIKit)
            let deviceType: UInt32 = UIDevice.current.userInterfaceIdiom == .pad ? 2 : 1
            #else
            let deviceType: UInt32 = 3 // macOS
            #endif
            appendBigEndian(&regMsg, deviceType)
            let memoryMB = UInt32(ProcessInfo.processInfo.physicalMemory / (1024 * 1024))
            appendBigEndian(&regMsg, memoryMB)

            try await Self.sendFrame(connection: connection, payload: regMsg)
            log("Registered as worker \(workerId) (device=\(deviceType), mem=\(memoryMB) MB)")
            status = "Registered. Waiting for pipeline assignment..."

            // Persistent loop: wait for CONFIG → train → repeat
            while true {
                // Wait for PIPELINE_CONFIG (ignore STOP messages from previous run)
                let configFrame = try await Self.receiveFrame(connection: connection)
                if configFrame[0] == PipelineMsg.stop {
                    log("Received STOP (between runs), continuing to wait for CONFIG...")
                    continue
                }
                guard configFrame[0] == PipelineMsg.config else {
                    log("Expected CONFIG, got 0x\(String(format: "%02x", configFrame[0]))")
                    continue
                }

                parsePipelineConfig(configFrame)
                log("Pipeline config received: stage \(stageInfo.stageIndex)/\(stageInfo.totalStages), "
                    + "layers [\(stageInfo.firstLayer)..\(stageInfo.lastLayer)), "
                    + (stageInfo.isHead ? "HEAD " : "")
                    + (stageInfo.isTail ? "TAIL" : ""))

                // Build model
                buildModel()

                // Send CONFIG_ACK: [1B type][4B worker_id][1B status]
                var ack = Data()
                ack.append(PipelineMsg.configAck)
                appendBigEndian(&ack, workerId)
                ack.append(0) // status = OK
                try await Self.sendFrame(connection: connection, payload: ack)

                pipelineConfigured = true
                status = "Configured as \(stageInfo.isHead ? "HEAD" : "TAIL") worker. Waiting for training start..."

                // Wait for PIPELINE_START
                let startFrame = try await Self.receiveFrame(connection: connection)
                guard startFrame[0] == PipelineMsg.start else {
                    log("Expected START, got 0x\(String(format: "%02x", startFrame[0]))")
                    continue
                }
                let totalMiniBatches = readBigEndianUInt32(startFrame, offset: 1)
                self.totalSteps = Int(totalMiniBatches)
                log("Pipeline START: \(totalMiniBatches) mini-batches")

                // Run training (awaits completion)
                await runTrainingRun(connection: connection)

                // Training done — loop back to wait for next CONFIG
                log("Training run complete. Waiting for next pipeline config...")
                status = "Connected. Waiting for next training run..."
                pipelineConfigured = false
            }

        } catch {
            log("Pipeline setup error: \(error)")
            status = "Disconnected: \(error.localizedDescription)"
            disconnect()
        }
    }

    private func parsePipelineConfig(_ data: Data) {
        var offset = 1 // skip type byte
        stageInfo.stageIndex = Int(readBigEndianUInt32(data, offset: offset)); offset += 4
        stageInfo.totalStages = Int(readBigEndianUInt32(data, offset: offset)); offset += 4
        stageInfo.firstLayer = Int(readBigEndianUInt32(data, offset: offset)); offset += 4
        stageInfo.lastLayer = Int(readBigEndianUInt32(data, offset: offset)); offset += 4
        stageInfo.isHead = data[offset] != 0; offset += 1
        stageInfo.isTail = data[offset] != 0; offset += 1
        stageInfo.numMicroBatches = Int(readBigEndianUInt32(data, offset: offset)); offset += 4

        config.vocabSize = Int(readBigEndianUInt32(data, offset: offset)); offset += 4
        config.dModel = Int(readBigEndianUInt32(data, offset: offset)); offset += 4
        config.nHeads = Int(readBigEndianUInt32(data, offset: offset)); offset += 4
        config.nLayers = Int(readBigEndianUInt32(data, offset: offset)); offset += 4
        config.seqLen = Int(readBigEndianUInt32(data, offset: offset)); offset += 4
        config.batchSize = Int(readBigEndianUInt32(data, offset: offset)); offset += 4
        let ffnMul = Int(readBigEndianUInt32(data, offset: offset)); offset += 4
        config.ffnHiddenMul = ffnMul

        config.learningRate = readBigEndianFloat(data, offset: offset); offset += 4
        config.dropout = readBigEndianFloat(data, offset: offset); offset += 4
    }

    private func buildModel() {
        log("Building GPT-2 model: d=\(config.dModel) h=\(config.nHeads) L=\(config.nLayers) seq=\(config.seqLen)")

        let m = GPT2Model(config: config)
        MLX.eval(m)
        self.model = m

        self.optimizer = AdamW(
            learningRate: config.learningRate,
            betas: (config.beta1, config.beta2),
            eps: 1e-8,
            weightDecay: config.weightDecay
        )

        let paramCount = m.parameters().flattenedValues().reduce(0) { $0 + $1.size }
        let paramStr = paramCount >= 1_000_000
            ? String(format: "%.1fM", Double(paramCount) / 1_000_000.0)
            : String(format: "%.1fK", Double(paramCount) / 1000.0)
        log("Model built: \(paramStr) params (evaluating layers [\(stageInfo.firstLayer)..\(stageInfo.lastLayer)))")
    }

    // MARK: - Training Loop

    private func runTrainingRun(connection: NWConnection) async {
        guard let model, let optimizer else {
            log("Model not built")
            return
        }

        isTraining = true
        progress = 0
        currentStep = 0

        let capturedStageInfo = self.stageInfo
        let capturedConfig = self.config
        let M = capturedStageInfo.numMicroBatches
        let capturedTotalSteps = self.totalSteps
        let capturedWorkerId = self.workerId

        do {
            if capturedStageInfo.isHead {
                try await runHeadLoop(
                    connection: connection, model: model, optimizer: optimizer,
                    config: capturedConfig, stageInfo: capturedStageInfo,
                    M: M, totalSteps: capturedTotalSteps, workerId: capturedWorkerId
                )
            } else if capturedStageInfo.isTail {
                try await runTailLoop(
                    connection: connection, model: model, optimizer: optimizer,
                    config: capturedConfig, stageInfo: capturedStageInfo,
                    M: M, totalSteps: capturedTotalSteps, workerId: capturedWorkerId
                )
            }
        } catch is PipelineStopSignal {
            log("Server sent STOP — training run interrupted")
        } catch {
            log("Training error: \(error)")
        }
        finishTraining()
    }

    // MARK: - Head Worker Loop

    private nonisolated func runHeadLoop(
        connection: NWConnection, model: GPT2Model, optimizer: AdamW,
        config: TransformerConfig, stageInfo: PipelineStageInfo,
        M: Int, totalSteps: Int, workerId: UInt32
    ) async throws {
        let splitAt = stageInfo.lastLayer

        await MainActor.run { [weak self] in
            self?.log("HEAD loop: layers [0..\(splitAt)), \(M) micro-batches, \(totalSteps) steps")
        }

        // Wrap frontSurrogateLoss to match valueAndGrad(model:, (Model, MLXArray, MLXArray) -> MLXArray)
        let lossFn = valueAndGrad(model: model) { (model: GPT2Model, inputs: MLXArray, upstreamGrads: MLXArray) -> MLXArray in
            frontSurrogateLoss(model: model, inputs: inputs, upstreamGrads: upstreamGrads, splitAt: splitAt)
        }

        var totalForwardMs: Double = 0
        var totalBackwardMs: Double = 0
        var totalSendMs: Double = 0
        var totalRecvMs: Double = 0
        var stepCount: Int = 0

        for miniBatch in 0..<totalSteps {
            if Task.isCancelled { return }

            let clock = ContinuousClock()

            // ── Receive data batches from server (or STOP) ──
            var batchTokens: [Int: MLXArray] = [:]
            for _ in 0..<M {
                let frame = try await Self.receiveFrameOrStop(connection: connection)
                guard frame[0] == PipelineMsg.dataBatch else { continue }
                let (mbId, tokens, _) = Self.parseDataBatch(frame)
                batchTokens[mbId] = tokens
            }

            // ── Forward + send activations ──
            for m in 0..<M {
                guard let tokens = batchTokens[m] else { continue }

                let t0 = clock.now
                let act = model.forwardFrontHalf(tokens, splitAt: splitAt)
                MLX.eval(act)
                let fwdMs = Self.durationMs(from: t0, to: clock.now)
                totalForwardMs += fwdMs

                // Send activation to server for relay
                let ts = clock.now
                do {
                    let actData = try saveToData(arrays: ["activation": act])
                    var msg = Data()
                    msg.append(PipelineMsg.activation)
                    Self.appendBE(&msg, UInt32(m))
                    Self.appendBE(&msg, UInt32(stageInfo.stageIndex))
                    Self.appendBE(&msg, UInt32(stageInfo.stageIndex + 1))
                    msg.append(0) // has_targets = false (server attaches them)
                    Self.appendBE(&msg, UInt32(0)) // targets_len = 0
                    Self.appendBE(&msg, UInt32(actData.count))
                    msg.append(actData)
                    try await Self.sendFrame(connection: connection, payload: msg)
                } catch {
                    await MainActor.run { [weak self] in
                        self?.log("Head: error sending activation: \(error)")
                    }
                    return
                }
                totalSendMs += Self.durationMs(from: ts, to: clock.now)
            }

            // ── Receive gradients + backward (with gradient accumulation) ──
            var accumulatedGrads: NestedDictionary<String, MLXArray>? = nil

            for m in 0..<M {
                guard let tokens = batchTokens[m] else { continue }

                let tr = clock.now
                let gradFrame = try await Self.receiveFrameOrStop(connection: connection)
                guard gradFrame[0] == PipelineMsg.gradient else {
                    await MainActor.run { [weak self] in
                        self?.log("Head: expected GRADIENT, got 0x\(String(format: "%02x", gradFrame[0]))")
                    }
                    continue
                }
                totalRecvMs += Self.durationMs(from: tr, to: clock.now)

                // Parse gradient
                let (_, _, upstreamGrads) = Self.parseGradient(gradFrame)

                // Backward via surrogate loss (compute grads only, don't step)
                let tb = clock.now
                let (_, grads) = lossFn(model, tokens, upstreamGrads)

                // Accumulate gradients
                if accumulatedGrads == nil {
                    accumulatedGrads = grads
                } else {
                    accumulatedGrads = accumulatedGrads!.mapValues(grads, transform: { acc, g in
                        acc + (g ?? MLXArray(Float(0)))
                    })
                }
                MLX.eval(accumulatedGrads!)
                totalBackwardMs += Self.durationMs(from: tb, to: clock.now)
            }

            // ── Single optimizer step with accumulated gradients ──
            if let finalGrads = accumulatedGrads {
                // Average gradients over micro-batches
                let scale = MLXArray(Float(1.0) / Float(M))
                let avgGrads = finalGrads.mapValues(transform: { g in g * scale })
                optimizer.update(model: model, gradients: avgGrads)
                MLX.eval(model, optimizer)
            }

            // ── Sync barrier ──
            let barrierFrame = try await Self.receiveFrameOrStop(connection: connection)
            guard barrierFrame[0] == PipelineMsg.syncBarrier else {
                await MainActor.run { [weak self] in
                    self?.log("Head: expected SYNC_BARRIER")
                }
                continue
            }
            let mbId = Self.readBE32(barrierFrame, offset: 1)

            var ack = Data()
            ack.append(PipelineMsg.syncAck)
            Self.appendBE(&ack, mbId)
            Self.appendBE(&ack, workerId)
            try await Self.sendFrame(connection: connection, payload: ack)

            stepCount += 1

            // Update metrics
            let avgFwd = totalForwardMs / Double(stepCount * M)
            let avgBwd = totalBackwardMs / Double(stepCount * M)
            let avgSnd = totalSendMs / Double(stepCount * M)
            let avgRcv = totalRecvMs / Double(stepCount * M)
            let seqTime = avgFwd + avgBwd + avgSnd + avgRcv
            let efficiency = seqTime > 0 ? (avgFwd + avgBwd) / seqTime * 100 : 0

            let capturedMiniBatch = miniBatch
            await MainActor.run { [weak self] in
                self?.currentStep = capturedMiniBatch + 1
                self?.progress = Double(capturedMiniBatch + 1) / Double(totalSteps)
                self?.avgForwardMs = avgFwd
                self?.avgBackwardMs = avgBwd
                self?.avgSendMs = avgSnd
                self?.avgRecvMs = avgRcv
                self?.pipelineEfficiency = efficiency
                self?.status = String(format: "HEAD step %d/%d — fwd=%.0fms bwd=%.0fms",
                                      capturedMiniBatch + 1, totalSteps, avgFwd, avgBwd)
            }

            if miniBatch % 10 == 0 || miniBatch < 5 {
                let logMsg = String(format: "[HEAD step %d] fwd=%.1fms bwd=%.1fms send=%.1fms recv=%.1fms eff=%.0f%%",
                                    miniBatch + 1, avgFwd, avgBwd, avgSnd, avgRcv, efficiency)
                await MainActor.run { [weak self] in self?.log(logMsg) }
            }
        }
    }

    // MARK: - Tail Worker Loop

    private nonisolated func runTailLoop(
        connection: NWConnection, model: GPT2Model, optimizer: AdamW,
        config: TransformerConfig, stageInfo: PipelineStageInfo,
        M: Int, totalSteps: Int, workerId: UInt32
    ) async throws {
        let fromLayer = stageInfo.firstLayer

        await MainActor.run { [weak self] in
            self?.log("TAIL loop: layers [\(fromLayer)..\(config.nLayers)), \(M) micro-batches, \(totalSteps) steps")
        }

        var totalForwardMs: Double = 0
        var totalBackwardMs: Double = 0
        var totalSendMs: Double = 0
        var totalRecvMs: Double = 0
        var stepCount: Int = 0
        var runningLoss: Float = 0

        for miniBatch in 0..<totalSteps {
            if Task.isCancelled { return }

            let clock = ContinuousClock()
            var accumulatedGrads: NestedDictionary<String, MLXArray>? = nil

            for m in 0..<M {
                // ── Receive activation from server (or STOP) ──
                let tr = clock.now
                let receivedActivation: MLXArray
                let targets: MLXArray
                let actFrame = try await Self.receiveFrameOrStop(connection: connection)
                guard actFrame[0] == PipelineMsg.activation else {
                    await MainActor.run { [weak self] in
                        self?.log("Tail: expected ACTIVATION, got 0x\(String(format: "%02x", actFrame[0]))")
                    }
                    continue
                }
                let parsed = Self.parseActivation(actFrame)
                receivedActivation = parsed.activation
                targets = parsed.targets
                totalRecvMs += Self.durationMs(from: tr, to: clock.now)

                // ── Forward + backward ──
                let tb = clock.now

                // Compute parameter gradients via valueAndGrad
                let capturedTargets = targets
                let capturedFromLayer = fromLayer
                let paramLossFn = valueAndGrad(model: model) { (model: GPT2Model, act: MLXArray, tgt: MLXArray) -> MLXArray in
                    tailLoss(model: model, inputActivation: act, targets: tgt, fromLayer: capturedFromLayer)
                }
                let (lossValue, paramGrads) = paramLossFn(model, receivedActivation, capturedTargets)
                let lossFloat = lossValue.item(Float.self)

                // Compute activation gradient for upstream via grad
                let gradFn = grad { (act: MLXArray) -> MLXArray in
                    tailLoss(model: model, inputActivation: act, targets: capturedTargets, fromLayer: capturedFromLayer)
                }
                let activationGrad = gradFn(receivedActivation)
                MLX.eval(activationGrad)

                // Accumulate parameter gradients (defer optimizer step)
                if accumulatedGrads == nil {
                    accumulatedGrads = paramGrads
                } else {
                    accumulatedGrads = accumulatedGrads!.mapValues(paramGrads, transform: { acc, g in
                        acc + (g ?? MLXArray(Float(0)))
                    })
                }

                let bwdMs = Self.durationMs(from: tb, to: clock.now)
                totalForwardMs += bwdMs * 0.4  // approximate split
                totalBackwardMs += bwdMs * 0.6

                runningLoss = runningLoss * 0.9 + lossFloat * 0.1
                if stepCount == 0 && m == 0 { runningLoss = lossFloat }

                // ── Send loss report ──
                do {
                    var lossMsg = Data()
                    lossMsg.append(PipelineMsg.lossReport)
                    Self.appendBE(&lossMsg, UInt32(miniBatch))
                    Self.appendBE(&lossMsg, UInt32(m))
                    Self.appendBEFloat(&lossMsg, lossFloat)
                    Self.appendBE(&lossMsg, UInt32(stepCount))
                    try await Self.sendFrame(connection: connection, payload: lossMsg)
                } catch {
                    await MainActor.run { [weak self] in
                        self?.log("Tail: error sending loss report: \(error)")
                    }
                }

                // ── Send gradient upstream ──
                let ts = clock.now
                do {
                    let gradData = try saveToData(arrays: ["grad": activationGrad])
                    var gradMsg = Data()
                    gradMsg.append(PipelineMsg.gradient)
                    Self.appendBE(&gradMsg, UInt32(m))
                    Self.appendBE(&gradMsg, UInt32(stageInfo.stageIndex))
                    Self.appendBE(&gradMsg, UInt32(stageInfo.stageIndex - 1))
                    Self.appendBEFloat(&gradMsg, lossFloat)
                    Self.appendBE(&gradMsg, UInt32(gradData.count))
                    gradMsg.append(gradData)
                    try await Self.sendFrame(connection: connection, payload: gradMsg)
                } catch {
                    await MainActor.run { [weak self] in
                        self?.log("Tail: error sending gradient: \(error)")
                    }
                    return
                }
                totalSendMs += Self.durationMs(from: ts, to: clock.now)
            }

            // ── Single optimizer step with accumulated gradients ──
            if let finalGrads = accumulatedGrads {
                let scale = MLXArray(Float(1.0) / Float(M))
                let avgGrads = finalGrads.mapValues(transform: { g in g * scale })
                optimizer.update(model: model, gradients: avgGrads)
                MLX.eval(model, optimizer)
            }

            // ── Sync barrier ──
            let barrierFrame = try await Self.receiveFrameOrStop(connection: connection)
            guard barrierFrame[0] == PipelineMsg.syncBarrier else {
                await MainActor.run { [weak self] in
                    self?.log("Tail: expected SYNC_BARRIER")
                }
                continue
            }
            let mbId = Self.readBE32(barrierFrame, offset: 1)

            var ack = Data()
            ack.append(PipelineMsg.syncAck)
            Self.appendBE(&ack, mbId)
            Self.appendBE(&ack, workerId)
            try await Self.sendFrame(connection: connection, payload: ack)

            stepCount += 1

            // Update metrics
            let totalMicroSteps = Double(stepCount * M)
            let avgFwd = totalForwardMs / totalMicroSteps
            let avgBwd = totalBackwardMs / totalMicroSteps
            let avgSnd = totalSendMs / totalMicroSteps
            let avgRcv = totalRecvMs / totalMicroSteps
            let seqTime = avgFwd + avgBwd + avgSnd + avgRcv
            let efficiency = seqTime > 0 ? (avgFwd + avgBwd) / seqTime * 100 : 0

            let capturedMiniBatch = miniBatch
            let capturedLoss = runningLoss
            await MainActor.run { [weak self] in
                self?.currentStep = capturedMiniBatch + 1
                self?.currentLoss = capturedLoss
                self?.progress = Double(capturedMiniBatch + 1) / Double(totalSteps)
                self?.avgForwardMs = avgFwd
                self?.avgBackwardMs = avgBwd
                self?.avgSendMs = avgSnd
                self?.avgRecvMs = avgRcv
                self?.pipelineEfficiency = efficiency
                self?.status = String(format: "TAIL step %d/%d — loss=%.4f eff=%.0f%%",
                                      capturedMiniBatch + 1, totalSteps, capturedLoss, efficiency)
            }

            if miniBatch % 10 == 0 || miniBatch < 5 {
                let logMsg = String(format: "[TAIL step %d] loss=%.4f fwd=%.1fms bwd=%.1fms send=%.1fms recv=%.1fms eff=%.0f%%",
                                    miniBatch + 1, runningLoss, avgFwd, avgBwd, avgSnd, avgRcv, efficiency)
                await MainActor.run { [weak self] in self?.log(logMsg) }
            }
        }
    }

    // MARK: - Message Parsing

    private nonisolated static func parseDataBatch(_ data: Data) -> (Int, MLXArray, MLXArray) {
        var offset = 1 // skip type
        let mbId = Int(readBE32(data, offset: offset)); offset += 4
        let _ = readBE32(data, offset: offset); offset += 4 // mini_batch_id
        let B = Int(readBE32(data, offset: offset)); offset += 4
        let S = Int(readBE32(data, offset: offset)); offset += 4

        let tokenCount = B * S
        var tokenValues = [Int32](repeating: 0, count: tokenCount)
        for i in 0..<tokenCount {
            tokenValues[i] = Int32(bigEndian: data.withUnsafeBytes {
                $0.loadUnaligned(fromByteOffset: offset + i * 4, as: Int32.self)
            })
        }
        offset += tokenCount * 4

        var targetValues = [Int32](repeating: 0, count: tokenCount)
        for i in 0..<tokenCount {
            targetValues[i] = Int32(bigEndian: data.withUnsafeBytes {
                $0.loadUnaligned(fromByteOffset: offset + i * 4, as: Int32.self)
            })
        }

        let tokens = MLXArray(tokenValues).reshaped(B, S)
        let targets = MLXArray(targetValues).reshaped(B, S)
        return (mbId, tokens, targets)
    }

    private nonisolated static func parseActivation(_ data: Data) -> (microBatchId: Int, activation: MLXArray, targets: MLXArray) {
        var offset = 1 // skip type
        let mbId = Int(readBE32(data, offset: offset)); offset += 4
        let _ = readBE32(data, offset: offset); offset += 4 // source_stage
        let _ = readBE32(data, offset: offset); offset += 4 // dest_stage

        let hasTargets = data[offset] != 0; offset += 1
        let targetsLen = Int(readBE32(data, offset: offset)); offset += 4

        var targets = MLXArray(Int32(0))
        if hasTargets && targetsLen > 0 {
            let tgtCount = targetsLen / 4
            var tgtValues = [Int32](repeating: 0, count: tgtCount)
            for i in 0..<tgtCount {
                tgtValues[i] = Int32(bigEndian: data.withUnsafeBytes {
                    $0.loadUnaligned(fromByteOffset: offset + i * 4, as: Int32.self)
                })
            }
            targets = MLXArray(tgtValues)
            offset += targetsLen
        }

        let stLen = Int(readBE32(data, offset: offset)); offset += 4
        let stData = data.subdata(in: offset..<(offset + stLen))

        let arrays = try! loadArrays(data: stData)
        let activation = arrays["activation"]!

        // Reshape targets to match activation's B, S
        if hasTargets {
            let B = activation.dim(0)
            let S = activation.dim(1)
            targets = targets.reshaped(B, S)
        }

        return (mbId, activation, targets)
    }

    private nonisolated static func parseGradient(_ data: Data) -> (microBatchId: Int, lossValue: Float, grad: MLXArray) {
        var offset = 1 // skip type
        let mbId = Int(readBE32(data, offset: offset)); offset += 4
        let _ = readBE32(data, offset: offset); offset += 4 // source_stage
        let _ = readBE32(data, offset: offset); offset += 4 // dest_stage
        let lossValue = readBEFloat(data, offset: offset); offset += 4
        let stLen = Int(readBE32(data, offset: offset)); offset += 4
        let stData = data.subdata(in: offset..<(offset + stLen))

        let arrays = try! loadArrays(data: stData)
        let gradient = arrays["grad"]!
        return (mbId, lossValue, gradient)
    }

    // MARK: - Internal Helpers

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

    func clearLog() {
        trainingLog.removeAll()
    }

    func stopTraining() {
        trainingTask?.cancel()
        trainingTask = nil
    }

    // MARK: - Big-Endian Helpers

    private func appendBigEndian(_ data: inout Data, _ value: UInt32) {
        withUnsafeBytes(of: value.bigEndian) { data.append(contentsOf: $0) }
    }

    private nonisolated static func appendBE(_ data: inout Data, _ value: UInt32) {
        withUnsafeBytes(of: value.bigEndian) { data.append(contentsOf: $0) }
    }

    private nonisolated static func appendBEFloat(_ data: inout Data, _ value: Float) {
        withUnsafeBytes(of: value.bitPattern.bigEndian) { data.append(contentsOf: $0) }
    }

    private nonisolated static func readBE32(_ data: Data, offset: Int) -> UInt32 {
        data.withUnsafeBytes { ptr in
            UInt32(bigEndian: ptr.loadUnaligned(fromByteOffset: offset, as: UInt32.self))
        }
    }

    private nonisolated static func readBEFloat(_ data: Data, offset: Int) -> Float {
        let bits = data.withUnsafeBytes { ptr in
            UInt32(bigEndian: ptr.loadUnaligned(fromByteOffset: offset, as: UInt32.self))
        }
        return Float(bitPattern: bits)
    }

    // MARK: - Network I/O

    nonisolated static func sendFrame(connection: NWConnection, payload: Data) async throws {
        let frame = withUnsafeBytes(of: UInt32(payload.count).bigEndian) { Data($0) } + payload
        try await withCheckedThrowingContinuation { (cont: CheckedContinuation<Void, Error>) in
            connection.send(content: frame, completion: .contentProcessed { error in
                if let error { cont.resume(throwing: error) }
                else { cont.resume() }
            })
        }
    }

    nonisolated static func receiveFrame(connection: NWConnection) async throws -> Data {
        let lengthData = try await receiveExactly(connection: connection, count: 4)
        let payloadLength = lengthData.withUnsafeBytes { ptr in
            UInt32(bigEndian: ptr.loadUnaligned(as: UInt32.self))
        }
        return try await receiveExactly(connection: connection, count: Int(payloadLength))
    }

    /// Receives a frame but throws PipelineStopSignal if it's a PIPELINE_STOP message.
    nonisolated static func receiveFrameOrStop(connection: NWConnection) async throws -> Data {
        let frame = try await receiveFrame(connection: connection)
        guard !frame.isEmpty, frame[0] != PipelineMsg.stop else {
            throw PipelineStopSignal()
        }
        return frame
    }

    private nonisolated static func receiveExactly(connection: NWConnection, count: Int) async throws -> Data {
        var buffer = Data()
        while buffer.count < count {
            let remaining = count - buffer.count
            let chunk: Data = try await withCheckedThrowingContinuation { cont in
                connection.receive(minimumIncompleteLength: 1, maximumLength: remaining) { data, _, _, error in
                    if let error { cont.resume(throwing: error) }
                    else if let data, !data.isEmpty { cont.resume(returning: data) }
                    else { cont.resume(throwing: NWError.posix(.ECONNRESET)) }
                }
            }
            buffer.append(chunk)
        }
        return buffer
    }

    private nonisolated static func durationMs(from start: ContinuousClock.Instant, to end: ContinuousClock.Instant) -> Double {
        let d = start.duration(to: end)
        let (seconds, attoseconds) = d.components
        return Double(seconds) * 1000.0 + Double(attoseconds) / 1_000_000_000_000_000.0
    }
}

// MARK: - Free function helpers for big-endian parsing (used in instance methods)

private func readBigEndianUInt32(_ data: Data, offset: Int) -> UInt32 {
    data.withUnsafeBytes { ptr in
        UInt32(bigEndian: ptr.loadUnaligned(fromByteOffset: offset, as: UInt32.self))
    }
}

private func readBigEndianFloat(_ data: Data, offset: Int) -> Float {
    let bits = data.withUnsafeBytes { ptr in
        UInt32(bigEndian: ptr.loadUnaligned(fromByteOffset: offset, as: UInt32.self))
    }
    return Float(bitPattern: bits)
}

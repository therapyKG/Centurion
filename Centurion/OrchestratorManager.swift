import CenturionMLX
import CommonCrypto
import Foundation
import Network
import OSLog

// MARK: - Orchestrator Message Types

private enum OrchMsg {
    // Orchestrator → Server
    nonisolated static let updateConfig: UInt8 = 0x90
    nonisolated static let startTraining: UInt8 = 0x91
    nonisolated static let stopTraining: UInt8 = 0x92
    nonisolated static let getStatus: UInt8 = 0x93
    nonisolated static let identify: UInt8 = 0x94
    nonisolated static let allowWorker: UInt8 = 0x95
    nonisolated static let restartServer: UInt8 = 0x96

    // Server → Orchestrator
    nonisolated static let statusReport: UInt8 = 0xA0
    nonisolated static let configAck: UInt8 = 0xA1
    nonisolated static let trainingStarted: UInt8 = 0xA2
    nonisolated static let trainingStopped: UInt8 = 0xA3
    nonisolated static let lossUpdate: UInt8 = 0xA4
    nonisolated static let error: UInt8 = 0xA5
    nonisolated static let allowWorkerAck: UInt8 = 0xA6
    nonisolated static let restartAck: UInt8 = 0xA7
    nonisolated static let profileReport: UInt8 = 0xA8

    // Auth (shared)
    nonisolated static let authChallenge: UInt8 = 0x30
    nonisolated static let authResponse: UInt8 = 0x31
    nonisolated static let authResult: UInt8 = 0x32
}

// MARK: - Server State

enum ServerState: String {
    case idle = "Idle"
    case configuring = "Configuring"
    case training = "Training"

    init(byte: UInt8) {
        switch byte {
        case 1: self = .configuring
        case 2: self = .training
        default: self = .idle
        }
    }
}

// MARK: - Worker Info (from status reports)

struct WorkerStatus: Identifiable {
    var id: UInt32 { workerId }
    let workerId: UInt32
    let deviceType: UInt32
    let memoryMB: UInt32
    let stageIndex: UInt32

    var deviceName: String {
        switch deviceType {
        case 1: return "iPhone"
        case 2: return "iPad"
        case 3: return "Mac"
        default: return "Unknown"
        }
    }
}

// MARK: - Worker Profile Result (from profiling phase)

struct WorkerProfileResult: Identifiable {
    var id: UInt32 { workerId }
    let workerId: UInt32
    let deviceType: UInt32
    let availableMemoryMB: UInt32
    let computeSpeed: Float     // layers/sec
    let assignedLayers: UInt32
    let firstLayer: UInt32
    let lastLayer: UInt32
    let isHead: Bool
    let isTail: Bool
    let estimatedStepMs: Float
    let maxLayers: UInt32
    let rttMs: Float

    var deviceName: String {
        switch deviceType {
        case 1: return "iPhone"
        case 2: return "iPad"
        case 3: return "Mac"
        default: return "Unknown"
        }
    }

    var roleName: String {
        if isHead { return "HEAD" }
        if isTail { return "TAIL" }
        return "MID"
    }
}

// MARK: - Orchestrator Manager

@MainActor
@Observable
final class OrchestratorManager {

    // MARK: - Observable State

    var status: String = "Configure and connect to orchestrator port."
    var isConnected: Bool = false
    var authFailed: Bool = false
    var workerBypassReady: Bool = false

    var serverState: ServerState = .idle
    var connectedWorkers: [WorkerStatus] = []
    var currentStep: Int = 0
    var totalSteps: Int = 0
    var latestLoss: Float = 0
    var lossHistory: [Float] = []

    // Profiling results
    var profilingResults: [WorkerProfileResult] = []
    var isProfiling: Bool = false

    var pipelineStageCount: Int {
        if !profilingResults.isEmpty {
            return profilingResults.count
        }
        return connectedWorkers.filter { $0.stageIndex != UInt32.max }.count
    }

    var pipelineBubbleEfficiency: Double? {
        let stages = pipelineStageCount
        guard stages > 0, microBatches > 0 else { return nil }
        return Double(microBatches) / Double(microBatches + stages - 1)
    }

    var pipelineBubbleFraction: Double? {
        guard let pipelineBubbleEfficiency else { return nil }
        return 1.0 - pipelineBubbleEfficiency
    }

    var pipelineStageBalance: Double? {
        guard !profilingResults.isEmpty else { return nil }
        let stageTimes = profilingResults.map { Double($0.estimatedStepMs) }.filter { $0 > 0 }
        guard stageTimes.count == profilingResults.count,
              let maxStageTime = stageTimes.max(),
              maxStageTime > 0 else { return nil }
        return stageTimes.reduce(0, +) / (Double(stageTimes.count) * maxStageTime)
    }

    var pipelineUtilization: Double? {
        guard let pipelineBubbleEfficiency else { return nil }
        return pipelineBubbleEfficiency * (pipelineStageBalance ?? 1.0)
    }

    // Editable config (GPT-2 Medium defaults)
    var dModel: Int = 1024
    var nHeads: Int = 16
    var nLayers: Int = 24
    var seqLen: Int = 128
    var batchSize: Int = 1
    var microBatches: Int = 4
    var configTotalSteps: Int = 200
    var learningRate: Float = 3e-4
    var vocabSize: Int = 50257
    var ffnHiddenMul: Int = 4
    var dropout: Float = 0.1

    var trainingLog: [GPTTrainingLogEntry] = []

    // Server connection
    var serverHost: String = "34.60.122.134"
    var serverPort: UInt16 = 9998
    var serverSecret: String = ""

    /// Called when the orchestrator disconnects (spontaneous or explicit).
    /// ConnectionView observes this to clean up worker-bypass state.
    var onDisconnect: (() -> Void)?

    /// Continuation for awaiting the worker bypass acknowledgment.
    private var bypassContinuation: CheckedContinuation<Bool, Never>?

    // Internal
    private var connection: NWConnection?
    private var messageTask: Task<Void, Never>?
    private var pollTask: Task<Void, Never>?
    private let logger = Logger(subsystem: "Centurion", category: "Orchestrator")

    // MARK: - Connection

    func connect() {
        disconnect()

        guard !serverSecret.isEmpty else {
            status = "Enter orchestrator secret first."
            log("Error: secret is empty")
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
                        // Send ORCH_IDENTIFY so server knows this is an orchestrator (not a worker)
                        do {
                            var identifyMsg = Data()
                            identifyMsg.append(OrchMsg.identify)
                            try await Self.sendFrame(connection: conn, payload: identifyMsg)
                        } catch {
                            self?.log("Failed to send identify: \(error)")
                            conn.cancel()
                            return
                        }
                        self?.isConnected = true
                        self?.status = "Connected to orchestrator port."
                        self?.log("Authenticated & identified. Listening for server messages...")
                        self?.startMessageLoop(connection: conn)
                        self?.startPolling(connection: conn)
                    } else {
                        self?.isConnected = false
                        self?.authFailed = true
                        self?.status = "Authentication failed — wrong secret"
                        self?.log("Auth failed")
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

        conn.start(queue: DispatchQueue(label: "com.centurion.orchestrator", qos: .userInitiated))
    }

    func disconnect() {
        messageTask?.cancel()
        messageTask = nil
        pollTask?.cancel()
        pollTask = nil
        connection?.cancel()
        connection = nil

        let wasConnected = isConnected
        isConnected = false
        workerBypassReady = false

        // Clear server-side state so UI doesn't show stale data
        serverState = .idle
        connectedWorkers = []
        currentStep = 0
        totalSteps = 0
        latestLoss = 0
        lossHistory = []
        profilingResults = []
        isProfiling = false

        // Cancel any pending bypass wait
        bypassContinuation?.resume(returning: false)
        bypassContinuation = nil

        if wasConnected {
            onDisconnect?()
        }
    }

    // MARK: - Authentication

    private func performAuth(connection: NWConnection, secret: String) async -> Bool {
        do {
            let challenge = try await Self.receiveFrame(connection: connection)
            guard challenge.count == 33, challenge[0] == OrchMsg.authChallenge else {
                log("Bad auth challenge: len=\(challenge.count)")
                return false
            }
            let nonce = challenge.subdata(in: 1..<33)
            let mac = Self.hmacSHA256(key: Data(secret.utf8), data: nonce)

            var response = Data()
            response.append(OrchMsg.authResponse)
            response.append(mac)
            try await Self.sendFrame(connection: connection, payload: response)

            let result = try await Self.receiveFrame(connection: connection)
            guard result.count == 2, result[0] == OrchMsg.authResult else {
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

    // MARK: - Commands

    func updateConfig() {
        guard let connection, isConnected else { return }
        Task {
            do {
                var msg = Data()
                msg.append(OrchMsg.updateConfig)
                appendBE(&msg, UInt32(vocabSize))
                appendBE(&msg, UInt32(dModel))
                appendBE(&msg, UInt32(nHeads))
                appendBE(&msg, UInt32(nLayers))
                appendBE(&msg, UInt32(seqLen))
                appendBE(&msg, UInt32(batchSize))
                appendBE(&msg, UInt32(ffnHiddenMul))
                appendBE(&msg, UInt32(microBatches))
                appendBE(&msg, UInt32(configTotalSteps))
                appendBEFloat(&msg, learningRate)
                appendBEFloat(&msg, dropout)
                try await Self.sendFrame(connection: connection, payload: msg)
                log("Config update sent")
            } catch {
                log("Failed to send config: \(error)")
            }
        }
    }

    func startTraining() {
        guard let connection, isConnected else { return }
        isProfiling = true
        profilingResults = []
        status = "Profiling workers..."
        Task {
            do {
                var msg = Data()
                msg.append(OrchMsg.startTraining)
                try await Self.sendFrame(connection: connection, payload: msg)
                log("Start training requested (profiling phase first)")
            } catch {
                log("Failed to send start: \(error)")
                isProfiling = false
            }
        }
    }

    func stopTraining() {
        guard let connection, isConnected else { return }
        Task {
            do {
                var msg = Data()
                msg.append(OrchMsg.stopTraining)
                try await Self.sendFrame(connection: connection, payload: msg)
                log("Stop training requested")
            } catch {
                log("Failed to send stop: \(error)")
            }
        }
    }

    func restartServer() {
        guard let connection, isConnected else { return }
        Task {
            do {
                var msg = Data()
                msg.append(OrchMsg.restartServer)
                try await Self.sendFrame(connection: connection, payload: msg)
                log("Server restart requested")
            } catch {
                log("Failed to send restart: \(error)")
            }
        }
    }

    func requestStatus() {
        guard let connection, isConnected else { return }
        Task {
            do {
                var msg = Data()
                msg.append(OrchMsg.getStatus)
                try await Self.sendFrame(connection: connection, payload: msg)
            } catch {
                log("Failed to request status: \(error)")
            }
        }
    }

    /// Ask the server to allow our IP to connect as a worker without HMAC auth.
    /// Returns `true` if the server acknowledged the bypass, `false` on failure/timeout.
    func requestWorkerBypass() async -> Bool {
        guard let connection, isConnected else { return false }
        workerBypassReady = false

        // Cancel any previous pending bypass wait
        bypassContinuation?.resume(returning: false)
        bypassContinuation = nil

        do {
            var msg = Data()
            msg.append(OrchMsg.allowWorker)
            try await Self.sendFrame(connection: connection, payload: msg)
            log("Requested worker auth bypass")
        } catch {
            log("Failed to request worker bypass: \(error)")
            return false
        }

        // Wait for the message loop to deliver the ACK, with a 10s timeout
        let result = await withTaskGroup(of: Bool.self) { group in
            group.addTask { @MainActor in
                await withCheckedContinuation { (cont: CheckedContinuation<Bool, Never>) in
                    self.bypassContinuation = cont
                }
            }
            group.addTask {
                try? await Task.sleep(for: .seconds(10))
                return false
            }
            let first = await group.next() ?? false
            group.cancelAll()
            return first
        }

        if !result {
            // Timeout or failure — clean up
            bypassContinuation?.resume(returning: false)
            bypassContinuation = nil
        }

        return result
    }

    // MARK: - Message Loop

    private func startMessageLoop(connection: NWConnection) {
        messageTask = Task.detached(priority: .userInitiated) { [weak self] in
            do {
                while !Task.isCancelled {
                    let frame = try await Self.receiveFrame(connection: connection)
                    await self?.handleMessage(frame)
                }
            } catch {
                await MainActor.run { [weak self] in
                    guard let self else { return }
                    self.log("Message loop ended: \(error)")
                    self.status = "Disconnected: \(error.localizedDescription)"
                    // Full cleanup including notifying listeners
                    self.disconnect()
                }
            }
        }
    }

    private func startPolling(connection: NWConnection) {
        pollTask = Task { [weak self] in
            while !Task.isCancelled {
                try? await Task.sleep(for: .seconds(3))
                self?.requestStatus()
            }
        }
    }

    private func handleMessage(_ data: Data) {
        guard !data.isEmpty else { return }
        let msgType = data[0]

        switch msgType {
        case OrchMsg.statusReport:
            parseStatusReport(data)

        case OrchMsg.configAck:
            let ok = data.count >= 2 && data[1] == 0
            log("Config ACK: \(ok ? "OK" : "FAIL")")
            status = ok ? "Config updated on server." : "Config update failed."

        case OrchMsg.trainingStarted:
            log("Training started by server")
            serverState = .training
            isProfiling = false
            lossHistory.removeAll()
            status = "Training in progress..."

        case OrchMsg.trainingStopped:
            if data.count >= 9 {
                let steps = Int(readBE32(data, offset: 1))
                let loss = readBEFloat(data, offset: 5)
                log("Training stopped: \(steps) steps, final loss \(String(format: "%.4f", loss))")
            }
            serverState = .idle
            status = "Training complete. Server idle."

        case OrchMsg.lossUpdate:
            if data.count >= 9 {
                let step = Int(readBE32(data, offset: 1))
                let loss = readBEFloat(data, offset: 5)
                currentStep = step
                latestLoss = loss
                lossHistory.append(loss)
            }

        case OrchMsg.allowWorkerAck:
            let ok = data.count >= 2 && data[1] == 0
            workerBypassReady = ok
            if ok {
                log("Server ready — worker bypass active")
            } else {
                log("Server denied worker bypass")
            }
            // Resume anyone awaiting the bypass result
            bypassContinuation?.resume(returning: ok)
            bypassContinuation = nil

        case OrchMsg.restartAck:
            log("Server restart complete")
            connectedWorkers = []
            serverState = .idle
            currentStep = 0
            totalSteps = 0
            latestLoss = 0
            lossHistory = []
            profilingResults = []
            isProfiling = false
            status = "Server restarted. Workers disconnected."

        case OrchMsg.error:
            isProfiling = false
            serverState = .idle
            if data.count >= 5 {
                let msgLen = Int(readBE32(data, offset: 1))
                if data.count >= 5 + msgLen {
                    let errMsg = String(data: data.subdata(in: 5..<(5 + msgLen)), encoding: .utf8) ?? "Unknown"
                    log("Server error: \(errMsg)")
                    status = "Error: \(errMsg)"
                }
            }

        case OrchMsg.profileReport:
            parseProfileReport(data)

        default:
            log("Unknown message: 0x\(String(format: "%02x", msgType))")
        }
    }

    private func parseProfileReport(_ data: Data) {
        guard data.count >= 5 else { return }
        var offset = 1

        let numWorkers = Int(readBE32(data, offset: offset)); offset += 4

        var results: [WorkerProfileResult] = []
        // Each worker entry: 4+4 + 4 + 4 + 4 + 4+4 + 1+1 + 4 + 4 + 4 = 42 bytes
        for _ in 0..<numWorkers {
            guard data.count >= offset + 42 else { break }

            let wId = readBE32(data, offset: offset); offset += 4
            let dType = readBE32(data, offset: offset); offset += 4
            let availMB = readBE32(data, offset: offset); offset += 4
            let speed = readBEFloat(data, offset: offset); offset += 4
            let assigned = readBE32(data, offset: offset); offset += 4
            let first = readBE32(data, offset: offset); offset += 4
            let last = readBE32(data, offset: offset); offset += 4
            let isHead = data[offset] != 0; offset += 1
            let isTail = data[offset] != 0; offset += 1
            let estStep = readBEFloat(data, offset: offset); offset += 4
            let maxL = readBE32(data, offset: offset); offset += 4
            let rtt = readBEFloat(data, offset: offset); offset += 4

            results.append(WorkerProfileResult(
                workerId: wId, deviceType: dType,
                availableMemoryMB: availMB, computeSpeed: speed,
                assignedLayers: assigned, firstLayer: first, lastLayer: last,
                isHead: isHead, isTail: isTail,
                estimatedStepMs: estStep, maxLayers: maxL, rttMs: rtt
            ))
        }

        profilingResults = results
        isProfiling = false
        status = "Profiling complete. Training starting..."

        for r in results {
            log(String(format: "%@ W%d (%@): %d layers [%d..%d), %.1f L/s, %dMB avail, est %.0fms/step",
                        r.roleName, r.workerId, r.deviceName,
                        r.assignedLayers, r.firstLayer, r.lastLayer,
                        r.computeSpeed, r.availableMemoryMB, r.estimatedStepMs))
        }
    }

    private func parseStatusReport(_ data: Data) {
        guard data.count >= 18 else { return }
        var offset = 1

        let stateByte = data[offset]; offset += 1
        serverState = ServerState(byte: stateByte)

        let numWorkers = Int(readBE32(data, offset: offset)); offset += 4
        currentStep = Int(readBE32(data, offset: offset)); offset += 4
        totalSteps = Int(readBE32(data, offset: offset)); offset += 4
        latestLoss = readBEFloat(data, offset: offset); offset += 4

        var workers: [WorkerStatus] = []
        for _ in 0..<numWorkers {
            guard data.count >= offset + 16 else { break }
            let wId = readBE32(data, offset: offset); offset += 4
            let dType = readBE32(data, offset: offset); offset += 4
            let mem = readBE32(data, offset: offset); offset += 4
            let stage = readBE32(data, offset: offset); offset += 4
            workers.append(WorkerStatus(workerId: wId, deviceType: dType, memoryMB: mem, stageIndex: stage))
        }
        connectedWorkers = workers

        if serverState != .training {
            status = "Server \(serverState.rawValue). \(numWorkers) worker(s) connected."
        }
    }

    // MARK: - Logging

    func log(_ message: String) {
        let entry = GPTTrainingLogEntry(timestamp: Date(), message: message)
        trainingLog.append(entry)
        logger.info("\(message, privacy: .public)")
    }

    func clearLog() {
        trainingLog.removeAll()
    }

    // MARK: - Big-Endian Helpers

    private func appendBE(_ data: inout Data, _ value: UInt32) {
        withUnsafeBytes(of: value.bigEndian) { data.append(contentsOf: $0) }
    }

    private func appendBEFloat(_ data: inout Data, _ value: Float) {
        withUnsafeBytes(of: value.bitPattern.bigEndian) { data.append(contentsOf: $0) }
    }

    private func readBE32(_ data: Data, offset: Int) -> UInt32 {
        data.withUnsafeBytes { ptr in
            UInt32(bigEndian: ptr.loadUnaligned(fromByteOffset: offset, as: UInt32.self))
        }
    }

    private func readBEFloat(_ data: Data, offset: Int) -> Float {
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
}

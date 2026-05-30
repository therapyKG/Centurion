import CenturionMLX
import Charts
import SwiftUI

struct OrchestratorView: View {
    @Bindable var manager: OrchestratorManager

    var body: some View {
        List {
            if manager.isConnected {
                // MARK: - Server State
                Section("Server State") {
                    LabeledContent("State") {
                        Text(manager.serverState.rawValue)
                            .fontWeight(.semibold)
                            .foregroundStyle(stateColor)
                    }
                }

                // MARK: - Workers
                Section("Connected Workers (\(manager.connectedWorkers.count))") {
                    if manager.connectedWorkers.isEmpty {
                        Text("No workers connected")
                            .foregroundStyle(.secondary)
                    } else {
                        ForEach(manager.connectedWorkers) { worker in
                            HStack {
                                Image(systemName: worker.deviceType == 2 ? "ipad" : "iphone")
                                    .foregroundStyle(.blue)
                                VStack(alignment: .leading) {
                                    Text("\(worker.deviceName)")
                                        .font(.body)
                                    Text("ID: \(worker.workerId) • \(worker.memoryMB) MB • \(worker.stageIndex == 0xFFFFFFFF ? "Unassigned" : "Stage \(worker.stageIndex)")")
                                        .font(.caption)
                                        .foregroundStyle(.secondary)
                                }
                            }
                        }
                    }
                }

                // MARK: - Model Config
                Section("Model Config") {
                    configField("d_model", value: $manager.dModel)
                    configField("Heads", value: $manager.nHeads)
                    configField("Layers", value: $manager.nLayers)
                    configField("Seq Length", value: $manager.seqLen)
                    configField("Batch Size", value: $manager.batchSize)
                    configField("Micro-batches", value: $manager.microBatches)
                    configField("Total Steps", value: $manager.configTotalSteps)
                    LabeledContent("Learning Rate") {
                        TextField("LR", value: $manager.learningRate, format: .number)
                            .keyboardType(.decimalPad)
                            .multilineTextAlignment(.trailing)
                            .frame(width: 100)
                    }

                    Button("Update Config") {
                        UIApplication.shared.sendAction(#selector(UIResponder.resignFirstResponder), to: nil, from: nil, for: nil)
                        manager.updateConfig()
                    }
                    .buttonStyle(.bordered)
                    .disabled(manager.serverState == .training)
                }

                // MARK: - Training Control
                Section("Training Control") {
                    HStack {
                        Button("Start Training") {
                            manager.startTraining()
                        }
                        .buttonStyle(.borderedProminent)
                        .tint(.green)
                        .disabled(
                            manager.serverState != .idle ||
                            manager.connectedWorkers.count < 2
                        )

                        Button("Stop Training") {
                            manager.stopTraining()
                        }
                        .buttonStyle(.bordered)
                        .tint(.red)
                        .disabled(manager.serverState != .training)
                    }

                    if manager.connectedWorkers.count < 2 && manager.serverState == .idle {
                        Text("Need at least 2 workers to start training")
                            .font(.caption)
                            .foregroundStyle(.orange)
                    }
                }

                // MARK: - Live Metrics
                if manager.serverState == .training || !manager.lossHistory.isEmpty {
                    Section("Training Metrics") {
                        LabeledContent("Step") {
                            Text("\(manager.currentStep)/\(manager.totalSteps)")
                                .monospacedDigit()
                        }

                        if manager.latestLoss > 0 {
                            LabeledContent("Latest Loss") {
                                Text(String(format: "%.4f", manager.latestLoss))
                                    .monospacedDigit()
                            }
                        }

                        if manager.totalSteps > 0 {
                            ProgressView(value: Double(manager.currentStep), total: Double(manager.totalSteps))
                        }

                        if manager.lossHistory.count >= 2 {
                            Chart {
                                ForEach(Array(manager.lossHistory.enumerated()), id: \.offset) { idx, loss in
                                    LineMark(
                                        x: .value("Step", idx),
                                        y: .value("Loss", loss)
                                    )
                                    .foregroundStyle(.blue)
                                }
                            }
                            .frame(height: 150)
                            .chartYAxisLabel("Loss")
                            .chartXAxisLabel("Update")
                        }
                    }
                }
            }

            // MARK: - Status
            Section {
                Text(manager.status)
                    .font(.callout)
            } header: {
                Text("Status")
            }

            // MARK: - Log
            if !manager.trainingLog.isEmpty {
                Section {
                    ForEach(manager.trainingLog) { entry in
                        HStack(alignment: .top, spacing: 8) {
                            Text(entry.timestamp, format: .dateTime.hour().minute().second().secondFraction(.fractional(2)))
                                .font(.caption2.monospaced())
                                .foregroundStyle(.secondary)
                            Text(entry.message)
                                .font(.caption.monospaced())
                        }
                    }
                } header: {
                    HStack {
                        Text("Log")
                        Spacer()
                        Button("Clear") { manager.clearLog() }
                            .font(.caption)
                    }
                }
            }
        }
        .navigationTitle("Orchestrator")
        .toolbar {
            ToolbarItemGroup(placement: .keyboard) {
                Spacer()
                Button("Done") {
                    UIApplication.shared.sendAction(#selector(UIResponder.resignFirstResponder), to: nil, from: nil, for: nil)
                }
            }
        }
        .scrollDismissesKeyboard(.interactively)
    }

    // MARK: - Helpers

    private var stateColor: Color {
        switch manager.serverState {
        case .idle: return .green
        case .configuring: return .orange
        case .training: return .blue
        }
    }

    private func configField(_ label: String, value: Binding<Int>) -> some View {
        LabeledContent(label) {
            TextField(label, value: value, format: .number)
                .keyboardType(.numberPad)
                .multilineTextAlignment(.trailing)
                .frame(width: 100)
        }
    }
}

#Preview {
    NavigationStack { OrchestratorView(manager: OrchestratorManager()) }
}

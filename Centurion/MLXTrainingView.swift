import CenturionMLX
import SwiftUI

struct MLXTrainingView: View {
    @State private var manager = MLXTrainingManager()
    @State private var monitor = HardwareMonitor()
    @State private var epochs: Int = 20
    @State private var showPerformanceStats = true

    var body: some View {
        List {
            Section("Dataset") {
                Picker("Corpus", selection: $manager.selectedDataset) {
                    ForEach(BuiltinDataset.allCases) { ds in
                        Text(ds.rawValue).tag(ds)
                    }
                }
                Toggle("GPT-2 BPE Tokenizer", isOn: $manager.useBPETokenizer)
                    .disabled(manager.isTraining)
            }

            Section("Architecture") {
                LabeledContent("d_model") {
                    Stepper("\(manager.config.dModel)", value: $manager.config.dModel, in: 32...1024, step: 64)
                }
                LabeledContent("Heads") {
                    Stepper("\(manager.config.nHeads)", value: $manager.config.nHeads, in: 1...16)
                }
                LabeledContent("Layers") {
                    Stepper("\(manager.config.nLayers)", value: $manager.config.nLayers, in: 1...24)
                }
                LabeledContent("Seq Length") {
                    Stepper("\(manager.config.seqLen)", value: $manager.config.seqLen, in: 16...1024, step: 16)
                }
                LabeledContent("Batch Size") {
                    Stepper("\(manager.config.batchSize)", value: $manager.config.batchSize, in: 1...64)
                }

                Button("GPT-2 Small Preset") {
                    manager.config = .gpt2Small
                }
                .font(.caption)

                Button("Load Dataset & Build Model") {
                    manager.loadAndBuild()
                }
                .buttonStyle(.borderedProminent)
                .disabled(manager.isTraining)
            }

            Section("Training") {
                Stepper("Epochs: \(epochs)", value: $epochs, in: 1...500)
                Stepper("Steps/epoch: \(manager.stepsPerEpoch)", value: $manager.stepsPerEpoch, in: 10...500, step: 10)

                HStack {
                    Button(manager.isTraining ? "Training…" : "Train") {
                        manager.startTraining(epochs: epochs)
                    }
                    .buttonStyle(.borderedProminent)
                    .disabled(!manager.isBuilt || manager.isTraining)

                    if manager.isTraining {
                        Button("Stop") {
                            manager.stopTraining()
                        }
                        .buttonStyle(.bordered)
                        .tint(.red)
                    }
                }

                if manager.isTraining || manager.progress > 0 {
                    ProgressView(value: manager.progress) {
                        Text(manager.status)
                            .font(.caption)
                    }
                }
            }

            if manager.currentLoss > 0 {
                Section("Metrics") {
                    LabeledContent("Epoch", value: "\(manager.currentEpoch)/\(manager.totalEpochs)")
                    LabeledContent("Loss", value: String(format: "%.4f", manager.currentLoss))
                    LabeledContent("Throughput", value: String(format: "%.0f tok/s", manager.tokensPerSecond))
                }
            }

            Section {
                DisclosureGroup(isExpanded: $showPerformanceStats) {
                    HardwareDiagnosticsView(monitor: monitor)
                    Button("Reset Peak Memory") {
                        monitor.resetPeakMemory()
                    }
                    .font(.caption)
                    .foregroundStyle(.secondary)
                } label: {
                    Label("Performance", systemImage: "gauge.with.dots.needle.33percent")
                }
            }

            if !manager.generatedSample.isEmpty {
                Section("Generated Sample") {
                    Text(manager.generatedSample)
                        .font(.caption.monospaced())
                        .lineLimit(nil)
                        .textSelection(.enabled)
                }
            }

            Section {
                Text(manager.status)
                    .font(.callout)
            } header: {
                Text("Status")
            }

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
                        Text("Training Log")
                        Spacer()
                        Button("Clear") { manager.clearLog() }
                            .font(.caption)
                    }
                }
            }
        }
        .navigationTitle("MLX Transformer")
        .onAppear {
            monitor.startMonitoring()
        }
        .onDisappear {
            monitor.stopMonitoring()
        }
    }
}

#Preview {
    NavigationStack { MLXTrainingView() }
}

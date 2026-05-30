import CenturionMLX
import SwiftUI

struct PipelineTrainingView: View {
    @Bindable var manager: PipelineTrainingManager
    @State private var monitor = HardwareMonitor()
    @State private var showPerformanceStats = true

    var body: some View {
        List {
            if manager.pipelineConfigured {
                Section("Pipeline Stage") {
                    LabeledContent("Role") {
                        Text(manager.stageInfo.isHead ? "HEAD" : manager.stageInfo.isTail ? "TAIL" : "MIDDLE")
                            .fontWeight(.semibold)
                            .foregroundStyle(manager.stageInfo.isHead ? .blue : .green)
                    }
                    LabeledContent("Stage") {
                        Text("\(manager.stageInfo.stageIndex + 1) of \(manager.stageInfo.totalStages)")
                    }
                    LabeledContent("Layers") {
                        Text("[\(manager.stageInfo.firstLayer)..\(manager.stageInfo.lastLayer))")
                            .monospacedDigit()
                    }
                    LabeledContent("Micro-batches") {
                        Text("\(manager.stageInfo.numMicroBatches)")
                    }
                }

                Section("Model Config") {
                    LabeledContent("d_model", value: "\(manager.config.dModel)")
                    LabeledContent("Heads", value: "\(manager.config.nHeads)")
                    LabeledContent("Total Layers", value: "\(manager.config.nLayers)")
                    LabeledContent("Seq Length", value: "\(manager.config.seqLen)")
                    LabeledContent("Batch Size", value: "\(manager.config.batchSize)")
                    LabeledContent("Vocab Size", value: "\(manager.config.vocabSize)")
                    LabeledContent("Learning Rate", value: String(format: "%.1e", manager.config.learningRate))
                }
            }

            if manager.isTraining || manager.progress > 0 {
                Section("Training") {
                    if manager.isTraining {
                        Button("Stop") {
                            manager.stopTraining()
                        }
                        .buttonStyle(.bordered)
                        .tint(.red)
                    }

                    ProgressView(value: manager.progress) {
                        Text(manager.status)
                            .font(.caption)
                    }

                    LabeledContent("Step", value: "\(manager.currentStep)/\(manager.totalSteps)")
                }
            }

            if manager.currentLoss > 0 || manager.avgForwardMs > 0 {
                Section("Pipeline Metrics") {
                    if manager.currentLoss > 0 {
                        LabeledContent("Loss") {
                            Text(String(format: "%.4f", manager.currentLoss))
                                .monospacedDigit()
                        }
                    }
                    LabeledContent("Avg Forward") {
                        Text(String(format: "%.1f ms", manager.avgForwardMs))
                            .monospacedDigit()
                    }
                    LabeledContent("Avg Backward") {
                        Text(String(format: "%.1f ms", manager.avgBackwardMs))
                            .monospacedDigit()
                    }
                    LabeledContent("Avg Send") {
                        Text(String(format: "%.1f ms", manager.avgSendMs))
                            .monospacedDigit()
                    }
                    LabeledContent("Avg Recv") {
                        Text(String(format: "%.1f ms", manager.avgRecvMs))
                            .monospacedDigit()
                    }
                    LabeledContent("Pipeline Efficiency") {
                        Text(String(format: "%.0f%%", manager.pipelineEfficiency))
                            .monospacedDigit()
                            .foregroundStyle(manager.pipelineEfficiency > 70 ? .green : .orange)
                    }
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
        .navigationTitle("Pipeline Training")
        .onAppear { monitor.startMonitoring() }
        .onDisappear { monitor.stopMonitoring() }
    }
}

#Preview {
    NavigationStack { PipelineTrainingView(manager: PipelineTrainingManager()) }
}

import SwiftUI
import CoreML

struct MLPlaygroundView: View {
    @State private var manager = UpdatableModelManager()
    @State private var modelName: String = "MNISTClassifier"
    @State private var epochs: Int = 5
    @State private var sampleCount: Int = 100
    @State private var selectedComputeUnits: MLComputeUnits = .all

    private let computeUnitOptions: [(label: String, value: MLComputeUnits)] = [
        ("All (CPU+GPU+ANE)", .all),
        ("CPU + ANE", .cpuAndNeuralEngine),
        ("CPU + GPU", .cpuAndGPU),
        ("CPU Only", .cpuOnly),
    ]

    var body: some View {
        List {
            Section("Model") {
                TextField("Model name (without extension)", text: $modelName)
                    .textInputAutocapitalization(.never)
                    .autocorrectionDisabled()
                HStack {
                    Button("Load") { manager.loadModel(named: modelName) }
                        .buttonStyle(.borderedProminent)
                    Button("Unload") { manager.unload() }
                        .buttonStyle(.bordered)
                        .disabled(manager.loadedModelName == nil)
                }
                if let name = manager.loadedModelName, !name.isEmpty {
                    LabeledContent("Loaded:", value: name)
                }
                if !manager.modelSummary.isEmpty {
                    Text(manager.modelSummary)
                        .font(.footnote.monospaced())
                }
            }

            Section("Inference") {
                Button("Warm-up Prediction") {
                    Task { await manager.warmUpPrediction() }
                }
                .disabled(manager.loadedModelName == nil)
                if !manager.lastPredictionSummary.isEmpty {
                    Text(manager.lastPredictionSummary)
                        .font(.footnote.monospaced())
                }
            }

            Section("Training") {
                Picker("Compute Units", selection: $selectedComputeUnits) {
                    ForEach(computeUnitOptions, id: \.value) { option in
                        Text(option.label).tag(option.value)
                    }
                }
                .onChange(of: selectedComputeUnits) {
                    manager.setComputeUnits(selectedComputeUnits)
                }

                Stepper("Epochs: \(epochs)", value: $epochs, in: 1...100)
                Stepper("Samples: \(sampleCount)", value: $sampleCount, in: 10...1000, step: 10)

                Button(manager.isTraining ? "Training…" : "Train with Synthetic Data") {
                    Task { await manager.trainWithSynthetic(samples: sampleCount, epochs: epochs) }
                }
                .disabled(manager.isTraining || manager.loadedModelName == nil)

                if manager.isTraining {
                    ProgressView(value: manager.progress) {
                        Text(manager.status)
                            .font(.caption)
                    }
                }
            }

            Section("Status") {
                Text(manager.status)
                    .font(.callout)
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
        .navigationTitle("ML Playground")
    }
}

#Preview {
    NavigationStack { MLPlaygroundView() }
}

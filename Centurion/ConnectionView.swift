import SwiftUI

struct ConnectionView: View {
    @Bindable var pipelineManager: PipelineTrainingManager
    @Bindable var orchestratorManager: OrchestratorManager
    @Binding var selectedTab: Int

    @State private var host: String = "34.60.122.134"
    @State private var portText: String = "9998"
    @State private var identity: String = ""
    @State private var secret: String = ""
    @State private var connectingWorkerViaBridge: Bool = false
    @FocusState private var focusedField: Field?

    private enum Field { case host, port, identity, secret }

    /// Whether the identity field looks like an orchestrator login.
    private var isOrchIdentity: Bool {
        let id = identity.trimmingCharacters(in: .whitespacesAndNewlines).lowercased()
        return id == "orch" || id == "orchestrator"
    }

    /// Whether any connection is active.
    private var isConnected: Bool {
        orchestratorManager.isConnected || pipelineManager.isConnected
    }

    var body: some View {
        List {
            if isConnected {
                connectedView
            } else {
                loginView
            }
        }
        .navigationTitle("Connect")
        .toolbar {
            ToolbarItemGroup(placement: .keyboard) {
                Spacer()
                Button("Done") {
                    focusedField = nil
                    UIApplication.shared.sendAction(#selector(UIResponder.resignFirstResponder), to: nil, from: nil, for: nil)
                }
            }
        }
        .scrollDismissesKeyboard(.interactively)
    }

    // MARK: - Login (not connected)

    @ViewBuilder
    private var loginView: some View {
        Section("Server") {
            TextField("Host", text: $host)
                .textInputAutocapitalization(.never)
                .autocorrectionDisabled()
                .keyboardType(.decimalPad)
                .focused($focusedField, equals: .host)

            TextField("Port", text: $portText)
                .keyboardType(.numberPad)
                .focused($focusedField, equals: .port)
        }

        Section("Credentials") {
            TextField("Identity", text: $identity)
                .textContentType(.username)
                .textInputAutocapitalization(.never)
                .autocorrectionDisabled()
                .focused($focusedField, equals: .identity)

            SecureField("Secret", text: $secret)
                .textContentType(.password)
                .textInputAutocapitalization(.never)
                .autocorrectionDisabled()
                .focused($focusedField, equals: .secret)
        }

        Section {
            if orchestratorManager.authFailed || pipelineManager.authFailed {
                Label("Authentication failed — check secret", systemImage: "xmark.shield.fill")
                    .foregroundStyle(.red)
                    .font(.caption)
            }

            Button("Connect") {
                focusedField = nil
                if isOrchIdentity {
                    connectOrchestrator()
                } else {
                    connectWorker()
                }
            }
            .buttonStyle(.borderedProminent)
            .disabled(secret.isEmpty || identity.isEmpty)
        }
    }

    // MARK: - Connected

    @ViewBuilder
    private var connectedView: some View {
        // Status
        Section("Connection") {
            HStack {
                Image(systemName: "checkmark.circle.fill")
                    .foregroundStyle(.green)
                if orchestratorManager.isConnected {
                    Text("Orchestrator")
                } else {
                    Text("Worker")
                }
                Spacer()
                Text("\(host):\(portText)")
                    .foregroundStyle(.secondary)
                    .font(.caption)
            }

            if pipelineManager.isConnected {
                HStack {
                    Image(systemName: "checkmark.circle.fill")
                        .foregroundStyle(.green)
                    Text("Worker")
                }
            }
        }

        // Orchestrator: offer to also connect as worker via server-side bypass
        if orchestratorManager.isConnected {
            if pipelineManager.isConnected {
                Section {
                    Button("Disconnect Worker") {
                        pipelineManager.disconnect()
                    }
                    .foregroundStyle(.orange)
                }
            } else {
                Section {
                    Button("Also Connect as Worker") {
                        connectWorkerViaBridge()
                    }
                    .buttonStyle(.borderedProminent)
                    .tint(.green)
                    .disabled(connectingWorkerViaBridge)

                    if connectingWorkerViaBridge {
                        HStack(spacing: 8) {
                            ProgressView()
                            Text("Requesting worker bypass…")
                                .font(.caption)
                                .foregroundStyle(.secondary)
                        }
                    }

                    if pipelineManager.authFailed {
                        Label("Worker connection failed", systemImage: "xmark.shield.fill")
                            .foregroundStyle(.red)
                            .font(.caption)
                    }
                }
            }
        }

        // Disconnect
        Section {
            Button("Disconnect") {
                orchestratorManager.disconnect()
                pipelineManager.disconnect()
                connectingWorkerViaBridge = false
                secret = ""
            }
            .foregroundStyle(.red)
        }
    }

    // MARK: - Actions

    private func connectWorker() {
        guard let port = UInt16(portText) else { return }
        pipelineManager.serverHost = host
        pipelineManager.serverPort = port
        pipelineManager.serverSecret = secret
        pipelineManager.connect()
        selectedTab = 1
    }

    private func connectOrchestrator() {
        guard let port = UInt16(portText) else { return }
        orchestratorManager.serverHost = host
        orchestratorManager.serverPort = port
        orchestratorManager.serverSecret = secret
        orchestratorManager.connect()
        selectedTab = 2
    }

    private func connectWorkerViaBridge() {
        guard let port = UInt16(portText) else { return }
        connectingWorkerViaBridge = true
        pipelineManager.authFailed = false

        // Ask the server to whitelist our IP for one-time worker auth bypass
        orchestratorManager.requestWorkerBypass()

        // Watch for the bypass acknowledgment, then connect as worker
        Task { @MainActor in
            // Poll for bypass readiness (up to 5 seconds)
            for _ in 0..<50 {
                if orchestratorManager.workerBypassReady {
                    break
                }
                try? await Task.sleep(for: .milliseconds(100))
            }

            guard orchestratorManager.workerBypassReady else {
                connectingWorkerViaBridge = false
                pipelineManager.authFailed = true
                return
            }

            // Server has whitelisted our IP — connect with a dummy secret
            // (the server will bypass HMAC verification for this connection)
            pipelineManager.serverHost = host
            pipelineManager.serverPort = port
            pipelineManager.serverSecret = "bypass"
            pipelineManager.connect()
            connectingWorkerViaBridge = false
            orchestratorManager.workerBypassReady = false
        }
    }
}

#Preview {
    @Previewable @State var tab = 0
    NavigationStack {
        ConnectionView(
            pipelineManager: PipelineTrainingManager(),
            orchestratorManager: OrchestratorManager(),
            selectedTab: $tab
        )
    }
}

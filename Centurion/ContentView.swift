import SwiftUI

struct ContentView: View {
    @State private var pipelineManager = PipelineTrainingManager()
    @State private var orchestratorManager = OrchestratorManager()
    @State private var selectedTab: Int = 0

    var body: some View {
        TabView(selection: $selectedTab) {
            NavigationStack {
                ConnectionView(
                    pipelineManager: pipelineManager,
                    orchestratorManager: orchestratorManager,
                    selectedTab: $selectedTab
                )
            }
            .tabItem {
                Label("Connect", systemImage: "link")
            }
            .tag(0)

            NavigationStack {
                PipelineTrainingView(manager: pipelineManager)
            }
            .tabItem {
                Label("Pipeline", systemImage: "arrow.triangle.swap")
            }
            .tag(1)

            NavigationStack {
                OrchestratorView(manager: orchestratorManager)
            }
            .tabItem {
                Label("Orchestrator", systemImage: "server.rack")
            }
            .tag(2)

            NavigationStack {
                MLXTrainingView()
            }
            .tabItem {
                Label("Transformer", systemImage: "brain")
            }
            .tag(3)
        }
    }
}

#Preview {
    ContentView()
}

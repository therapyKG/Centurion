import SwiftUI

struct ContentView: View {
    var body: some View {
        TabView {
            NavigationStack {
                MLXTrainingView()
            }
            .tabItem {
                Label("Transformer", systemImage: "brain")
            }

            NavigationStack {
                MLPlaygroundView()
            }
            .tabItem {
                Label("Core ML", systemImage: "cpu")
            }
        }
    }
}

#Preview {
    ContentView()
}

import SwiftUI
import Metal
import os

// MARK: - Hardware Monitor

/// Polls hardware metrics (thermal state, memory, CPU usage, Metal memory) on a timer.
@MainActor
@Observable
final class HardwareMonitor {
    var thermalState: ProcessInfo.ThermalState = .nominal
    var appMemoryMB: Double = 0
    var totalRAMGB: Double = 0
    var metalMemoryMB: Double = 0
    var peakMetalMemoryMB: Double = 0
    var cpuUsagePercent: Double = 0
    var isMonitoring: Bool = false

    private var timer: Timer?
    private var metalDevice: MTLDevice?

    func startMonitoring(device: MTLDevice? = MTLCreateSystemDefaultDevice()) {
        guard !isMonitoring else { return }
        self.metalDevice = device
        isMonitoring = true
        totalRAMGB = Double(ProcessInfo.processInfo.physicalMemory) / (1024 * 1024 * 1024)
        update()
        timer = Timer.scheduledTimer(withTimeInterval: 1.5, repeats: true) { [weak self] _ in
            guard let self else { return }
            Task { @MainActor in
                self.update()
            }
        }
    }

    func stopMonitoring() {
        timer?.invalidate()
        timer = nil
        isMonitoring = false
    }

    func resetPeakMemory() {
        peakMetalMemoryMB = metalMemoryMB
    }

    private func update() {
        thermalState = ProcessInfo.processInfo.thermalState

        // App available memory
        appMemoryMB = Double(os_proc_available_memory()) / (1024 * 1024)

        // Metal allocated memory
        if let metalDevice {
            metalMemoryMB = Double(metalDevice.currentAllocatedSize) / (1024 * 1024)
            peakMetalMemoryMB = max(peakMetalMemoryMB, metalMemoryMB)
        }

        // CPU usage via Mach task_info
        cpuUsagePercent = Self.getProcessCPUUsage()
    }

    /// Get process CPU usage percentage using Mach APIs.
    nonisolated private static func getProcessCPUUsage() -> Double {
        var threadsListPtr: thread_act_array_t?
        var threadsCount = mach_msg_type_number_t(0)
        let result = task_threads(mach_task_self_, &threadsListPtr, &threadsCount)
        guard result == KERN_SUCCESS, let threadsList = threadsListPtr else { return 0 }

        var totalUsage: Double = 0
        for i in 0..<Int(threadsCount) {
            var info = thread_basic_info()
            var infoCount = mach_msg_type_number_t(MemoryLayout<thread_basic_info>.size / MemoryLayout<integer_t>.size)
            let kr = withUnsafeMutablePointer(to: &info) { infoPtr in
                infoPtr.withMemoryRebound(to: integer_t.self, capacity: Int(infoCount)) { rawPtr in
                    thread_info(threadsList[i], thread_flavor_t(THREAD_BASIC_INFO), rawPtr, &infoCount)
                }
            }
            if kr == KERN_SUCCESS && (info.flags & TH_FLAGS_IDLE) == 0 {
                totalUsage += Double(info.cpu_usage) / Double(TH_USAGE_SCALE) * 100.0
            }
        }

        // Deallocate thread list
        let size = vm_size_t(MemoryLayout<thread_t>.stride * Int(threadsCount))
        vm_deallocate(mach_task_self_, vm_address_t(bitPattern: threadsList), size)

        return totalUsage
    }
}

// MARK: - Diagnostics View

struct HardwareDiagnosticsView: View {
    let monitor: HardwareMonitor

    var body: some View {
        LabeledContent("Thermal") {
            HStack(spacing: 4) {
                Circle()
                    .fill(thermalColor)
                    .frame(width: 8, height: 8)
                Text(thermalLabel)
                    .foregroundStyle(thermalColor)
            }
        }
        LabeledContent("Available RAM") {
            Text(String(format: "%.0f MB / %.1f GB", monitor.appMemoryMB, monitor.totalRAMGB))
                .monospacedDigit()
        }
        LabeledContent("Metal Memory") {
            VStack(alignment: .trailing, spacing: 2) {
                Text(String(format: "%.1f MB", monitor.metalMemoryMB))
                    .monospacedDigit()
                Text(String(format: "peak: %.1f MB", monitor.peakMetalMemoryMB))
                    .font(.caption2)
                    .foregroundStyle(.secondary)
                    .monospacedDigit()
            }
        }
        LabeledContent("CPU Usage") {
            Text(String(format: "%.0f%%", monitor.cpuUsagePercent))
                .monospacedDigit()
        }
    }

    private var thermalColor: Color {
        switch monitor.thermalState {
        case .nominal: return .green
        case .fair: return .yellow
        case .serious: return .orange
        case .critical: return .red
        @unknown default: return .gray
        }
    }

    private var thermalLabel: String {
        switch monitor.thermalState {
        case .nominal: return "Nominal"
        case .fair: return "Fair"
        case .serious: return "Serious"
        case .critical: return "Critical"
        @unknown default: return "Unknown"
        }
    }
}

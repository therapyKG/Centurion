#!/bin/bash
set -e

PROJECT="/Users/kurtgu/Centurion/Centurion.xcodeproj"
SCHEME="Centurion"
DERIVED_DATA="/Users/kurtgu/Centurion/.build_dd"
CONFIG="Debug"

DEVICES=(
    "00008140-0002083222FB001C"  # Therapy16Pro (iPhone)
    "00008142-000845E43AF3801C"  # iPad (2)
    "00008103-001415491EB9001E"  # iPadPro#0011
)

echo "=== Building $SCHEME ==="
xcodebuild -project "$PROJECT" \
    -scheme "$SCHEME" \
    -configuration "$CONFIG" \
    -derivedDataPath "$DERIVED_DATA" \
    -destination "id=${DEVICES[0]}" \
    build 2>&1 | tail -5

APP_PATH="$DERIVED_DATA/Build/Products/$CONFIG-iphoneos/Centurion.app"

if [ ! -d "$APP_PATH" ]; then
    echo "ERROR: App not found at $APP_PATH"
    exit 1
fi

echo "=== Installing to ${#DEVICES[@]} devices in parallel ==="
pids=()
for dev in "${DEVICES[@]}"; do
    echo "  -> Installing to $dev ..."
    xcrun devicectl device install app --device "$dev" "$APP_PATH" 2>&1 &
    pids+=($!)
done

# Wait for all installs
failed=0
for pid in "${pids[@]}"; do
    if ! wait "$pid"; then
        failed=$((failed + 1))
    fi
done

if [ "$failed" -eq 0 ]; then
    echo "=== All ${#DEVICES[@]} devices installed successfully ==="
else
    echo "=== WARNING: $failed device(s) failed to install ==="
    exit 1
fi

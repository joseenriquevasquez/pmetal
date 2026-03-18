#!/usr/bin/env bash
# Build pmetal-gui with updater signing enabled.
# Retrieves the Tauri signing key from macOS Keychain.
#
# Prerequisites:
#   1. Generate a keypair:
#      cd crates/pmetal-gui && npx @tauri-apps/cli signer generate -w ~/.tauri/pmetal.key
#
#   2. Store in Keychain:
#      security add-generic-password -a pmetal -s tauri-updater-signing-key \
#        -w "$(cat ~/.tauri/pmetal.key)" login.keychain-db
#
#   3. If you set a password on the key:
#      security add-generic-password -a pmetal -s tauri-updater-signing-password \
#        -w "YOUR_PASSWORD" login.keychain-db
#
# Usage:
#   ./crates/pmetal-gui/scripts/build-with-updater.sh [extra tauri build args...]

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
GUI_DIR="$(dirname "$SCRIPT_DIR")"

# Retrieve signing key from Keychain
echo "Retrieving Tauri signing key from Keychain..."
TAURI_SIGNING_PRIVATE_KEY=$(security find-generic-password -a pmetal -s tauri-updater-signing-key -w 2>/dev/null) || {
  echo "ERROR: Signing key not found in Keychain."
  echo "Store it with:"
  echo '  security add-generic-password -a pmetal -s tauri-updater-signing-key -w "$(cat ~/.tauri/pmetal.key)" login.keychain-db'
  exit 1
}
export TAURI_SIGNING_PRIVATE_KEY

# Retrieve optional password
TAURI_SIGNING_PRIVATE_KEY_PASSWORD=$(security find-generic-password -a pmetal -s tauri-updater-signing-password -w 2>/dev/null) || TAURI_SIGNING_PRIVATE_KEY_PASSWORD=""
export TAURI_SIGNING_PRIVATE_KEY_PASSWORD

echo "Signing key loaded. Building..."
cd "$GUI_DIR"
bun run tauri build "$@"

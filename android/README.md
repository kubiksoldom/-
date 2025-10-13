# CryptoBot Remote (Android)

This module contains the Kotlin/Jetpack Android companion application for TradeApp. The app
provides three primary features:

1. **Download** – scan QR codes or paste download tokens/URLs exposed by the desktop APK Manager
   and hand them off to Android's `DownloadManager` followed by an `ACTION_VIEW` install intent.
2. **Remote controls** – invoke the TradeApp HTTP endpoints (`/control/panic_close`,
   `/control/entries`, `/control/pairs`, `/download/<token>`) and stream the responses to an in-app
   log.
3. **Settings** – persist host/IP, port, optional PIN header, and trusted IP whitelist via Jetpack
   DataStore, with a one-click connectivity check.

The application targets Android 8.1 (API 26) and newer and produces a single release artefact that
supports both `arm64-v8a` and `armeabi-v7a` ABIs.

## Project layout

```
android/
├── build.gradle.kts         # Root Gradle build
├── settings.gradle.kts      # Project definition
└── app/
    ├── build.gradle.kts     # Android application module
    ├── src/main/
    │   ├── AndroidManifest.xml
    │   ├── java/com/tradeapp/cryptobotremote
    │   │   ├── MainActivity.kt
    │   │   ├── GatewayViewModel.kt
    │   │   ├── data/SettingsRepository.kt
    │   │   ├── network/TradeGatewayClient.kt
    │   │   ├── ui/... (Download, Remote, Settings fragments)
    │   │   └── util/LogAdapter.kt
    │   └── res/...          # Material components layouts, navigation graph & icons
    └── proguard-rules.pro
```

The `GatewayViewModel` coordinates persistent settings, HTTP calls (via OkHttp), and the per-tab log
feeds consumed by each fragment. QR scanning is handled by ZXing Embedded's `ScanContract` with ML
Kit available as a fallback dependency.

## Building locally

1. Install Android Studio Jellyfish (or newer) with the Android SDK platform tools and the
   `Android Gradle Plugin` v8.5+.
2. Open the `android/` directory in Android Studio and let it download the Gradle wrapper
   dependencies (Gradle 8.7, Kotlin 1.9.24).
3. Create or import a keystore for release signing. For CI smoke tests and local installs you can
   rely on the default debug keystore – the Gradle configuration signs the `release` variant with it
   so the generated APK is immediately installable:

   ```bash
   ./gradlew assembleRelease
   ```

   The resulting APK will be placed under
   `android/app/build/outputs/apk/release/cryptobot_v3.apk` and a copy is written to
   `build/output/cryptobot_v3.apk` for downstream packaging.
4. Copy or rename the artefact to `build/output/cryptobot_v3.apk` if you need to redistribute it
   outside of CI.

To produce a Play-safe build, configure `signingConfigs` in `android/app/build.gradle.kts` with your
release keystore and pass the passwords through environment variables or a `gradle.properties`
entry.

## CI considerations

The repository does not ship pre-baked Android SDK images. Ensure the CI job installs the Android
SDK components and runs `./gradlew assembleRelease` before publishing the generated
`cryptobot_v3.apk` as a job artefact. The workflow also caches Gradle and Android SDK directories to
speed up subsequent builds.

Desktop TradeApp code remains untouched by the Android client – all changes are confined to the new
`android/app` Gradle project.

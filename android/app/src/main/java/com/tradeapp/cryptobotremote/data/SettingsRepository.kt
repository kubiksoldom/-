package com.tradeapp.cryptobotremote.data

import android.content.Context
import androidx.datastore.core.DataStore
import androidx.datastore.preferences.core.Preferences
import androidx.datastore.preferences.core.edit
import androidx.datastore.preferences.core.intPreferencesKey
import androidx.datastore.preferences.core.stringPreferencesKey
import androidx.datastore.preferences.preferencesDataStore
import kotlinx.coroutines.flow.Flow
import kotlinx.coroutines.flow.map

private val Context.settingsDataStore: DataStore<Preferences> by preferencesDataStore(
    name = "cryptobot_settings"
)

class SettingsRepository(private val context: Context) {

    private val hostKey = stringPreferencesKey("host")
    private val portKey = intPreferencesKey("port")
    private val pinKey = stringPreferencesKey("pin")
    private val trustedIpKey = stringPreferencesKey("trusted_ips")

    val settings: Flow<GatewaySettings> = context.settingsDataStore.data.map { prefs ->
        GatewaySettings(
            host = prefs[hostKey] ?: "",
            port = prefs[portKey] ?: DEFAULT_PORT,
            pin = prefs[pinKey] ?: "",
            trustedIps = prefs[trustedIpKey] ?: ""
        )
    }

    suspend fun save(settings: GatewaySettings) {
        context.settingsDataStore.edit { prefs ->
            prefs[hostKey] = settings.host
            prefs[portKey] = settings.port
            prefs[pinKey] = settings.pin
            prefs[trustedIpKey] = settings.trustedIps
        }
    }

    companion object {
        const val DEFAULT_PORT = 8000
    }
}

data class GatewaySettings(
    val host: String,
    val port: Int,
    val pin: String,
    val trustedIps: String
) {
    val isComplete: Boolean
        get() = host.isNotBlank() && port > 0

    fun baseUrl(): String = "http://$host:$port"
}

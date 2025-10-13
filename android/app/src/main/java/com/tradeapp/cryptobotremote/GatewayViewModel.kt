package com.tradeapp.cryptobotremote

import android.app.Application
import androidx.lifecycle.AndroidViewModel
import androidx.lifecycle.viewModelScope
import com.tradeapp.cryptobotremote.data.GatewaySettings
import com.tradeapp.cryptobotremote.data.SettingsRepository
import com.tradeapp.cryptobotremote.network.TradeGatewayClient
import com.tradeapp.cryptobotremote.network.TradeResponse
import com.tradeapp.cryptobotremote.util.LogEntry
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.SharingStarted
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.flow.stateIn
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import kotlinx.coroutines.Dispatchers

class GatewayViewModel(application: Application) : AndroidViewModel(application) {

    private val repository = SettingsRepository(application)
    private val client = TradeGatewayClient()

    val settingsState: StateFlow<GatewaySettings> = repository.settings.stateIn(
        scope = viewModelScope,
        started = SharingStarted.WhileSubscribed(5000),
        initialValue = GatewaySettings("", SettingsRepository.DEFAULT_PORT, "", "")
    )

    private val _downloadLogs = MutableStateFlow<List<LogEntry>>(emptyList())
    val downloadLogs: StateFlow<List<LogEntry>> = _downloadLogs.asStateFlow()

    private val _remoteLogs = MutableStateFlow<List<LogEntry>>(emptyList())
    val remoteLogs: StateFlow<List<LogEntry>> = _remoteLogs.asStateFlow()

    private val _settingsLogs = MutableStateFlow<List<LogEntry>>(emptyList())
    val settingsLogs: StateFlow<List<LogEntry>> = _settingsLogs.asStateFlow()

    fun saveSettings(settings: GatewaySettings) {
        viewModelScope.launch {
            repository.save(settings)
            appendLog(_settingsLogs, "Saved settings for ${'$'}{settings.host}:${'$'}{settings.port}")
        }
    }

    suspend fun pingGateway(): TradeResponse {
        val settings = requireSettings()
        appendLog(_settingsLogs, "Pinging ${'$'}{settings.baseUrl()}")
        return runCatching { client.ping(settings) }
            .onFailure { appendLog(_settingsLogs, "Ping failed: ${'$'}{it.message}") }
            .getOrElse { throw it }
            .also { appendLog(_settingsLogs, "Ping: ${'$'}it") }
    }

    suspend fun panicClose(): TradeResponse = executeRemoteAction(_remoteLogs) {
        client.panicClose(requireSettings())
    }

    suspend fun pauseEntries(): TradeResponse = executeRemoteAction(_remoteLogs) {
        client.pauseEntries(requireSettings())
    }

    suspend fun resumeEntries(): TradeResponse = executeRemoteAction(_remoteLogs) {
        client.resumeEntries(requireSettings())
    }

    suspend fun applyPairs(pairs: String): TradeResponse = executeRemoteAction(_remoteLogs) {
        val sanitized = pairs.split(',').joinToString(",") { it.trim().uppercase() }.trim(',')
        require(sanitized.isNotBlank()) { "Pairs cannot be empty" }
        client.setPairs(requireSettings(), sanitized)
    }

    fun clearDownloadLogs() { _downloadLogs.value = emptyList() }
    fun clearRemoteLogs() { _remoteLogs.value = emptyList() }
    fun clearSettingsLogs() { _settingsLogs.value = emptyList() }

    suspend fun buildDownloadUrl(rawInput: String): String {
        val settings = requireSettings()
        return client.buildDownloadUrl(rawInput, settings)
    }

    suspend fun logDownloadAttempt(url: String) {
        withContext(Dispatchers.Main) {
            appendLog(_downloadLogs, "Downloading ${'$'}url")
        }
    }

    fun logDownloadResult(message: String) {
        appendLog(_downloadLogs, message)
    }

    private suspend fun executeRemoteAction(
        sink: MutableStateFlow<List<LogEntry>>,
        block: suspend () -> TradeResponse
    ): TradeResponse {
        val settings = requireSettings()
        appendLog(sink, "Calling ${'$'}{settings.baseUrl()}")
        return runCatching { block() }
            .onFailure { appendLog(sink, "Failed: ${'$'}{it.message}") }
            .getOrElse { throw it }
            .also { appendLog(sink, it.toString()) }
    }

    private fun appendLog(target: MutableStateFlow<List<LogEntry>>, message: String) {
        val entry = LogEntry(System.currentTimeMillis(), message)
        target.value = (target.value + entry).takeLast(MAX_LOG_ITEMS)
    }

    private suspend fun requireSettings(): GatewaySettings {
        val settings = settingsState.value
        if (!settings.isComplete) {
            throw IllegalStateException("Host and port must be configured")
        }
        return settings
    }

    companion object {
        private const val MAX_LOG_ITEMS = 100
    }
}

package com.tradeapp.cryptobotremote.network

import com.tradeapp.cryptobotremote.data.GatewaySettings
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext
import okhttp3.MediaType.Companion.toMediaType
import okhttp3.OkHttpClient
import okhttp3.Request
import okhttp3.RequestBody
import okhttp3.RequestBody.Companion.toRequestBody
import okhttp3.logging.HttpLoggingInterceptor
import okhttp3.HttpUrl.Companion.toHttpUrlOrNull
import java.io.IOException
import java.util.concurrent.TimeUnit

class TradeGatewayClient {

    private val client: OkHttpClient = OkHttpClient.Builder()
        .callTimeout(10, TimeUnit.SECONDS)
        .connectTimeout(5, TimeUnit.SECONDS)
        .readTimeout(10, TimeUnit.SECONDS)
        .writeTimeout(10, TimeUnit.SECONDS)
        .addInterceptor(HttpLoggingInterceptor().apply {
            level = HttpLoggingInterceptor.Level.BASIC
        })
        .build()

    suspend fun ping(settings: GatewaySettings): TradeResponse =
        execute(settings, settings.baseUrl())

    suspend fun panicClose(settings: GatewaySettings): TradeResponse =
        execute(settings, settings.baseUrl() + "/control/panic_close", method = "POST")

    suspend fun pauseEntries(settings: GatewaySettings): TradeResponse =
        execute(settings, settings.baseUrl() + "/control/entries", method = "POST", query = mapOf("state" to "pause"))

    suspend fun resumeEntries(settings: GatewaySettings): TradeResponse =
        execute(settings, settings.baseUrl() + "/control/entries", method = "POST", query = mapOf("state" to "resume"))

    suspend fun setPairs(settings: GatewaySettings, pairs: String): TradeResponse =
        execute(settings, settings.baseUrl() + "/control/pairs", method = "POST", query = mapOf("set" to pairs))

    suspend fun download(settings: GatewaySettings, url: String): TradeResponse =
        execute(settings, url)

    private suspend fun execute(
        settings: GatewaySettings,
        url: String,
        method: String = "GET",
        query: Map<String, String> = emptyMap(),
        body: RequestBody? = null
    ): TradeResponse = withContext(Dispatchers.IO) {
        val httpUrlBuilder = url.toHttpUrlOrNull()?.newBuilder()
            ?: throw IOException("Invalid URL: $url")
        query.forEach { (key, value) ->
            httpUrlBuilder.addQueryParameter(key, value)
        }
        val httpUrl = httpUrlBuilder.build()

        val requestBuilder = Request.Builder()
            .url(httpUrl)
            .method(method, if (method == "GET" && body == null) null else body ?: EmptyBody)

        if (settings.pin.isNotBlank()) {
            requestBuilder.addHeader("X-Auth-PIN", settings.pin)
        }
        if (settings.trustedIps.isNotBlank()) {
            requestBuilder.addHeader("X-Trusted-IPs", settings.trustedIps)
        }

        val request = requestBuilder.build()
        client.newCall(request).execute().use { response ->
            TradeResponse(
                statusCode = response.code,
                message = response.body?.string().orEmpty(),
                success = response.isSuccessful
            )
        }
    }

    fun buildDownloadUrl(rawInput: String, settings: GatewaySettings): String {
        val trimmed = rawInput.trim()
        if (trimmed.isEmpty()) {
            throw IllegalArgumentException("Input is empty")
        }
        return if (trimmed.startsWith("http://") || trimmed.startsWith("https://")) {
            trimmed
        } else {
            settings.baseUrl().trimEnd('/') + "/download/" + trimmed.removePrefix("/download/").removePrefix("download/")
        }
    }

    companion object {
        private val EmptyBody: RequestBody = ByteArray(0).toRequestBody("application/octet-stream".toMediaType())
    }
}

data class TradeResponse(
    val statusCode: Int,
    val message: String,
    val success: Boolean
) {
    override fun toString(): String = "[$statusCode] ${'$'}message"
}

package com.tradeapp.cryptobotremote.ui.settings

import android.os.Bundle
import android.view.LayoutInflater
import android.view.View
import android.view.ViewGroup
import androidx.fragment.app.Fragment
import androidx.fragment.app.activityViewModels
import androidx.lifecycle.lifecycleScope
import com.google.android.material.snackbar.Snackbar
import com.tradeapp.cryptobotremote.GatewayViewModel
import com.tradeapp.cryptobotremote.data.GatewaySettings
import com.tradeapp.cryptobotremote.databinding.FragmentSettingsBinding
import com.tradeapp.cryptobotremote.util.LogAdapter
import kotlinx.coroutines.launch

class SettingsFragment : Fragment() {

    private var _binding: FragmentSettingsBinding? = null
    private val binding get() = _binding!!
    private val viewModel: GatewayViewModel by activityViewModels()
    private val logAdapter = LogAdapter()
    private var isSettingFields = false

    override fun onCreateView(
        inflater: LayoutInflater,
        container: ViewGroup?,
        savedInstanceState: Bundle?
    ): View {
        _binding = FragmentSettingsBinding.inflate(inflater, container, false)
        return binding.root
    }

    override fun onViewCreated(view: View, savedInstanceState: Bundle?) {
        super.onViewCreated(view, savedInstanceState)

        binding.settingsLogRecyclerView.apply {
            layoutManager = androidx.recyclerview.widget.LinearLayoutManager(requireContext())
            adapter = logAdapter
        }

        viewLifecycleOwner.lifecycleScope.launch {
            viewModel.settingsLogs.collect { logs ->
                logAdapter.submitList(logs)
                if (logs.isNotEmpty()) {
                    binding.settingsLogRecyclerView.scrollToPosition(logs.lastIndex)
                }
            }
        }

        viewLifecycleOwner.lifecycleScope.launch {
            viewModel.settingsState.collect { settings ->
                if (!isSettingFields) {
                    isSettingFields = true
                    binding.hostInput.setText(settings.host)
                    binding.portInput.setText(settings.port.toString())
                    binding.pinInput.setText(settings.pin)
                    binding.trustedIpInput.setText(settings.trustedIps)
                    isSettingFields = false
                }
            }
        }

        binding.saveButton.setOnClickListener { saveSettings() }
        binding.pingButton.setOnClickListener { pingGateway() }
        binding.clearSettingsLogsButton.setOnClickListener { viewModel.clearSettingsLogs() }
    }

    override fun onDestroyView() {
        super.onDestroyView()
        _binding = null
    }

    private fun saveSettings() {
        val host = binding.hostInput.text?.toString().orEmpty().trim()
        val port = binding.portInput.text?.toString()?.toIntOrNull() ?: 0
        val pin = binding.pinInput.text?.toString().orEmpty().trim()
        val trustedIps = binding.trustedIpInput.text?.toString().orEmpty().trim()

        if (host.isBlank() || port <= 0) {
            Snackbar.make(binding.root, com.tradeapp.cryptobotremote.R.string.message_no_host, Snackbar.LENGTH_SHORT).show()
            return
        }

        viewModel.saveSettings(GatewaySettings(host, port, pin, trustedIps))
        Snackbar.make(binding.root, com.tradeapp.cryptobotremote.R.string.message_settings_saved, Snackbar.LENGTH_SHORT).show()
    }

    private fun pingGateway() {
        viewLifecycleOwner.lifecycleScope.launch {
            try {
                val response = viewModel.pingGateway()
                val message = if (response.success) {
                    getString(com.tradeapp.cryptobotremote.R.string.message_ping_success)
                } else {
                    getString(com.tradeapp.cryptobotremote.R.string.message_ping_failed)
                }
                Snackbar.make(binding.root, message, Snackbar.LENGTH_SHORT).show()
            } catch (ex: Exception) {
                Snackbar.make(binding.root, ex.message ?: getString(com.tradeapp.cryptobotremote.R.string.message_ping_failed), Snackbar.LENGTH_LONG).show()
            }
        }
    }
}

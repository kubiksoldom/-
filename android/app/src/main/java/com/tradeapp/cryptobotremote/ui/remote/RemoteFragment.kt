package com.tradeapp.cryptobotremote.ui.remote

import android.os.Bundle
import android.view.LayoutInflater
import android.view.View
import android.view.ViewGroup
import androidx.fragment.app.Fragment
import androidx.fragment.app.activityViewModels
import androidx.lifecycle.lifecycleScope
import com.google.android.material.snackbar.Snackbar
import com.tradeapp.cryptobotremote.GatewayViewModel
import com.tradeapp.cryptobotremote.databinding.FragmentRemoteBinding
import com.tradeapp.cryptobotremote.network.TradeResponse
import com.tradeapp.cryptobotremote.util.LogAdapter
import kotlinx.coroutines.launch

class RemoteFragment : Fragment() {

    private var _binding: FragmentRemoteBinding? = null
    private val binding get() = _binding!!
    private val viewModel: GatewayViewModel by activityViewModels()
    private val logAdapter = LogAdapter()

    override fun onCreateView(
        inflater: LayoutInflater,
        container: ViewGroup?,
        savedInstanceState: Bundle?
    ): View {
        _binding = FragmentRemoteBinding.inflate(inflater, container, false)
        return binding.root
    }

    override fun onViewCreated(view: View, savedInstanceState: Bundle?) {
        super.onViewCreated(view, savedInstanceState)

        binding.remoteLogRecyclerView.apply {
            layoutManager = androidx.recyclerview.widget.LinearLayoutManager(requireContext())
            adapter = logAdapter
        }

        viewLifecycleOwner.lifecycleScope.launch {
            viewModel.remoteLogs.collect { logs ->
                logAdapter.submitList(logs)
                if (logs.isNotEmpty()) {
                    binding.remoteLogRecyclerView.scrollToPosition(logs.lastIndex)
                }
            }
        }

        binding.panicCloseButton.setOnClickListener { triggerAction { viewModel.panicClose() } }
        binding.pauseEntriesButton.setOnClickListener { triggerAction { viewModel.pauseEntries() } }
        binding.resumeEntriesButton.setOnClickListener { triggerAction { viewModel.resumeEntries() } }
        binding.applyPairsButton.setOnClickListener { applyPairs() }
        binding.clearRemoteLogsButton.setOnClickListener { viewModel.clearRemoteLogs() }
    }

    override fun onDestroyView() {
        super.onDestroyView()
        _binding = null
    }

    private fun triggerAction(action: suspend () -> TradeResponse) {
        viewLifecycleOwner.lifecycleScope.launch {
            try {
                val response = action()
                Snackbar.make(binding.root, response.toString(), Snackbar.LENGTH_SHORT).show()
            } catch (ex: Exception) {
                Snackbar.make(binding.root, ex.message ?: "Action failed", Snackbar.LENGTH_LONG).show()
            }
        }
    }

    private fun applyPairs() {
        val pairs = binding.pairsInput.text?.toString().orEmpty()
        if (pairs.isBlank()) {
            Snackbar.make(binding.root, com.tradeapp.cryptobotremote.R.string.message_invalid_pairs, Snackbar.LENGTH_SHORT).show()
            return
        }
        triggerAction { viewModel.applyPairs(pairs) }
    }
}

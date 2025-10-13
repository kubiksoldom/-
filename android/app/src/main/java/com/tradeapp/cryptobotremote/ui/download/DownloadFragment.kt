package com.tradeapp.cryptobotremote.ui.download

import android.Manifest
import android.app.DownloadManager
import android.content.BroadcastReceiver
import android.content.Context
import android.content.Intent
import android.content.IntentFilter
import android.content.pm.PackageManager
import android.net.Uri
import android.os.Bundle
import android.os.Environment
import android.view.LayoutInflater
import android.view.View
import android.view.ViewGroup
import androidx.activity.result.contract.ActivityResultContracts
import androidx.core.content.ContextCompat
import androidx.fragment.app.Fragment
import androidx.fragment.app.activityViewModels
import androidx.lifecycle.lifecycleScope
import com.google.android.material.snackbar.Snackbar
import com.journeyapps.barcodescanner.ScanContract
import com.journeyapps.barcodescanner.ScanOptions
import com.tradeapp.cryptobotremote.GatewayViewModel
import com.tradeapp.cryptobotremote.databinding.FragmentDownloadBinding
import com.tradeapp.cryptobotremote.util.LogAdapter
import kotlinx.coroutines.launch

class DownloadFragment : Fragment() {

    private var _binding: FragmentDownloadBinding? = null
    private val binding get() = _binding!!
    private val viewModel: GatewayViewModel by activityViewModels()
    private val logAdapter = LogAdapter()
    private var downloadId: Long? = null
    private var downloadManager: DownloadManager? = null

    private val cameraPermissionLauncher = registerForActivityResult(
        ActivityResultContracts.RequestPermission()
    ) { granted ->
        if (granted) {
            launchQrScannerInternal()
        } else {
            _binding?.root?.let {
                Snackbar.make(
                    it,
                    getString(com.tradeapp.cryptobotremote.R.string.message_camera_permission_required),
                    Snackbar.LENGTH_LONG
                ).show()
            }
        }
    }

    private val barcodeLauncher = registerForActivityResult(ScanContract()) { result ->
        if (result.contents != null) {
            binding.urlInput.setText(result.contents)
        } else {
            Snackbar.make(binding.root, getString(com.tradeapp.cryptobotremote.R.string.message_qr_not_supported), Snackbar.LENGTH_SHORT).show()
        }
    }

    private val downloadReceiver = object : BroadcastReceiver() {
        override fun onReceive(context: Context?, intent: Intent?) {
            val id = intent?.getLongExtra(DownloadManager.EXTRA_DOWNLOAD_ID, -1L)
            if (id != null && id == downloadId) {
                handleDownloadFinished()
            }
        }
    }

    override fun onCreateView(
        inflater: LayoutInflater,
        container: ViewGroup?,
        savedInstanceState: Bundle?
    ): View {
        _binding = FragmentDownloadBinding.inflate(inflater, container, false)
        downloadManager = requireContext().getSystemService(Context.DOWNLOAD_SERVICE) as DownloadManager
        return binding.root
    }

    override fun onViewCreated(view: View, savedInstanceState: Bundle?) {
        super.onViewCreated(view, savedInstanceState)

        binding.logRecyclerView.apply {
            layoutManager = androidx.recyclerview.widget.LinearLayoutManager(requireContext())
            adapter = logAdapter
        }

        viewLifecycleOwner.lifecycleScope.launch {
            viewModel.downloadLogs.collect { logs ->
                logAdapter.submitList(logs)
                if (logs.isNotEmpty()) {
                    binding.logRecyclerView.scrollToPosition(logs.lastIndex)
                }
            }
        }

        binding.scanQrButton.setOnClickListener { launchQrScanner() }
        binding.openLinkButton.setOnClickListener { openLink() }
        binding.downloadButton.setOnClickListener { enqueueDownload() }
        binding.clearLogsButton.setOnClickListener { viewModel.clearDownloadLogs() }

        val prefill = arguments?.getString("prefillUrl")
        if (!prefill.isNullOrBlank()) {
            binding.urlInput.setText(prefill)
            arguments?.remove("prefillUrl")
        }
    }

    override fun onStart() {
        super.onStart()
        ContextCompat.registerReceiver(
            requireContext(),
            downloadReceiver,
            IntentFilter(DownloadManager.ACTION_DOWNLOAD_COMPLETE),
            ContextCompat.RECEIVER_NOT_EXPORTED
        )
    }

    override fun onStop() {
        super.onStop()
        try {
            requireContext().unregisterReceiver(downloadReceiver)
        } catch (_: IllegalArgumentException) {
            // Receiver not registered (e.g. fragment stopped twice)
        }
    }

    override fun onDestroyView() {
        super.onDestroyView()
        _binding = null
    }

    private fun launchQrScanner() {
        when {
            ContextCompat.checkSelfPermission(requireContext(), Manifest.permission.CAMERA) == PackageManager.PERMISSION_GRANTED -> {
                launchQrScannerInternal()
            }
            shouldShowRequestPermissionRationale(Manifest.permission.CAMERA) -> {
                Snackbar.make(binding.root, getString(com.tradeapp.cryptobotremote.R.string.message_camera_permission_required), Snackbar.LENGTH_LONG).show()
            }
            else -> {
                cameraPermissionLauncher.launch(Manifest.permission.CAMERA)
            }
        }
    }

    private fun launchQrScannerInternal() {
        val options = ScanOptions().apply {
            setDesiredBarcodeFormats(ScanOptions.QR_CODE)
            setPrompt(getString(com.tradeapp.cryptobotremote.R.string.action_scan_qr))
            setBeepEnabled(false)
        }
        barcodeLauncher.launch(options)
    }

    private fun openLink() {
        val input = binding.urlInput.text?.toString().orEmpty()
        viewLifecycleOwner.lifecycleScope.launch {
            try {
                val url = viewModel.buildDownloadUrl(input)
                val intent = Intent(Intent.ACTION_VIEW, Uri.parse(url))
                startActivity(intent)
            } catch (ex: Exception) {
                Snackbar.make(binding.root, ex.message ?: getString(com.tradeapp.cryptobotremote.R.string.message_download_failed), Snackbar.LENGTH_LONG).show()
            }
        }
    }

    private fun enqueueDownload() {
        val input = binding.urlInput.text?.toString().orEmpty()
        viewLifecycleOwner.lifecycleScope.launch {
            try {
                val url = viewModel.buildDownloadUrl(input)
                val manager = downloadManager
                if (manager == null) {
                    Snackbar.make(binding.root, getString(com.tradeapp.cryptobotremote.R.string.message_download_failed), Snackbar.LENGTH_LONG).show()
                    return@launch
                }
                viewModel.logDownloadAttempt(url)
                val request = DownloadManager.Request(Uri.parse(url))
                    .setTitle("cryptobot_v3.apk")
                    .setDescription("TradeApp Android client")
                    .setMimeType(APK_MIME_TYPE)
                    .setNotificationVisibility(DownloadManager.Request.VISIBILITY_VISIBLE_NOTIFY_COMPLETED)
                    .setDestinationInExternalFilesDir(requireContext(), Environment.DIRECTORY_DOWNLOADS, "cryptobot_v3.apk")
                    .setAllowedOverMetered(true)
                    .setAllowedOverRoaming(true)
                downloadId = manager.enqueue(request)
                Snackbar.make(binding.root, com.tradeapp.cryptobotremote.R.string.message_download_started, Snackbar.LENGTH_SHORT).show()
            } catch (ex: Exception) {
                viewModel.logDownloadResult("Download failed: ${'$'}{ex.message}")
                Snackbar.make(binding.root, ex.message ?: getString(com.tradeapp.cryptobotremote.R.string.message_download_failed), Snackbar.LENGTH_LONG).show()
            }
        }
    }

    private fun handleDownloadFinished() {
        val manager = downloadManager ?: return
        val id = downloadId ?: return
        val uri = manager.getUriForDownloadedFile(id)
        if (uri != null) {
            viewModel.logDownloadResult("Download complete: ${'$'}uri")
            val installIntent = Intent(Intent.ACTION_VIEW).apply {
                setDataAndType(uri, APK_MIME_TYPE)
                flags = Intent.FLAG_GRANT_READ_URI_PERMISSION or Intent.FLAG_ACTIVITY_NEW_TASK
            }
            try {
                startActivity(installIntent)
                Snackbar.make(binding.root, com.tradeapp.cryptobotremote.R.string.message_install_prompt, Snackbar.LENGTH_SHORT).show()
            } catch (ex: Exception) {
                viewModel.logDownloadResult("Install failed: ${'$'}{ex.message}")
                Snackbar.make(binding.root, ex.message ?: getString(com.tradeapp.cryptobotremote.R.string.message_download_failed), Snackbar.LENGTH_LONG).show()
            }
        } else {
            viewModel.logDownloadResult("Download completed but URI missing")
        }
    }

    companion object {
        private const val APK_MIME_TYPE = "application/vnd.android.package-archive"
    }
}

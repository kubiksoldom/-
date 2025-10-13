package com.tradeapp.cryptobotremote

import android.content.Intent
import android.net.Uri
import android.os.Bundle
import androidx.activity.viewModels
import androidx.appcompat.app.AppCompatActivity
import androidx.core.view.isVisible
import androidx.navigation.NavController
import androidx.navigation.fragment.NavHostFragment
import androidx.navigation.ui.setupWithNavController
import com.google.android.material.snackbar.Snackbar
import com.tradeapp.cryptobotremote.databinding.ActivityMainBinding
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.Job
import kotlinx.coroutines.launch

class MainActivity : AppCompatActivity() {

    private lateinit var binding: ActivityMainBinding
    private val viewModel: GatewayViewModel by viewModels()
    private lateinit var navController: NavController
    private val activityJob = Job()
    private val activityScope = CoroutineScope(Dispatchers.Main + activityJob)

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        binding = ActivityMainBinding.inflate(layoutInflater)
        setContentView(binding.root)

        setSupportActionBar(binding.toolbar)

        val navHostFragment = supportFragmentManager
            .findFragmentById(binding.navHostFragment.id) as NavHostFragment
        navController = navHostFragment.navController
        binding.bottomNav.setupWithNavController(navController)

        binding.toolbar.isVisible = true

        handleIntent(intent)
    }

    override fun onNewIntent(intent: Intent?) {
        super.onNewIntent(intent)
        intent?.let { handleIntent(it) }
    }

    override fun onDestroy() {
        super.onDestroy()
        activityJob.cancel()
    }

    private fun handleIntent(intent: Intent) {
        if (intent.action == Intent.ACTION_VIEW) {
            val data: Uri? = intent.data
            if (data != null) {
                activityScope.launch {
                    try {
                        val downloadUrl = viewModel.buildDownloadUrl(data.toString())
                        viewModel.logDownloadResult("Deep link: ${'$'}downloadUrl")
                        navigateToDownload(downloadUrl)
                    } catch (ex: Exception) {
                        Snackbar.make(binding.root, ex.message ?: "Invalid link", Snackbar.LENGTH_LONG).show()
                    }
                }
            }
        }
    }

    private fun navigateToDownload(url: String) {
        binding.bottomNav.selectedItemId = com.tradeapp.cryptobotremote.R.id.navigation_download
        val bundle = Bundle().apply { putString("prefillUrl", url) }
        val options = androidx.navigation.navOptions {
            launchSingleTop = true
        }
        navController.navigate(com.tradeapp.cryptobotremote.R.id.downloadFragment, bundle, options)
    }
}

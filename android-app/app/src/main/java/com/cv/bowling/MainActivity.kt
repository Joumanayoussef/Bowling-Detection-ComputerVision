package com.cv.bowling

import android.Manifest
import android.app.Activity
import android.content.Intent
import android.content.pm.PackageManager
import android.net.Uri
import android.os.Bundle
import android.os.Handler
import android.os.Looper
import android.provider.OpenableColumns
import android.widget.Toast
import androidx.activity.result.contract.ActivityResultContracts
import androidx.appcompat.app.AppCompatActivity
import androidx.camera.core.*
import androidx.camera.lifecycle.ProcessCameraProvider
import androidx.core.content.ContextCompat
import androidx.lifecycle.ViewModelProvider
import com.cv.bowling.databinding.ActivityMainBinding
import java.io.File
import java.io.FileOutputStream
import java.util.concurrent.Executors

class MainActivity : AppCompatActivity() {

    private lateinit var binding      : ActivityMainBinding
    private lateinit var viewModel    : GameViewModel
    private lateinit var detector     : TFLiteDetector
    private lateinit var analyzer     : CameraAnalyzer

    private val pinTracker   = PinTracker()
    private val pathTracker  = PathTracker()
    private val scoreManager = ScoreManager()

    private val executor = Executors.newSingleThreadExecutor()
    private val handler  = Handler(Looper.getMainLooper())
    private var running  = false

    // FPS update
    private val fpsRunnable = object : Runnable {
        override fun run() {
            if (running) {
                binding.fpsText.text = "${analyzer.fps} FPS"
                handler.postDelayed(this, 1000)
            }
        }
    }

    // ── Permission launcher ──
    private val requestCameraPermission = registerForActivityResult(
        ActivityResultContracts.RequestPermission()
    ) { granted ->
        if (granted) startCamera()
        else Toast.makeText(this, "Camera permission required", Toast.LENGTH_LONG).show()
    }

    // ── Video file picker launcher ──
    private val pickVideoLauncher = registerForActivityResult(
        ActivityResultContracts.StartActivityForResult()
    ) { result ->
        if (result.resultCode == Activity.RESULT_OK) {
            result.data?.data?.let { uri ->
                handleVideoSelected(uri)
            }
        }
    }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        binding   = ActivityMainBinding.inflate(layoutInflater)
        viewModel = ViewModelProvider(this)[GameViewModel::class.java]
        setContentView(binding.root)

        // Load TFLite model
        try {
            detector = TFLiteDetector(this)
        } catch (e: Exception) {
            Toast.makeText(this,
                "Model not found! Copy bowling_model.tflite to assets/",
                Toast.LENGTH_LONG).show()
            return
        }

        analyzer = CameraAnalyzer(detector, pinTracker, pathTracker, scoreManager) { result ->
            viewModel.updateResult(result)
        }

        setupObservers()
        setupButtons()

        if (ContextCompat.checkSelfPermission(this, Manifest.permission.CAMERA)
            == PackageManager.PERMISSION_GRANTED) startCamera()
        else requestCameraPermission.launch(Manifest.permission.CAMERA)
    }

    private fun setupObservers() {
        viewModel.result.observe(this) { result ->
            binding.overlayView.update(result)
        }
        viewModel.score.observe(this) { score ->
            binding.scoreValueText.text = score.toString()
        }
        viewModel.status.observe(this) { status ->
            binding.statusText.text = status
        }
    }

    private fun setupButtons() {
        // ── Live camera START/STOP ──
        binding.btnStart.setOnClickListener {
            if (!running) startLiveDetection() else stopLiveDetection()
        }

        // ── Reset ──
        binding.btnReset.setOnClickListener {
            pinTracker.reset(); pathTracker.reset(); scoreManager.reset()
            binding.scoreValueText.text = "0"
            binding.overlayView.setRunning(false)
            binding.overlayView.setMessage("Press LIVE CAM or UPLOAD VIDEO")
            viewModel.updateStatus("READY")
            if (running) {
                running = false
                analyzer.stop()
                binding.btnStart.text = "▶ LIVE CAM"
                binding.btnStart.backgroundTintList =
                    ContextCompat.getColorStateList(this, R.color.primary)
                handler.removeCallbacks(fpsRunnable)
            }
        }

        // ── Upload video ──
        binding.btnUploadVideo.setOnClickListener {
            openVideoPicker()
        }
    }

    // ─────────────────────────────────────────────────────────
    //  Open file picker to select a video
    // ─────────────────────────────────────────────────────────
    private fun openVideoPicker() {
        val intent = Intent(Intent.ACTION_GET_CONTENT).apply {
            type = "video/*"
            addCategory(Intent.CATEGORY_OPENABLE)
        }
        pickVideoLauncher.launch(Intent.createChooser(intent, "Select Video"))
    }

    // ─────────────────────────────────────────────────────────
    //  Handle selected video — copy to cache then analyze
    // ─────────────────────────────────────────────────────────
    private fun handleVideoSelected(uri: Uri) {
        Toast.makeText(this, "Video selected — analyzing...", Toast.LENGTH_SHORT).show()

        // Stop live camera if running
        if (running) stopLiveDetection()

        // Copy video to cache (needed for MediaMetadataRetriever)
        val videoFile = copyUriToCache(uri) ?: run {
            Toast.makeText(this, "Cannot read video file", Toast.LENGTH_SHORT).show()
            return
        }

        // Open VideoResultActivity to analyze and show results
        val intent = Intent(this, VideoResultActivity::class.java).apply {
            putExtra("VIDEO_PATH", videoFile.absolutePath)
        }
        startActivity(intent)
    }

    // Copy video from Uri (content://) to a local cache file
    private fun copyUriToCache(uri: Uri): File? {
        return try {
            val fileName = getFileName(uri) ?: "video.mp4"
            val cacheFile = File(cacheDir, fileName)
            contentResolver.openInputStream(uri)?.use { input ->
                FileOutputStream(cacheFile).use { output ->
                    input.copyTo(output)
                }
            }
            cacheFile
        } catch (e: Exception) {
            null
        }
    }

    private fun getFileName(uri: Uri): String? {
        var name: String? = null
        contentResolver.query(uri, null, null, null, null)?.use { cursor ->
            if (cursor.moveToFirst()) {
                val idx = cursor.getColumnIndex(OpenableColumns.DISPLAY_NAME)
                if (idx >= 0) name = cursor.getString(idx)
            }
        }
        return name ?: "video_${System.currentTimeMillis()}.mp4"
    }

    // ─────────────────────────────────────────────────────────
    //  Live camera detection
    // ─────────────────────────────────────────────────────────
    private fun startLiveDetection() {
        running = true
        analyzer.start()
        binding.overlayView.setRunning(true)
        binding.btnStart.text = "⏹ STOP"
        binding.btnStart.backgroundTintList =
            ContextCompat.getColorStateList(this, android.R.color.holo_red_light)
        viewModel.updateStatus("DETECTING — drive the car!")
        handler.post(fpsRunnable)
    }

    private fun stopLiveDetection() {
        running = false
        analyzer.stop()
        binding.overlayView.setRunning(false)
        binding.btnStart.text = "▶ LIVE CAM"
        binding.btnStart.backgroundTintList =
            ContextCompat.getColorStateList(this, R.color.primary)
        viewModel.updateStatus("Stopped")
        handler.removeCallbacks(fpsRunnable)
    }

    private fun startCamera() {
        ProcessCameraProvider.getInstance(this).addListener({
            val provider = ProcessCameraProvider.getInstance(this).get()
            val preview  = Preview.Builder().build().apply {
                setSurfaceProvider(binding.previewView.surfaceProvider)
            }
            val analysis = ImageAnalysis.Builder()
                .setTargetResolution(android.util.Size(640, 480))
                .setBackpressureStrategy(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST)
                .build().also { it.setAnalyzer(executor, analyzer) }
            try {
                provider.unbindAll()
                provider.bindToLifecycle(this,
                    CameraSelector.DEFAULT_BACK_CAMERA, preview, analysis)
            } catch (e: Exception) {
                Toast.makeText(this, "Camera error", Toast.LENGTH_SHORT).show()
            }
        }, ContextCompat.getMainExecutor(this))
    }

    override fun onDestroy() {
        super.onDestroy()
        executor.shutdown()
        handler.removeCallbacks(fpsRunnable)
        if (::detector.isInitialized) detector.close()
    }
}

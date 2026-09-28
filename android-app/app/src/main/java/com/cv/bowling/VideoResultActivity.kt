package com.cv.bowling

import android.graphics.Bitmap
import android.os.Bundle
import android.os.Handler
import android.os.Looper
import android.view.View
import android.widget.*
import androidx.appcompat.app.AppCompatActivity
import androidx.lifecycle.lifecycleScope
import com.cv.bowling.databinding.ActivityVideoResultBinding
import kotlinx.coroutines.launch

/**
 * VideoResultActivity
 * ───────────────────
 * Shows the results after analyzing a video file.
 * Plays back annotated frames with car path + hit labels.
 * Shows final score and hit order list.
 */
class VideoResultActivity : AppCompatActivity() {

    private lateinit var binding   : ActivityVideoResultBinding
    private var annotatedFrames    = listOf<Bitmap>()
    private var frameIndex         = 0
    private var isPlaying          = false
    private val handler            = Handler(Looper.getMainLooper())

    // Plays frames at ~8 FPS (every 125ms)
    private val framePlayer = object : Runnable {
        override fun run() {
            if (!isPlaying || annotatedFrames.isEmpty()) return
            binding.frameView.setImageBitmap(annotatedFrames[frameIndex])
            frameIndex = (frameIndex + 1) % annotatedFrames.size
            handler.postDelayed(this, 125)
        }
    }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        binding = ActivityVideoResultBinding.inflate(layoutInflater)
        setContentView(binding.root)

        val videoPath = intent.getStringExtra("VIDEO_PATH") ?: run {
            showError("No video path provided")
            return
        }

        analyzeVideo(videoPath)

        binding.btnPlayPause.setOnClickListener {
            isPlaying = !isPlaying
            binding.btnPlayPause.text = if (isPlaying) "⏸ Pause" else "▶ Play"
            if (isPlaying) handler.post(framePlayer)
        }

        binding.btnRestart.setOnClickListener {
            frameIndex = 0
            isPlaying  = true
            binding.btnPlayPause.text = "⏸ Pause"
            handler.post(framePlayer)
        }

        binding.btnBack.setOnClickListener { finish() }
    }

    private fun analyzeVideo(videoPath: String) {
        binding.progressBar.visibility  = View.VISIBLE
        binding.statusText.text         = "Loading model..."
        binding.resultPanel.visibility  = View.GONE

        lifecycleScope.launch {
            try {
                val detector = TFLiteDetector(this@VideoResultActivity)
                val analyzer = VideoAnalyzer(this@VideoResultActivity)

                binding.statusText.text = "Analyzing video..."

                val result = analyzer.analyzeVideo(
                    videoPath  = videoPath,
                    detector   = detector,
                    onProgress = { current, total ->
                        val pct = (current * 100f / total).toInt()
                        binding.progressBar.progress = pct
                        binding.statusText.text = "Analyzing... $current/$total frames ($pct%)"
                    }
                )

                // Show results
                binding.progressBar.visibility = View.GONE
                binding.resultPanel.visibility = View.VISIBLE

                annotatedFrames = result.annotatedFrames

                // Auto-play
                if (annotatedFrames.isNotEmpty()) {
                    isPlaying = true
                    frameIndex = 0
                    handler.post(framePlayer)
                    binding.btnPlayPause.text = "⏸ Pause"
                }

                // Score summary
                binding.scoreText.text = "Score: ${result.totalScore}"
                binding.pathText.text  = "Path points: ${result.carPathPoints}"

                // Hit order list
                if (result.hitOrder.isEmpty()) {
                    binding.hitListText.text = "No hits detected.\nMake sure the model file is in assets/."
                } else {
                    val sb = StringBuilder()
                    sb.appendLine("Bottles hit in order:\n")
                    result.hitOrder.forEach { hit ->
                        val ordinal = when(hit.hitOrder){1->"1st";2->"2nd";3->"3rd";else->"${hit.hitOrder}th"}
                        sb.appendLine("  $ordinal hit → Pin ${hit.pinId}  (${hit.timeMs/1000f}s)")
                    }
                    binding.hitListText.text = sb.toString()
                }

                detector.close()

            } catch (e: Exception) {
                showError("Error: ${e.message}")
            }
        }
    }

    private fun showError(msg: String) {
        binding.progressBar.visibility = View.GONE
        binding.statusText.text        = msg
    }

    override fun onDestroy() {
        super.onDestroy()
        isPlaying = false
        handler.removeCallbacks(framePlayer)
    }
}

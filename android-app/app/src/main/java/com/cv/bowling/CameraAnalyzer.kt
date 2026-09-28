package com.cv.bowling

import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.graphics.ImageFormat
import android.graphics.PointF
import android.graphics.Rect
import android.graphics.RectF
import android.graphics.YuvImage
import androidx.camera.core.ImageAnalysis
import androidx.camera.core.ImageProxy
import java.io.ByteArrayOutputStream

/**
 * CameraAnalyzer
 * ──────────────
 * Receives each camera frame, runs the full pipeline:
 *
 *  1. Convert YUV_420_888 → JPEG → Bitmap
 *  2. Run TFLiteDetector → get list of Detection objects
 *  3. Find car detection → get bounding box + centroid
 *  4. Update PathTracker with car centroid
 *  5. Update PinTracker with pin detections
 *  6. Run ScoreManager collision check
 *  7. Build FrameResult → send to UI via callback
 */
class CameraAnalyzer(
    private val detector    : TFLiteDetector,
    private val pinTracker  : PinTracker,
    private val pathTracker : PathTracker,
    private val scoreManager: ScoreManager,
    private val onResult    : (FrameResult) -> Unit
) : ImageAnalysis.Analyzer {

    private var active  = false
    private var lastFps = System.currentTimeMillis()
    private var fpsCnt  = 0
    var fps             = 0
        private set

    fun start() { active = true  }
    fun stop()  { active = false }

    override fun analyze(image: ImageProxy) {
        if (!active) { image.close(); return }

        try {
            val bitmap = yuvToBitmap(image) ?: run { image.close(); return }
            val W = bitmap.width
            val H = bitmap.height

            // ── Run YOLO inference ──
            val detections = detector.detect(bitmap)
            bitmap.recycle()

            // ── Find car ──
            val carDet = detections
                .filter { it.classId == Classes.CAR }
                .maxByOrNull { it.confidence }

            val carBox = carDet?.box
            if (carBox != null) {
                val cx = (carBox.left + carBox.right) / 2f
                val cy = (carBox.top  + carBox.bottom) / 2f
                pathTracker.addPoint(cx, cy)
            }

            // ── Update pin tracking ──
            val now  = System.currentTimeMillis()
            val pins = pinTracker.update(detections, now)

            // ── Scoring ──
            val score = scoreManager.checkCollisions(carBox, pins, pinTracker, now)

            // ── Build result ──
            val result = FrameResult(
                detections  = detections,
                trackedPins = pins,
                carBox      = carBox,
                carPath     = pathTracker.getPath(),
                score       = score,
                imageW      = W,
                imageH      = H
            )

            // FPS
            fpsCnt++
            val t = System.currentTimeMillis()
            if (t - lastFps >= 1000) { fps = fpsCnt; fpsCnt = 0; lastFps = t }

            onResult(result)

        } catch (e: Exception) {
            e.printStackTrace()
        } finally {
            image.close()
        }
    }

    // ─────────────────────────────────────────────────────────
    //  Convert CameraX YUV frame → Bitmap
    // ─────────────────────────────────────────────────────────
    private fun yuvToBitmap(image: ImageProxy): Bitmap? {
        return try {
            val yBuf = image.planes[0].buffer
            val uBuf = image.planes[1].buffer
            val vBuf = image.planes[2].buffer
            val ySize = yBuf.remaining()
            val uSize = uBuf.remaining()
            val vSize = vBuf.remaining()
            val nv21  = ByteArray(ySize + uSize + vSize)
            yBuf.get(nv21, 0, ySize)
            vBuf.get(nv21, ySize, vSize)
            uBuf.get(nv21, ySize + vSize, uSize)
            val yuv  = YuvImage(nv21, ImageFormat.NV21, image.width, image.height, null)
            val out  = ByteArrayOutputStream()
            yuv.compressToJpeg(Rect(0, 0, image.width, image.height), 85, out)
            val b    = out.toByteArray()
            BitmapFactory.decodeByteArray(b, 0, b.size)
        } catch (e: Exception) { null }
    }
}

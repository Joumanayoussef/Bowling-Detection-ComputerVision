package com.cv.bowling

import android.content.Context
import android.graphics.Bitmap
import android.graphics.RectF
import org.tensorflow.lite.Interpreter
import java.io.FileInputStream
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.MappedByteBuffer
import java.nio.channels.FileChannel
import kotlin.math.max
import kotlin.math.min

/**
 * TFLiteDetector
 * ──────────────
 * Loads bowling_model.tflite from assets and runs YOLOv8 inference.
 *
 * Pipeline:
 *  1. Resize input bitmap to 320x320
 *  2. Convert to float32 RGB ByteBuffer (normalized 0-1)
 *  3. Run TFLite interpreter
 *  4. Parse YOLOv8 output tensor → list of Detections
 *  5. Apply NMS (Non-Max Suppression) to remove duplicate boxes
 */
class TFLiteDetector(context: Context) {

    private val interpreter   : Interpreter
    private val labels        : List<String>
    private val inputSize     = 320
    private val confThreshold = 0.40f   // minimum confidence
    private val nmsThreshold  = 0.45f   // NMS IoU threshold

    init {
        // Load model
        val opts = Interpreter.Options().apply { setNumThreads(4) }
        interpreter = Interpreter(loadModelFile(context, "bowling_model.tflite"), opts)

        // Load labels
        labels = context.assets.open("labels.txt")
            .bufferedReader().readLines()
            .filter { it.isNotBlank() }
    }

    // ─────────────────────────────────────────────────────────
    //  MAIN INFERENCE — called every frame
    //  Returns list of Detection objects in ORIGINAL image coords
    // ─────────────────────────────────────────────────────────
    fun detect(bitmap: Bitmap): List<Detection> {
        val origW = bitmap.width.toFloat()
        val origH = bitmap.height.toFloat()

        // 1. Resize to 320x320
        val resized  = Bitmap.createScaledBitmap(bitmap, inputSize, inputSize, true)
        val inputBuf = bitmapToByteBuffer(resized)
        resized.recycle()

        // 2. Run inference
        // YOLOv8 output shape: [1, num_classes+4, num_boxes]
        // For 4 classes: [1, 8, 8400]
        val numBoxes    = 8400
        val numAttribs  = 4 + labels.size   // x,y,w,h + class scores
        val outputArray = Array(1) { Array(numAttribs) { FloatArray(numBoxes) } }
        interpreter.run(inputBuf, outputArray)

        // 3. Parse output
        val detections = mutableListOf<Detection>()
        val output     = outputArray[0]

        for (i in 0 until numBoxes) {
            // Find best class
            var maxConf   = 0f
            var classId   = 0
            for (c in 0 until labels.size) {
                val score = output[4 + c][i]
                if (score > maxConf) { maxConf = score; classId = c }
            }
            if (maxConf < confThreshold) continue

            // YOLOv8 output: cx, cy, w, h (normalized 0-1)
            val cx = output[0][i]
            val cy = output[1][i]
            val bw = output[2][i]
            val bh = output[3][i]

            // Convert to pixel coords in ORIGINAL image space
            val left   = ((cx - bw / 2f) * origW).coerceIn(0f, origW)
            val top    = ((cy - bh / 2f) * origH).coerceIn(0f, origH)
            val right  = ((cx + bw / 2f) * origW).coerceIn(0f, origW)
            val bottom = ((cy + bh / 2f) * origH).coerceIn(0f, origH)

            if (right > left && bottom > top) {
                detections.add(Detection(
                    classId    = classId,
                    label      = labels.getOrElse(classId) { "unknown" },
                    confidence = maxConf,
                    box        = RectF(left, top, right, bottom)
                ))
            }
        }

        // 4. NMS per class
        return nonMaxSuppression(detections)
    }

    // ─────────────────────────────────────────────────────────
    //  NON-MAX SUPPRESSION
    //  Removes overlapping boxes — keeps highest confidence one.
    //  Same algorithm used inside YOLO during training.
    // ─────────────────────────────────────────────────────────
    private fun nonMaxSuppression(dets: List<Detection>): List<Detection> {
        val result  = mutableListOf<Detection>()
        val byClass = dets.groupBy { it.classId }

        for ((_, classDets) in byClass) {
            val sorted  = classDets.sortedByDescending { it.confidence }.toMutableList()
            val kept    = mutableListOf<Detection>()

            while (sorted.isNotEmpty()) {
                val best = sorted.removeAt(0)
                kept.add(best)
                sorted.removeAll { iou(best.box, it.box) > nmsThreshold }
            }
            result.addAll(kept)
        }
        return result
    }

    private fun iou(a: RectF, b: RectF): Float {
        val iL    = max(a.left, b.left);   val iT = max(a.top, b.top)
        val iR    = min(a.right, b.right); val iB = min(a.bottom, b.bottom)
        if (iR <= iL || iB <= iT) return 0f
        val inter = (iR - iL) * (iB - iT)
        val union = a.width()*a.height() + b.width()*b.height() - inter
        return if (union <= 0f) 0f else inter / union
    }

    // ─────────────────────────────────────────────────────────
    //  Convert Bitmap → ByteBuffer (float32, normalized 0-1)
    // ─────────────────────────────────────────────────────────
    private fun bitmapToByteBuffer(bmp: Bitmap): ByteBuffer {
        val buf = ByteBuffer
            .allocateDirect(1 * inputSize * inputSize * 3 * 4)
            .apply { order(ByteOrder.nativeOrder()) }

        val pixels = IntArray(inputSize * inputSize)
        bmp.getPixels(pixels, 0, inputSize, 0, 0, inputSize, inputSize)

        for (px in pixels) {
            buf.putFloat(((px shr 16) and 0xFF) / 255f)   // R
            buf.putFloat(((px shr 8)  and 0xFF) / 255f)   // G
            buf.putFloat((px          and 0xFF) / 255f)   // B
        }
        buf.rewind()
        return buf
    }

    private fun loadModelFile(ctx: Context, filename: String): MappedByteBuffer {
        val fd      = ctx.assets.openFd(filename)
        val stream  = FileInputStream(fd.fileDescriptor)
        val channel = stream.channel
        return channel.map(FileChannel.MapMode.READ_ONLY, fd.startOffset, fd.declaredLength)
    }

    fun close() = interpreter.close()
}

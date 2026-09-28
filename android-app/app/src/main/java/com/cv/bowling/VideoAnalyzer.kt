package com.cv.bowling

import android.content.Context
import android.graphics.Bitmap
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.Paint
import android.graphics.PointF
import android.graphics.RectF
import android.graphics.Typeface
import android.media.MediaMetadataRetriever
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext
import kotlin.math.atan2
import kotlin.math.cos
import kotlin.math.sin
import kotlin.math.sqrt

/**
 * VideoAnalyzer
 * ─────────────
 * Processes a video file frame by frame using the YOLO TFLite model.
 * Used when user uploads a pre-recorded video instead of live camera.
 *
 * Pipeline per frame:
 *  1. Extract frame as Bitmap from video file
 *  2. Run TFLiteDetector → get detections
 *  3. Track car → update path
 *  4. Track pins → detect state changes (standing → fallen)
 *  5. Score hits in order of timestamp
 *  6. Annotate frame with boxes + path + labels
 *  7. Return all annotated frames for playback
 */
class VideoAnalyzer(private val context: Context) {

    data class VideoResult(
        val annotatedFrames : List<Bitmap>,
        val hitOrder        : List<HitEvent>,
        val totalScore      : Int,
        val carPathPoints   : Int,
        val durationSeconds : Float
    )

    data class HitEvent(
        val pinId     : Int,
        val hitOrder  : Int,
        val timeMs    : Long,
        val position  : PointF
    )

    suspend fun analyzeVideo(
        videoPath : String,
        detector  : TFLiteDetector,
        onProgress: (Int, Int) -> Unit   // current frame, total frames
    ): VideoResult = withContext(Dispatchers.Default) {

        val retriever = MediaMetadataRetriever()
        retriever.setDataSource(videoPath)

        val durationMs = retriever.extractMetadata(
            MediaMetadataRetriever.METADATA_KEY_DURATION)?.toLong() ?: 0L
        val origW = retriever.extractMetadata(
            MediaMetadataRetriever.METADATA_KEY_VIDEO_WIDTH)?.toInt() ?: 640
        val origH = retriever.extractMetadata(
            MediaMetadataRetriever.METADATA_KEY_VIDEO_HEIGHT)?.toInt() ?: 480

        // Extract one frame every 150ms for smooth playback
        val INTERVAL_MS = 150L
        val frames = mutableListOf<Pair<Bitmap, Long>>()  // bitmap, timestamp
        var t = 0L
        while (t <= durationMs) {
            val bmp = retriever.getFrameAtTime(
                t * 1000L, MediaMetadataRetriever.OPTION_CLOSEST_SYNC)
            if (bmp != null) frames.add(bmp to t)
            t += INTERVAL_MS
        }
        retriever.release()

        val totalFrames = frames.size

        // ── Per-frame state tracking ──
        val pinTracker   = PinTracker()
        val pathTracker  = PathTracker()
        val scoreManager = ScoreManager()
        val hitEvents    = mutableListOf<HitEvent>()
        val annotated    = mutableListOf<Bitmap>()

        for ((fi, pair) in frames.withIndex()) {
            val (bitmap, timeMs) = pair

            // Report progress to UI
            withContext(Dispatchers.Main) { onProgress(fi + 1, totalFrames) }

            // Run YOLO detection
            val detections = try { detector.detect(bitmap) }
            catch (e: Exception) { emptyList() }

            // Find car
            val carDet = detections
                .filter { it.classId == Classes.CAR }
                .maxByOrNull { it.confidence }
            val carBox = carDet?.box

            // Update path
            if (carBox != null) {
                val cx = (carBox.left + carBox.right) / 2f
                val cy = (carBox.top + carBox.bottom) / 2f
                pathTracker.addPoint(cx, cy)
            }

            // Update pin tracking — detect fallen pins
            val pins = pinTracker.update(detections, timeMs)

            // Score collisions
            scoreManager.checkCollisions(carBox, pins, pinTracker, timeMs)

            // Collect new hit events this frame
            for (pin in pins) {
                if (pin.state == PinState.FALLEN && pin.hitOrder > 0) {
                    val already = hitEvents.any { it.pinId == pin.id }
                    if (!already) {
                        hitEvents.add(HitEvent(
                            pinId    = pin.id,
                            hitOrder = pin.hitOrder,
                            timeMs   = pin.hitTime,
                            position = PointF(
                                (pin.box.left + pin.box.right) / 2f,
                                (pin.box.top  + pin.box.bottom) / 2f
                            )
                        ))
                    }
                }
            }

            // Annotate this frame
            val hitsUpToNow = hitEvents.toList()
            val annotatedBmp = annotateFrame(
                bmp          = bitmap.copy(Bitmap.Config.ARGB_8888, true),
                detections   = detections,
                carBox       = carBox,
                carPath      = pathTracker.getPath(),
                trackedPins  = pins,
                hitEvents    = hitsUpToNow,
                imageW       = bitmap.width,
                imageH       = bitmap.height
            )
            annotated.add(annotatedBmp)
        }

        val sortedHits = hitEvents.sortedBy { it.timeMs }

        VideoResult(
            annotatedFrames  = annotated,
            hitOrder         = sortedHits,
            totalScore       = pinTracker.getScore(),
            carPathPoints    = pathTracker.getPath().size,
            durationSeconds  = durationMs / 1000f
        )
    }

    // ─────────────────────────────────────────────────────────
    //  Draw all CV results on a single frame
    // ─────────────────────────────────────────────────────────
    private fun annotateFrame(
        bmp         : Bitmap,
        detections  : List<Detection>,
        carBox      : RectF?,
        carPath     : List<PointF>,
        trackedPins : List<TrackedPin>,
        hitEvents   : List<HitEvent>,
        imageW      : Int,
        imageH      : Int
    ): Bitmap {
        val canvas = Canvas(bmp)
        val W = bmp.width.toFloat()
        val H = bmp.height.toFloat()
        val sx = W / imageW
        val sy = H / imageH

        // ── 1. Draw car path ──
        drawPath(canvas, carPath, sx, sy)

        // ── 2. Draw standing pins (white boxes) ──
        for (pin in trackedPins) {
            if (pin.state == PinState.STANDING) {
                drawStandingPin(canvas, pin, sx, sy)
            }
        }

        // ── 3. Draw hit pins (green boxes + order number) ──
        for (pin in trackedPins) {
            if (pin.state == PinState.FALLEN && pin.hitOrder > 0) {
                drawHitPin(canvas, pin, sx, sy)
            }
        }

        // ── 4. Draw car box ──
        carBox?.let { drawCar(canvas, it, sx, sy) }

        // ── 5. Score panel ──
        if (hitEvents.isNotEmpty()) {
            drawScorePanel(canvas, hitEvents)
        }

        return bmp
    }

    private fun drawPath(canvas: Canvas, path: List<PointF>, sx: Float, sy: Float) {
        if (path.size < 2) return

        val dotPaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
            color = Color.parseColor("#00BCD4")
            style = Paint.Style.FILL
        }
        var acc = 0f; val sp = 20f
        for (i in 1 until path.size) {
            val x0 = path[i-1].x*sx; val y0 = path[i-1].y*sy
            val x1 = path[i].x*sx;   val y1 = path[i].y*sy
            val len = hypot(x1-x0, y1-y0)
            if (len < 1f) continue
            val dx=(x1-x0)/len; val dy=(y1-y0)/len
            var w=sp-acc
            while(w<=len){ canvas.drawCircle(x0+dx*w,y0+dy*w,5f,dotPaint); w+=sp }
            acc=(len-(w-sp)).coerceAtLeast(0f)
        }
        // START
        canvas.drawCircle(path[0].x*sx, path[0].y*sy, 12f,
            Paint(Paint.ANTI_ALIAS_FLAG).apply { color=Color.parseColor("#43A047") })
        canvas.drawText("START", path[0].x*sx+16f, path[0].y*sy+10f,
            makePaint(Color.WHITE, 24f).apply { setShadowLayer(4f,0f,0f,Color.BLACK) })

        // Arrow at end
        if (path.size >= 3) {
            val p2=path[path.size-2]; val p1=path.last()
            val angle=atan2((p1.y-p2.y).toDouble(),(p1.x-p2.x).toDouble())
            canvas.drawLine(p1.x*sx,p1.y*sy,
                (p1.x*sx+cos(angle)*36).toFloat(),(p1.y*sy+sin(angle)*36).toFloat(),
                Paint(Paint.ANTI_ALIAS_FLAG).apply {
                    color=Color.parseColor("#FF5722"); strokeWidth=5f; strokeCap=Paint.Cap.ROUND
                })
        }
    }

    private fun drawStandingPin(canvas: Canvas, pin: TrackedPin, sx: Float, sy: Float) {
        val l=pin.box.left*sx; val t=pin.box.top*sy
        val r=pin.box.right*sx; val b=pin.box.bottom*sy
        canvas.drawRect(l,t,r,b, Paint(Paint.ANTI_ALIAS_FLAG).apply {
            color=Color.WHITE; style=Paint.Style.STROKE; strokeWidth=3f })
        canvas.drawText("P${pin.id}", (l+r)/2f, t-8f,
            makePaint(Color.WHITE, 22f, Paint.Align.CENTER).apply {
                setShadowLayer(3f,0f,0f,Color.BLACK) })
    }

    private fun drawHitPin(canvas: Canvas, pin: TrackedPin, sx: Float, sy: Float) {
        val l=pin.box.left*sx; val t=pin.box.top*sy
        val r=pin.box.right*sx; val b=pin.box.bottom*sy
        val cx=(l+r)/2f; val cy=(t+b)/2f

        // Green fill
        canvas.drawRect(l,t,r,b, Paint(Paint.ANTI_ALIAS_FLAG).apply {
            color=Color.parseColor("#5500C853") })
        // Green border
        canvas.drawRect(l,t,r,b, Paint(Paint.ANTI_ALIAS_FLAG).apply {
            color=Color.parseColor("#00C853"); style=Paint.Style.STROKE; strokeWidth=5f })
        // Big number
        canvas.drawText(pin.hitOrder.toString(), cx, cy+22f,
            Paint(Paint.ANTI_ALIAS_FLAG).apply {
                color=Color.WHITE; textSize=68f; typeface=Typeface.DEFAULT_BOLD
                textAlign=Paint.Align.CENTER; setShadowLayer(6f,0f,0f,Color.BLACK) })
        // Badge
        val badge="Hit #${pin.hitOrder}"
        val bw=badge.length*14f+22f
        canvas.drawRoundRect(cx-bw/2,t-52f,cx+bw/2,t-6f,10f,10f,
            Paint(Paint.ANTI_ALIAS_FLAG).apply { color=Color.parseColor("#00C853") })
        canvas.drawText(badge,cx,t-18f,
            Paint(Paint.ANTI_ALIAS_FLAG).apply {
                color=Color.BLACK; textSize=26f; typeface=Typeface.DEFAULT_BOLD
                textAlign=Paint.Align.CENTER })
    }

    private fun drawCar(canvas: Canvas, box: RectF, sx: Float, sy: Float) {
        val l=box.left*sx; val t=box.top*sy
        val r=box.right*sx; val b=box.bottom*sy
        canvas.drawRect(l,t,r,b, Paint(Paint.ANTI_ALIAS_FLAG).apply {
            color=Color.parseColor("#442196F3"); style=Paint.Style.FILL })
        canvas.drawRect(l,t,r,b, Paint(Paint.ANTI_ALIAS_FLAG).apply {
            color=Color.parseColor("#2196F3"); style=Paint.Style.STROKE; strokeWidth=5f })
        canvas.drawText("CAR",(l+r)/2f,t-10f,
            makePaint(Color.parseColor("#2196F3"),26f,Paint.Align.CENTER).apply {
                setShadowLayer(4f,0f,0f,Color.BLACK) })
    }

    private fun drawScorePanel(canvas: Canvas, hits: List<HitEvent>) {
        val ph = 52f + hits.size * 36f
        canvas.drawRoundRect(10f,10f,280f,ph,14f,14f,
            Paint(Paint.ANTI_ALIAS_FLAG).apply { color=Color.parseColor("#CC000000") })
        canvas.drawText("Score: ${hits.size}", 18f, 44f,
            makePaint(Color.parseColor("#4CAF50"),30f).apply { typeface=Typeface.DEFAULT_BOLD })
        hits.sortedBy{it.hitOrder}.forEachIndexed { i, h ->
            val s=when(h.hitOrder){1->"1st";2->"2nd";3->"3rd";else->"${h.hitOrder}th"}
            canvas.drawText("$s hit → Pin ${h.pinId}  @${h.timeMs/1000f}s",
                18f, 82f+i*36f, makePaint(Color.WHITE, 22f))
        }
    }

    private fun makePaint(
        color: Int, size: Float,
        align: Paint.Align = Paint.Align.LEFT
    ) = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        this.color=color; textSize=size; textAlign=align
    }

    private fun hypot(dx: Float, dy: Float) =
        sqrt((dx*dx+dy*dy).toDouble()).toFloat()
}

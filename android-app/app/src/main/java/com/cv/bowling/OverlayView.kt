package com.cv.bowling

import android.content.Context
import android.graphics.*
import android.util.AttributeSet
import android.view.View
import kotlin.math.atan2
import kotlin.math.cos
import kotlin.math.sin

/**
 * OverlayView
 * ───────────
 * Custom View drawn on top of CameraX preview.
 *
 * Draws:
 *  - Blue box   = RC car
 *  - White box  = standing pin
 *  - Green box  = fallen pin + "Hit #N" label + order number
 *  - Cyan dots  = car path trajectory
 *  - Score panel top-right
 *
 * Coordinate mapping:
 *  TFLite runs at 320x320 input but detections are mapped back
 *  to the original camera frame size. OverlayView scales those
 *  coords to whatever the view's actual pixel size is.
 */
class OverlayView @JvmOverloads constructor(
    context: Context, attrs: AttributeSet? = null
) : View(context, attrs) {

    private var result   : FrameResult? = null
    private var running  = false
    private var buildMsg = "Press START"

    // ── Paints ──────────────────────────────────────────────
    private val carBoxPaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        color = Color.parseColor("#2196F3"); style = Paint.Style.STROKE; strokeWidth = 5f
    }
    private val carFillPaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        color = Color.parseColor("#442196F3"); style = Paint.Style.FILL
    }
    private val standingBoxPaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        color = Color.WHITE; style = Paint.Style.STROKE; strokeWidth = 3f
    }
    private val hitBoxPaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        color = Color.parseColor("#00C853"); style = Paint.Style.STROKE; strokeWidth = 5f
    }
    private val hitFillPaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        color = Color.parseColor("#5500C853"); style = Paint.Style.FILL
    }
    private val hitBadgePaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        color = Color.parseColor("#00C853")
    }
    private val pathDotPaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        color = Color.parseColor("#00BCD4"); style = Paint.Style.FILL
    }
    private val arrowPaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        color = Color.parseColor("#FF5722"); strokeWidth = 5f; strokeCap = Paint.Cap.ROUND
    }
    private val statusPaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        color = Color.parseColor("#FFFF00"); textSize = 34f; textAlign = Paint.Align.CENTER
        setShadowLayer(4f, 0f, 0f, Color.BLACK)
    }
    // ────────────────────────────────────────────────────────

    fun update(r: FrameResult) { result = r; postInvalidate() }
    fun setRunning(r: Boolean) { running = r; postInvalidate() }
    fun setMessage(m: String)  { buildMsg = m; postInvalidate() }

    override fun onDraw(canvas: Canvas) {
        super.onDraw(canvas)

        if (!running) {
            canvas.drawText(buildMsg, width / 2f, height / 2f, statusPaint)
            return
        }

        val r = result ?: return

        // Scale: map detection coords (in r.imageW x r.imageH) → view pixels
        val sx = width.toFloat()  / r.imageW
        val sy = height.toFloat() / r.imageH

        // 1. Car path
        drawPath(canvas, r.carPath, sx, sy)

        // 2. Pins
        for (pin in r.trackedPins) drawPin(canvas, pin, sx, sy)

        // 3. Car box on top
        r.carBox?.let { drawCar(canvas, it, sx, sy) }
    }

    private fun drawPath(canvas: Canvas, path: List<PointF>, sx: Float, sy: Float) {
        if (path.size < 2) return

        val dotPaint = pathDotPaint
        var acc = 0f; val sp = 20f
        for (i in 1 until path.size) {
            val x0 = path[i-1].x*sx; val y0 = path[i-1].y*sy
            val x1 = path[i].x*sx;   val y1 = path[i].y*sy
            val len = dist(x0, y0, x1, y1)
            if (len < 1f) continue
            val dx = (x1-x0)/len; val dy = (y1-y0)/len
            var w = sp - acc
            while (w <= len) { canvas.drawCircle(x0+dx*w, y0+dy*w, 5f, dotPaint); w += sp }
            acc = (len - (w - sp)).coerceAtLeast(0f)
        }

        // START marker
        canvas.drawCircle(path[0].x*sx, path[0].y*sy, 12f,
            Paint(Paint.ANTI_ALIAS_FLAG).apply { color = Color.parseColor("#43A047") })
        canvas.drawText("START", path[0].x*sx, path[0].y*sy - 16f,
            textPaint(Color.WHITE, 22f))

        // Direction arrow
        if (path.size >= 3) {
            val p2 = path[path.size-2]; val p1 = path.last()
            val angle = atan2((p1.y-p2.y).toDouble(), (p1.x-p2.x).toDouble())
            canvas.drawLine(p1.x*sx, p1.y*sy,
                (p1.x*sx + cos(angle)*36).toFloat(),
                (p1.y*sy + sin(angle)*36).toFloat(), arrowPaint)
        }
    }

    private fun drawPin(canvas: Canvas, pin: TrackedPin, sx: Float, sy: Float) {
        val l = pin.box.left*sx; val t = pin.box.top*sy
        val r = pin.box.right*sx; val b = pin.box.bottom*sy
        val cx = (l+r)/2f; val cy = (t+b)/2f

        when (pin.state) {
            PinState.STANDING -> {
                canvas.drawRect(l, t, r, b, standingBoxPaint)
                canvas.drawText("Pin ${pin.id}", cx, t - 8f,
                    textPaint(Color.WHITE, 22f, Paint.Align.CENTER))
            }
            PinState.FALLEN -> {
                canvas.drawRect(l, t, r, b, hitFillPaint)
                canvas.drawRect(l, t, r, b, hitBoxPaint)

                // Large hit number
                canvas.drawText(pin.hitOrder.toString(), cx, cy + 24f,
                    Paint(Paint.ANTI_ALIAS_FLAG).apply {
                        color = Color.WHITE; textSize = 68f; typeface = Typeface.DEFAULT_BOLD
                        textAlign = Paint.Align.CENTER
                        setShadowLayer(6f, 0f, 0f, Color.BLACK)
                    })

                // "Hit #N" badge above box
                val badge = "Hit #${pin.hitOrder}"
                val bw    = badge.length * 14f + 22f
                canvas.drawRoundRect(cx-bw/2, t-52f, cx+bw/2, t-6f, 10f, 10f, hitBadgePaint)
                canvas.drawText(badge, cx, t-18f,
                    Paint(Paint.ANTI_ALIAS_FLAG).apply {
                        color = Color.BLACK; textSize = 26f; typeface = Typeface.DEFAULT_BOLD
                        textAlign = Paint.Align.CENTER
                    })
            }
        }
    }

    private fun drawCar(canvas: Canvas, box: RectF, sx: Float, sy: Float) {
        val l = box.left*sx; val t = box.top*sy
        val r = box.right*sx; val b = box.bottom*sy
        canvas.drawRect(l, t, r, b, carFillPaint)
        canvas.drawRect(l, t, r, b, carBoxPaint)
        canvas.drawText("CAR", (l+r)/2f, t - 10f,
            Paint(Paint.ANTI_ALIAS_FLAG).apply {
                color = Color.parseColor("#2196F3"); textSize = 26f
                typeface = Typeface.DEFAULT_BOLD; textAlign = Paint.Align.CENTER
                setShadowLayer(4f, 0f, 0f, Color.BLACK)
            })
        canvas.drawCircle((l+r)/2f, (t+b)/2f, 6f,
            Paint(Paint.ANTI_ALIAS_FLAG).apply { color = Color.parseColor("#2196F3") })
    }

    private fun textPaint(
        color: Int, size: Float,
        align: Paint.Align = Paint.Align.LEFT
    ) = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        this.color = color; textSize = size; textAlign = align
        typeface = Typeface.DEFAULT_BOLD; setShadowLayer(3f, 0f, 0f, Color.BLACK)
    }

    private fun dist(x0: Float, y0: Float, x1: Float, y1: Float) =
        Math.sqrt(((x1-x0)*(x1-x0) + (y1-y0)*(y1-y0)).toDouble()).toFloat()
}

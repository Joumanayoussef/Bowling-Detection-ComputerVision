package com.cv.bowling

import android.graphics.PointF
import kotlin.math.sqrt

/**
 * PathTracker
 * ───────────
 * Records the RC car's center point every frame and
 * smooths it with a moving average to remove jitter.
 *
 * Only adds a new point if the car moved more than MIN_DIST pixels
 * — this avoids cluttering the path when the car is stationary.
 */
class PathTracker {

    private val raw       = mutableListOf<PointF>()
    private val smoothed  = mutableListOf<PointF>()
    private val WINDOW    = 5       // smoothing window size
    private val MIN_DIST  = 6f      // minimum movement to record

    fun addPoint(x: Float, y: Float) {
        val last = raw.lastOrNull()
        val dist = if (last != null)
            sqrt((x - last.x)*(x - last.x) + (y - last.y)*(y - last.y))
        else Float.MAX_VALUE

        if (dist >= MIN_DIST) {
            raw.add(PointF(x, y))
            recomputeSmoothed()
        }
    }

    private fun recomputeSmoothed() {
        smoothed.clear()
        for (i in raw.indices) {
            val from   = maxOf(0, i - WINDOW / 2)
            val to     = minOf(raw.size - 1, i + WINDOW / 2)
            val window = raw.subList(from, to + 1)
            smoothed.add(PointF(
                window.map { it.x }.average().toFloat(),
                window.map { it.y }.average().toFloat()
            ))
        }
    }

    fun getPath() = smoothed.toList()
    fun isEmpty() = raw.isEmpty()

    fun reset() { raw.clear(); smoothed.clear() }
}

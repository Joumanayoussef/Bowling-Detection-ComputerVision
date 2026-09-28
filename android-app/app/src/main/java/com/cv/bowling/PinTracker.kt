package com.cv.bowling

import android.graphics.RectF
import kotlin.math.sqrt

/**
 * PinTracker
 * ──────────
 * Tracks pins across frames and detects when they fall.
 *
 * Strategy:
 *  - Setup phase (first N frames): lock in pin positions from
 *    standing_pin detections. These are the bottles before any hit.
 *  - Game phase: watch each tracked pin.
 *    If it transitions from STANDING → FALLEN class, or disappears
 *    after the car was nearby → count as hit.
 *  - Assign hit order by timestamp: first pin to fall = Hit #1, etc.
 *
 * Matching: nearest-neighbor by IoU + center distance.
 */
class PinTracker {

    private val pins       = mutableMapOf<Int, TrackedPin>()
    private var nextPinId  = 1
    private var hitCounter = 0
    val hitPins            = mutableListOf<TrackedPin>()

    // ─────────────────────────────────────────────────────────
    //  Update pins from current frame detections
    // ─────────────────────────────────────────────────────────
    fun update(detections: List<Detection>, currentTime: Long): List<TrackedPin> {

        val standingDets = detections.filter { it.classId == Classes.STANDING_PIN }
        val fallenDets   = detections.filter { it.classId == Classes.FALLEN_PIN }

        // ── Match existing pins to standing detections ──
        val matched     = mutableSetOf<Int>()
        val usedPinIds  = mutableSetOf<Int>()

        for (det in standingDets) {
            var bestDist = 120f
            var bestId   = -1
            for ((id, pin) in pins) {
                if (id in usedPinIds) continue
                if (pin.state == PinState.FALLEN) continue
                val d = centerDist(det.box, pin.box)
                if (d < bestDist) { bestDist = d; bestId = id }
            }
            if (bestId >= 0) {
                pins[bestId]!!.box = det.box
                matched.add(bestId)
                usedPinIds.add(bestId)
            } else {
                // New pin detected — add to tracking
                val id = nextPinId++
                pins[id] = TrackedPin(id, det.box)
            }
        }

        // ── Check fallen detections → mark matching pins as fallen ──
        for (det in fallenDets) {
            // Find the standing pin closest to this fallen detection
            val closest = pins.values
                .filter { it.state == PinState.STANDING }
                .minByOrNull { centerDist(det.box, it.box) }

            if (closest != null && centerDist(det.box, closest.box) < 150f) {
                markFallen(closest.id, currentTime)
            }
        }

        return pins.values.toList()
    }

    fun markFallen(pinId: Int, time: Long) {
        val pin = pins[pinId] ?: return
        if (pin.state == PinState.FALLEN) return
        hitCounter++
        pin.state    = PinState.FALLEN
        pin.hitOrder = hitCounter
        pin.hitTime  = time
        hitPins.add(pin)
    }

    fun getPins()   = pins.values.toList()
    fun getScore()  = hitCounter

    private fun centerDist(a: RectF, b: RectF): Float {
        val dx = (a.left+a.right)/2 - (b.left+b.right)/2
        val dy = (a.top+a.bottom)/2 - (b.top+b.bottom)/2
        return sqrt(dx*dx + dy*dy)
    }

    fun reset() {
        pins.clear(); nextPinId = 1
        hitCounter = 0; hitPins.clear()
    }
}

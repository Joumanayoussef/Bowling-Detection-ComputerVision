package com.cv.bowling

import android.graphics.RectF
import kotlin.math.sqrt

/**
 * ScoreManager
 * ────────────
 * Detects car-pin collisions and triggers scoring.
 *
 * Two detection methods:
 *  1. Class transition: YOLO detects standing_pin → fallen_pin
 *     This is the primary method — most reliable.
 *  2. Proximity: car center within PROXIMITY px of pin center
 *     Fallback for when YOLO misses the transition frame.
 *
 * A pin is only counted once (scored set prevents double-counting).
 */
class ScoreManager {

    private val scored    = mutableSetOf<Int>()  // pin IDs already scored
    private val PROXIMITY = 80f                  // px in original image coords

    fun checkCollisions(
        carBox     : RectF?,
        pins       : List<TrackedPin>,
        pinTracker : PinTracker,
        time       : Long
    ): Int {
        if (carBox == null) return pinTracker.getScore()

        val carCx = (carBox.left + carBox.right) / 2f
        val carCy = (carBox.top  + carBox.bottom) / 2f

        for (pin in pins) {
            if (pin.id in scored) continue
            if (pin.state == PinState.FALLEN) {
                // Already marked fallen by PinTracker (class transition)
                scored.add(pin.id)
                continue
            }

            // Proximity check: car close enough to pin?
            val pinCx = (pin.box.left + pin.box.right) / 2f
            val pinCy = (pin.box.top  + pin.box.bottom) / 2f
            val dist  = sqrt((carCx-pinCx)*(carCx-pinCx) + (carCy-pinCy)*(carCy-pinCy))

            if (dist < PROXIMITY) {
                scored.add(pin.id)
                pinTracker.markFallen(pin.id, time)
            }
        }

        return pinTracker.getScore()
    }

    fun reset() = scored.clear()
}

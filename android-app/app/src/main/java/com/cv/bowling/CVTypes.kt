package com.cv.bowling

import android.graphics.PointF
import android.graphics.RectF

// ── Class indices matching labels.txt order ──
// 0 = ball, 1 = car, 2 = fallen pin, 3 = standing pin
object Classes {
    const val BALL         = 0
    const val CAR          = 1
    const val FALLEN_PIN   = 2
    const val STANDING_PIN = 3
}

data class Detection(
    val classId    : Int,
    val label      : String,
    val confidence : Float,
    val box        : RectF      // in original image coords
)

enum class PinState { STANDING, FALLEN }

data class TrackedPin(
    val id       : Int,
    var box      : RectF,
    var state    : PinState = PinState.STANDING,
    var hitOrder : Int = 0,     // 0 = not hit, 1 = first hit, etc.
    var hitTime  : Long = 0L
)

data class FrameResult(
    val detections : List<Detection>,
    val trackedPins: List<TrackedPin>,
    val carBox     : RectF?,
    val carPath    : List<PointF>,
    val score      : Int,
    val imageW     : Int,
    val imageH     : Int
)

package com.cv.bowling

import androidx.lifecycle.LiveData
import androidx.lifecycle.MutableLiveData
import androidx.lifecycle.ViewModel

class GameViewModel : ViewModel() {
    private val _result = MutableLiveData<FrameResult>()
    val result: LiveData<FrameResult> = _result

    private val _score = MutableLiveData(0)
    val score: LiveData<Int> = _score

    private val _status = MutableLiveData("Press START — point at the scene")
    val status: LiveData<String> = _status

    fun updateResult(r: FrameResult) {
        _result.postValue(r)
        _score.postValue(r.score)
    }

    fun updateStatus(s: String) { _status.postValue(s) }
}

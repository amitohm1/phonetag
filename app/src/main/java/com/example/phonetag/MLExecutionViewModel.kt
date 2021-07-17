package com.example.phonetag

import android.graphics.Bitmap
import androidx.lifecycle.LiveData
import androidx.lifecycle.MutableLiveData
import androidx.lifecycle.ViewModel
import android.util.Log


private const val TAG = "MLExecutionViewModel"

class MLExecutionViewModel : ViewModel() {

    private val _resultingBitmap = MutableLiveData<ModelExecutionResult>()

    val resultingBitmap: LiveData<ModelExecutionResult>
        get() = _resultingBitmap


    // the execution of the model has to be on the same thread where the interpreter
    // was created
    fun onApplyModel(
        imageSegmentationModel: ImageSegmentationModelExecutor?,
        contentImage: Bitmap
    ) {

            try {
                val result = imageSegmentationModel?.execute(contentImage)
                _resultingBitmap.postValue(result)
            } catch (e: Exception) {
                Log.e(TAG, "Fail to execute ImageSegmentationModelExecutor: ${e.message}")
                _resultingBitmap.postValue(null)
            }

    }
}

/*
 * Copyright 2019 Google LLC
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     https://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
package com.example.phonetag

import android.content.Context
import android.graphics.Bitmap
import android.graphics.Color
import android.os.SystemClock
import androidx.core.graphics.ColorUtils
import android.util.Log
import android.widget.ImageView
import kotlinx.coroutines.GlobalScope
import kotlinx.coroutines.launch
import java.io.FileInputStream
import java.io.IOException
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.MappedByteBuffer
import java.nio.channels.FileChannel
import kotlin.random.Random
import org.tensorflow.lite.Interpreter
import org.tensorflow.lite.gpu.CompatibilityList
import org.tensorflow.lite.gpu.GpuDelegate
import androidx.lifecycle.lifecycleScope
import kotlinx.coroutines.runBlocking

/**
 * Class responsible to run the Image Segmentation model.
 * more information about the DeepLab model being used can
 * be found here:
 * https://ai.googleblog.com/2018/03/semantic-image-segmentation-with.html
 * https://www.tensorflow.org/lite/models/segmentation/overview
 * https://github.com/tensorflow/models/tree/master/research/deeplab
 *
 * Label names: 'background', 'aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus',
 * 'car', 'cat', 'chair', 'cow', 'diningtable', 'dog', 'horse', 'motorbike',
 * 'person', 'pottedplant', 'sheep', 'sofa', 'train', 'tv'
 */
class ImageSegmentationModelExecutor(
  context: Context,
  private var useGPU: Boolean = false
) {
  private var gpuDelegate: GpuDelegate? = null

  private val segmentationMasks: ByteBuffer
  private val interpreter: Interpreter

  private var fullTimeExecutionTime = 0L
  private var preprocessTime = 0L
  private var imageSegmentationTime = 0L
  private var maskFlatteningTime = 0L

  private var numberThreads = 4

  init {

    interpreter = getInterpreter(context, imageSegmentationModel, useGPU)
    segmentationMasks = ByteBuffer.allocateDirect(1 * imageSize * imageSize * NUM_CLASSES * 4)
    segmentationMasks.order(ByteOrder.nativeOrder())
  }

  fun execute(data: Bitmap): ModelExecutionResult {
    try {
      fullTimeExecutionTime = SystemClock.uptimeMillis()

      val scaledBitmap =
        ImageUtils.scaleBitmapAndKeepRatio(
          data,
          imageSize, imageSize
        )

      val contentArray =
        ImageUtils.bitmapToByteBuffer(
          scaledBitmap,
          imageSize,
          imageSize,
          IMAGE_MEAN,
          IMAGE_STD
        )


      interpreter.run(contentArray, segmentationMasks)


      var maskOnly = scaledBitmap

      runBlocking {
           launch { maskOnly = doWork(segmentationMasks, imageSize, segmentColors) }
      }

      fullTimeExecutionTime = SystemClock.uptimeMillis() - fullTimeExecutionTime
      Log.d(TAG, "Total time execution $fullTimeExecutionTime")


      return ModelExecutionResult(
        //maskImageApplied,
        //scaledBitmap,
        maskOnly,
        formatExecutionLog(),
        //itemsFound
      )
    } catch (e: Exception) {
      val exceptionLog = "something went wrong: ${e.message}"
      Log.d(TAG, exceptionLog)

      val emptyBitmap =
        ImageUtils.createEmptyBitmap(
          imageSize,
          imageSize
        )
      return ModelExecutionResult(
        emptyBitmap,
        exceptionLog
      )
    }
  }

  // base: https://github.com/tensorflow/tensorflow/blob/master/tensorflow/lite/java/demo/app/src/main/java/com/example/android/tflitecamerademo/ImageClassifier.java
  @Throws(IOException::class)
  private fun loadModelFile(context: Context, modelFile: String): MappedByteBuffer {
    val fileDescriptor = context.assets.openFd(modelFile)
    val inputStream = FileInputStream(fileDescriptor.fileDescriptor)
    val fileChannel = inputStream.channel
    val startOffset = fileDescriptor.startOffset
    val declaredLength = fileDescriptor.declaredLength
    val retFile = fileChannel.map(FileChannel.MapMode.READ_ONLY, startOffset, declaredLength)
    fileDescriptor.close()
    return retFile
  }

  @Throws(IOException::class)
  private fun getInterpreter(
    context: Context,
    modelName: String,
    useGpu: Boolean = false
  ): Interpreter {

    val compatList = CompatibilityList()

    val options = Interpreter.Options().apply{
      if(compatList.isDelegateSupportedOnThisDevice){
        // if the device has a supported GPU, add the GPU delegate
        val delegateOptions = compatList.bestOptionsForThisDevice
        this.addDelegate(GpuDelegate(delegateOptions))
      } else {
        // if the GPU is not supported, run on 4 threads
        this.setNumThreads(4)
      }
    }

    return Interpreter(loadModelFile(context, modelName), options)
  }

  private fun formatExecutionLog(): String {
    val sb = StringBuilder()
    sb.append("Input Image Size: $imageSize x $imageSize\n")
    sb.append("GPU enabled: $useGPU\n")
    sb.append("Number of threads: $numberThreads\n")
    sb.append("Pre-process execution time: $preprocessTime ms\n")
    sb.append("Model execution time: $imageSegmentationTime ms\n")
    sb.append("Mask flatten time: $maskFlatteningTime ms\n")
    sb.append("Full execution time: $fullTimeExecutionTime ms\n")
    return sb.toString()
  }

  fun close() {
    interpreter.close()
    if (gpuDelegate != null) {
      gpuDelegate!!.close()
    }
  }


  suspend fun doWork(
    inputBuffer: ByteBuffer,
    imageWidth: Int,
    colors: IntArray
  ):  Bitmap {
    /*
    Function to parellelize getting the mask bit map, we use coroutine to do this and divide
    the output bitmap into 4 quadrants as follows:
    Q1 Q2
    Q3 Q4

     */


    val imageHeight = imageWidth

    val conf = Bitmap.Config.ARGB_8888
    val maskBitmap = Bitmap.createBitmap(imageWidth, imageHeight, conf)

    val mSegmentBits = Array(imageWidth) { IntArray(imageHeight) }
    inputBuffer.rewind()

    val j1 = GlobalScope.launch { // Q1

      for (y in 0 until (imageWidth / 2)) {
        for (x in 0 until (imageWidth / 2)) {
          var maxVal = 0f
          mSegmentBits[x][y] = 0
          for (c in 0 until NUM_CLASSES) {
            val value = inputBuffer
              .getFloat((y * imageWidth * NUM_CLASSES + x * NUM_CLASSES + c) * 4)
//            val value = float_array[y * imageWidth * NUM_CLASSES + x * NUM_CLASSES + c]
            if (c == 0 || value > maxVal) {
              maxVal = value
              mSegmentBits[x][y] = c
            }
          }

          if (mSegmentBits[x][y] == 15) {
            maskBitmap.setPixel(x, y, colors[mSegmentBits[x][y]])
          }

        }

    }

    }

    val j2 = GlobalScope.launch { // Q2

      for (y in (imageWidth / 2) until imageWidth) {
        for (x in 0 until (imageWidth / 2)) {
          var maxVal = 0f
          mSegmentBits[x][y] = 0
          for (c in 0 until NUM_CLASSES) {
            val value = inputBuffer
              .getFloat((y * imageWidth * NUM_CLASSES + x * NUM_CLASSES + c) * 4)
//            val value = float_array[y * imageWidth * NUM_CLASSES + x * NUM_CLASSES + c]
            if (c == 0 || value > maxVal) {
              maxVal = value
              mSegmentBits[x][y] = c
            }
          }

          if (mSegmentBits[x][y] == 15) {
            maskBitmap.setPixel(x, y, colors[mSegmentBits[x][y]])
          }

        }

      }

    }

    val j3 = GlobalScope.launch { // Q3

      for (y in 0 until (imageWidth / 2)) {
        for (x in (imageWidth / 2) until imageWidth) {
          var maxVal = 0f
          mSegmentBits[x][y] = 0
          for (c in 0 until NUM_CLASSES) {
            val value = inputBuffer
              .getFloat((y * imageWidth * NUM_CLASSES + x * NUM_CLASSES + c) * 4)
//            val value = float_array[y * imageWidth * NUM_CLASSES + x * NUM_CLASSES + c]
            if (c == 0 || value > maxVal) {
              maxVal = value
              mSegmentBits[x][y] = c
            }
          }

          if (mSegmentBits[x][y] == 15) {
            maskBitmap.setPixel(x, y, colors[mSegmentBits[x][y]])
          }

        }

      }

    }


    val j4 = GlobalScope.launch { // Q4

      for (y in (imageWidth / 2) until imageWidth) {
        for (x in (imageWidth / 2) until imageWidth) {
          var maxVal = 0f
          mSegmentBits[x][y] = 0
          for (c in 0 until NUM_CLASSES) {
            val value = inputBuffer
              .getFloat((y * imageWidth * NUM_CLASSES + x * NUM_CLASSES + c) * 4)
//            val value = float_array[y * imageWidth * NUM_CLASSES + x * NUM_CLASSES + c]
            if (c == 0 || value > maxVal) {
              maxVal = value
              mSegmentBits[x][y] = c
            }
          }

          if (mSegmentBits[x][y] == 15) {
            maskBitmap.setPixel(x, y, colors[mSegmentBits[x][y]])
          }

        }

      }

    }

    j1.join()
    j2.join()
    j3.join()
    j4.join()

    return maskBitmap

  }



  companion object {

    public const val TAG = "SegmentationInterpreter"
    private const val imageSegmentationModel = "deeplabv3_257_mv_gpu.tflite"
    private const val imageSize = 257
    const val NUM_CLASSES = 21
    private const val IMAGE_MEAN = 127.5f
    private const val IMAGE_STD = 127.5f

    val segmentColors = IntArray(NUM_CLASSES)
    val labelsArrays = arrayOf(
      "background", "aeroplane", "bicycle", "bird", "boat", "bottle", "bus",
      "car", "cat", "chair", "cow", "dining table", "dog", "horse", "motorbike",
      "person", "potted plant", "sheep", "sofa", "train", "tv"
    )

    init {

      val random = Random(System.currentTimeMillis())
      segmentColors[0] = Color.TRANSPARENT
      for (i in 1 until NUM_CLASSES) {
        segmentColors[i] = Color.argb(
          (200),
          getRandomRGBInt(
            random
          ),
          getRandomRGBInt(
            random
          ),
          getRandomRGBInt(
            random
          )
        )
      }
      segmentColors[15] = Color.argb(200, 255, 0, 0)

    }

    private fun getRandomRGBInt(random: Random) = (255 * random.nextFloat()).toInt()
  }
}
package com.example.phonetag

import android.Manifest
import android.annotation.SuppressLint
import android.content.pm.PackageManager
import android.graphics.Bitmap
import android.graphics.Matrix
import android.os.Bundle
import android.os.SystemClock
import android.util.Log
import android.widget.ImageView
import android.widget.Toast
import androidx.appcompat.app.AppCompatActivity
import androidx.camera.core.*
import androidx.camera.lifecycle.ProcessCameraProvider
import androidx.core.app.ActivityCompat
import androidx.core.content.ContextCompat
import androidx.lifecycle.Observer
import androidx.lifecycle.ViewModelProvider
import kotlinx.android.synthetic.main.activity_main.*
import org.tensorflow.lite.task.vision.segmenter.Segmentation
import java.io.File
import java.util.concurrent.ExecutorService
import java.util.concurrent.Executors


typealias PlayerListener = (empty: List<Segmentation>) -> Unit

class MainActivity : AppCompatActivity() {
    private var imageCapture: ImageCapture? = null

    private lateinit var outputDirectory: File
    private lateinit var cameraExecutor: ExecutorService
    
    private var imageSegmentationModel: ImageSegmentationModelExecutor? = null

    private inner class PlayerAnalyzer(private val listener: PlayerListener) : ImageAnalysis.Analyzer {


        val converter = YuvToRgbConverter(applicationContext)

        @SuppressLint("UnsafeOptInUsageError")
        override fun analyze(image: ImageProxy) {

            image.image?.let {

                val bm1 =
                    Bitmap.createBitmap(
                        it.width,
                        it.height,
                        Bitmap.Config.ARGB_8888
                    )

                converter.yuvToRgb(it, bm1)

                val matrix = Matrix()
                matrix.postRotate(90F)

                val bm2 = Bitmap.createBitmap(bm1, 0, 0, bm1.width, bm1.height, matrix, false)

                val result = imageSegmentationModel?.execute(bm2)


                if (result != null) {
                    updateUIWithResults(result)
                }

                listener(emptyList())

                image.close()
            }
        }

    }



    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)

        // Request camera permissions
        if (allPermissionsGranted()) {
            startCamera()
        } else {
            ActivityCompat.requestPermissions(
                this, REQUIRED_PERMISSIONS, REQUEST_CODE_PERMISSIONS)
        }

        outputDirectory = getOutputDirectory()

        cameraExecutor = Executors.newSingleThreadExecutor()

        createModelExecutor()
    }

    private fun updateUIWithResults(modelExecutionResult: ModelExecutionResult) {
        val view = findViewById<ImageView>(R.id.model)
        val bitmapMask = modelExecutionResult.bitmapMaskOnly

        val w = view.width
        val h = view.height

        val scaledBitmapMask = ImageUtils.scaleBitmapAndKeepRatio(bitmapMask, h, w)

        runOnUiThread {
            view.setImageBitmap(scaledBitmapMask)
        }

    }


    private fun createModelExecutor() {
        if (imageSegmentationModel != null) {
            imageSegmentationModel!!.close()
            imageSegmentationModel = null
        }
        try {
            imageSegmentationModel = ImageSegmentationModelExecutor(this, true)
        } catch (e: Exception) {
            Log.e(TAG, "Fail to create ImageSegmentationModelExecutor: ${e.message}")
        }
    }


    @SuppressLint("RestrictedApi")
    private fun startCamera() {
        val cameraProviderFuture = ProcessCameraProvider.getInstance(this)
        cameraProviderFuture.addListener(Runnable {
            // Used to bind the lifecycle of cameras to the lifecycle owner
            val cameraProvider: ProcessCameraProvider = cameraProviderFuture.get()

            // Preview
            val preview = Preview.Builder()
                .build()
                .also {
                    it.setSurfaceProvider(viewFinder.surfaceProvider)
                }

            imageCapture = ImageCapture.Builder()
                .build()


            val imageAnalyzer = ImageAnalysis.Builder()
                .build()
                .also {
                    it.setAnalyzer(cameraExecutor, PlayerAnalyzer {
                    })
                }

            // Select back camera as a default
            val cameraSelector = CameraSelector.DEFAULT_BACK_CAMERA

            try {
                // Unbind use cases before rebinding
                cameraProvider.unbindAll()

                // Bind use cases to camera
                cameraProvider.bindToLifecycle(
                    this, cameraSelector, preview, imageCapture, imageAnalyzer)

            } catch(exc: Exception) {
                Log.e(TAG, "Use case binding failed", exc)
            }

        }, ContextCompat.getMainExecutor(this))
    }

    private fun allPermissionsGranted() = REQUIRED_PERMISSIONS.all {
        ContextCompat.checkSelfPermission(
            baseContext, it) == PackageManager.PERMISSION_GRANTED
    }

    private fun getOutputDirectory(): File {
        val mediaDir = externalMediaDirs.firstOrNull()?.let {
            File(it, resources.getString(R.string.app_name)).apply { mkdirs() } }
        return if (mediaDir != null && mediaDir.exists())
            mediaDir else filesDir
    }

    override fun onDestroy() {
        super.onDestroy()
        cameraExecutor.shutdown()
    }

    companion object {
        private const val TAG = "CameraXBasic"
        private const val REQUEST_CODE_PERMISSIONS = 10
        private val REQUIRED_PERMISSIONS = arrayOf(
            Manifest.permission.CAMERA, Manifest.permission.READ_EXTERNAL_STORAGE,
            Manifest.permission.WRITE_EXTERNAL_STORAGE, Manifest.permission.INTERNET
        )
    }

    @SuppressLint("MissingSuperCall")
    override fun onRequestPermissionsResult(
        requestCode: Int, permissions: Array<String>, grantResults:
        IntArray) {
        if (requestCode == REQUEST_CODE_PERMISSIONS) {
            if (allPermissionsGranted()) {
                startCamera()
            } else {
                Toast.makeText(this,
                    "Permissions not granted by the user.",
                    Toast.LENGTH_SHORT).show()
                finish()
            }
        }
    }


}
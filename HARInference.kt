package com.twocity.handedness.core

import android.content.Context
import android.hardware.Sensor
import android.hardware.SensorEvent
import android.hardware.SensorEventListener
import android.hardware.SensorManager
import android.os.Handler
import android.os.Looper
import android.util.Log
import org.tensorflow.lite.Interpreter
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.util.concurrent.locks.ReentrantLock
import kotlin.concurrent.withLock
import org.json.JSONObject

class HARInference(
    private val context: Context,
    private val callback: HARCallback
) {
    private var interpreter: Interpreter? = null
    private var preprocessParams: PreprocessParams? = null
    private val sensorManager = context.getSystemService(Context.SENSOR_SERVICE) as SensorManager
    private val windowSize = 128 // 从训练代码 DEFAULT_CONFIG 中的 segment_size
    private val numChannels = 6  // 从训练代码 DEFAULT_CONFIG 中的 num_channels

    private val accData = ArrayList<FloatArray>()
    private val gyroData = ArrayList<FloatArray>()
    private val lock = ReentrantLock()

    private val sensorListener = object : SensorEventListener {
        override fun onSensorChanged(event: SensorEvent) {
            lock.withLock {
                when (event.sensor.type) {
                    Sensor.TYPE_ACCELEROMETER -> {
                        accData.add(event.values.clone())
                    }
                    Sensor.TYPE_GYROSCOPE -> {
                        gyroData.add(event.values.clone())
                    }
                }
                
                // Only process when we have enough data from both sensors
                if (accData.size >= windowSize && gyroData.size >= windowSize) {
                    processSensorData()
                }
            }
        }

        override fun onAccuracyChanged(sensor: Sensor?, accuracy: Int) {}
    }

    data class PreprocessParams(
        val meanAcc: Float,
        val stdAcc: Float,
        val meanGyro: Float,
        val stdGyro: Float
    )

    interface HARCallback {
        fun onActivityRecognized(activity: String, confidence: Float)
        fun onError(error: String)
    }

    fun start() {
        try {
            loadModel()
            loadPreprocessParams()
            initializeSensors()
        } catch (e: Exception) {
            Log.d("HARInference", "Initialization failed: ${e.message}", e)
            callback.onError("Initialization failed: ${e.message}")
        }
    }

    private fun loadModel() {
        try {
            context.assets.open("har/mobile_hart_motionsense.tflite").use { fileInputStream ->
                val modelBuffer = ByteBuffer.allocateDirect(fileInputStream.available())
                    .order(ByteOrder.nativeOrder())

                // Read bytes into the ByteBuffer
                val bytes = ByteArray(fileInputStream.available())
                fileInputStream.read(bytes)
                modelBuffer.put(bytes)
                modelBuffer.rewind()  // Reset position to beginning

                val options = Interpreter.Options()
                interpreter = Interpreter(modelBuffer, options)
            }
        } catch (e: Exception) {
            throw RuntimeException("Error loading model: ${e.message}")
        }
    }

    private fun loadPreprocessParams() {
        try {
            context.assets.open("har/preprocessing_params_motionsense.json").use { inputStream ->
                val jsonString = inputStream.bufferedReader().use { it.readText() }
                val jsonObject = JSONObject(jsonString)

                preprocessParams = PreprocessParams(
                    meanAcc = jsonObject.getDouble("mean_acc").toFloat(),
                    stdAcc = jsonObject.getDouble("std_acc").toFloat(),
                    meanGyro = jsonObject.getDouble("mean_gyro").toFloat(),
                    stdGyro = jsonObject.getDouble("std_gyro").toFloat()
                )
            }
        } catch (e: Exception) {
            throw RuntimeException("Error loading preprocessing parameters: ${e.message}", e.cause)
        }
    }

    private fun initializeSensors() {
        val accelerometer = sensorManager.getDefaultSensor(Sensor.TYPE_ACCELEROMETER)
        val gyroscope = sensorManager.getDefaultSensor(Sensor.TYPE_GYROSCOPE)

        sensorManager.registerListener(
            sensorListener,
            accelerometer,
            SensorManager.SENSOR_DELAY_GAME
        )
        sensorManager.registerListener(sensorListener, gyroscope, SensorManager.SENSOR_DELAY_GAME)
    }

    private fun processSensorData() {
        try {
            val data = preprocessSensorData()
            val result = runInference(data)
            val (activity, confidence) = interpretResult(result)
            callback.onActivityRecognized(activity, confidence)

            // Clear old data while keeping some overlap
            lock.withLock {
                accData.subList(0, windowSize / 2).clear()
                gyroData.subList(0, windowSize / 2).clear()
            }
        } catch (e: Exception) {
            Log.d("HARInference", "Processing failed: ${e.message}", e)
            callback.onError("Processing failed: ${e.message}")
        }
    }

    private fun preprocessSensorData(): FloatArray {
        val params = preprocessParams ?: throw RuntimeException("Preprocessing parameters not loaded")
        val processedData = FloatArray(windowSize * numChannels)

        for (i in 0 until windowSize) {
            val accValues = accData[i]
            val gyroValues = gyroData[i]

            // Process accelerometer data (first 3 channels)
            for (j in 0..2) {
                processedData[i * numChannels + j] = (accValues[j] - params.meanAcc) / params.stdAcc
            }
            
            // Process gyroscope data (last 3 channels)
            for (j in 0..2) {
                processedData[i * numChannels + j + 3] = (gyroValues[j] - params.meanGyro) / params.stdGyro
            }
        }
        return processedData
    }

    private fun runInference(input: FloatArray): FloatArray {
        val inputBuffer = ByteBuffer.allocateDirect(windowSize * numChannels * 4)
            .order(ByteOrder.nativeOrder())
        input.forEach { inputBuffer.putFloat(it) }

        val outputBuffer = ByteBuffer.allocateDirect(6 * 4)  // 6 classes
            .order(ByteOrder.nativeOrder())

        interpreter?.run(inputBuffer, outputBuffer)

        val output = FloatArray(6)
        outputBuffer.rewind()
        for (i in output.indices) {
            output[i] = outputBuffer.float
        }
        return output
    }

    private fun interpretResult(output: FloatArray): Pair<String, Float> {
        val activities =
            arrayOf("Downstairs", "Upstairs", "Sitting", "Standing", "Walking", "Jogging")
        val maxIndex = output.indices.maxByOrNull { output[it] } ?: 0
        return Pair(activities[maxIndex], output[maxIndex])
    }

    fun release() {
        sensorManager.unregisterListener(sensorListener)
        interpreter?.close()
        lock.withLock {
            accData.clear()
            gyroData.clear()
        }
    }
}

package com.twocity.handedness.core

import android.annotation.SuppressLint
import android.content.Context
import android.hardware.*
import android.util.Log
import org.tensorflow.lite.Interpreter
import org.tensorflow.lite.support.common.FileUtil
import org.json.JSONObject
import java.io.IOException
import java.nio.MappedByteBuffer
import kotlin.math.max
import kotlin.math.min
import java.util.ArrayDeque
import kotlin.math.abs

/**
 * HARInference:
 * - 分别采集加速度计、陀螺仪数据到各自缓冲
 * - 分别对加速度计/陀螺仪按指定频率重采样
 * - 在同一时间轴上合并得到 [accX,accY,accZ, gyroX,gyroY,gyroZ]
 * - 依据训练阶段输出的 JSON 做数据标准化
 * - 利用 TFLite 模型进行推断
 *
 * 支持多个数据集: RealWorld, HHAR, UCI, SHL, MotionSense, COMBINED，默认 "UCI"
 */
class HARInference(
    private val context: Context,
    private val callback: HARCallback,
    private val dataSetName: String = "UCI",      // 默认数据集
    private val desiredSamplingRate: Int = 50,    // 目标采样率(Hz)
    private val windowSize: Int = 128,            // 窗口大小
    private val resampleConfig: ResampleConfig = ResampleConfig.default(windowSize)  // 重采样配置
) : SensorEventListener {

    companion object {
        private const val TAG = "HARInference"

        // =========================== 活动标签映射 ===========================
        private val LABELS_UCI =
            arrayOf("Walking", "Upstairs", "Downstairs", "Sitting", "Standing", "Lying")
        private val LABELS_HHAR =
            arrayOf("Sitting", "Standing", "Walking", "Upstairs", "Downstairs", "Biking")
        private val LABELS_REALWORLD = arrayOf(
            "Downstairs","Upstairs","Jumping","Lying","Running","Sitting","Standing","Walking"
        )
        private val LABELS_MOTIONSENSE =
            arrayOf("Downstairs", "Upstairs", "Sitting", "Standing", "Walking", "Jogging")
        private val LABELS_SHL =
            arrayOf("Standing","Walking","Runing","Biking","Car","Bus","Train","Subway")
        // COMBINED: 仅示例
        private val LABELS_COMBINED = arrayOf(
            "Walk", "Upstair","Downstair","Sit","Stand","Lay","Jump",
            "Run","Bike","Car","Bus","Train","Subway"
        )

        fun getActivityLabelsByDataSet(dsName: String): Array<String> {
            return when (dsName.trim().lowercase()) {
                "hhar" -> LABELS_HHAR
                "realworld" -> LABELS_REALWORLD
                "motionsense" -> LABELS_MOTIONSENSE
                "shl" -> LABELS_SHL
                "combined" -> LABELS_COMBINED
                else -> LABELS_UCI // default
            }
        }

        // =========================== 文件名映射 ===========================
        fun getModelFileName(dsName: String): String {
            // 也可根据 dsName 返回不同 tflite
            return "har/${dsName.lowercase()}/MobileHART.tflite"
        }

        fun getPreprocessingParamsFileName(dsName: String): String {
            return "har/${dsName.lowercase()}/preprocessing_params.json"
        }
    }

    // 回调接口，输出活动识别结果或错误
    interface HARCallback {
        fun onActivityRecognized(activity: String, confidence: Float)
        fun onError(error: String)
    }

    // ================================== 传感器相关 ==================================
    private var sensorManager: SensorManager? = null
    private var accSensor: Sensor? = null
    private var gyrSensor: Sensor? = null

    // 分别维护加速度计、陀螺仪缓冲区
    private val accBuffer = mutableListOf<AccData>()
    private val gyroBuffer = mutableListOf<GyroData>()

    // 同步锁对象
    private val bufferLock = Object()

    // 数据结构：加速度计记录
    data class AccData(
        val timestamp: Long,
        val x: Float,
        val y: Float,
        val z: Float
    )
    // 数据结构：陀螺仪记录
    data class GyroData(
        val timestamp: Long,
        val x: Float,
        val y: Float,
        val z: Float
    )
    // 合并后结构
    data class SensorDataAll(
        val timestamp: Long,
        val accX: Float,
        val accY: Float,
        val accZ: Float,
        val gyroX: Float,
        val gyroY: Float,
        val gyroZ: Float
    )

    // ================================== TFLite 模型 & 预处理 ==================================
    private var tfliteInterpreter: Interpreter? = null

    // 标准化参数: 用于 (x - mean) / std
    private var accMeanX: Float = 0f
    private var accStdX: Float = 1f
    private var accMeanY: Float = 0f
    private var accStdY: Float = 1f
    private var accMeanZ: Float = 0f
    private var accStdZ: Float = 1f
    private var gyroMeanX: Float = 0f
    private var gyroStdX: Float = 1f
    private var gyroMeanY: Float = 0f
    private var gyroStdY: Float = 1f
    private var gyroMeanZ: Float = 0f
    private var gyroStdZ: Float = 1f

    // 时间戳参考
    private var lastAccTimestampNs: Long = -1L
    private var lastGyroTimestampNs: Long = -1L

    private var activityLabels: Array<String> = getActivityLabelsByDataSet(dataSetName)

    // 添加成员变量
    private val recentPredictions = ArrayDeque<Pair<String, Float>>(resampleConfig.smoothWindowSize)
    private var lastInferenceTime = 0L

    // 内部重采样配置类
    class ResampleConfig private constructor(
        val minSamples: Int,           // 最少需要的样本数
        val bufferOverlap: Float,      // 缓冲区重叠率
        val smoothWindowSize: Int,     // 平滑窗口大小
        val confidenceThreshold: Float, // 置信度阈值
        val minInferenceInterval: Long // 最小推理间隔(毫秒)
    ) {
        companion object {
            // 默认配置
            fun default(windowSize: Int) = ResampleConfig(
                minSamples = windowSize,
                bufferOverlap = 0.5f,
                smoothWindowSize = 5,
                confidenceThreshold = 0.6f,
                minInferenceInterval = 1000L
            )

            // 高性能配置 - 更快的响应
            fun highPerformance(windowSize: Int) = ResampleConfig(
                minSamples = windowSize,
                bufferOverlap = 0.25f,
                smoothWindowSize = 3,
                confidenceThreshold = 0.7f,
                minInferenceInterval = 500L
            )

            // 高精度配置 - 更稳定的结果
            fun highAccuracy(windowSize: Int) = ResampleConfig(
                minSamples = windowSize,
                bufferOverlap = 0.75f,
                smoothWindowSize = 7,
                confidenceThreshold = 0.9f,
                minInferenceInterval = 2000L
            )
        }
    }

    /**
     * 启动：加载模型 + 注册传感器
     */
    fun start() {
        try {
            sensorManager = context.getSystemService(Context.SENSOR_SERVICE) as SensorManager
            accSensor = sensorManager?.getDefaultSensor(Sensor.TYPE_ACCELEROMETER)
            gyrSensor = sensorManager?.getDefaultSensor(Sensor.TYPE_GYROSCOPE)

            if (accSensor == null || gyrSensor == null) {
                callback.onError("Accelerometer or Gyroscope not found.")
                return
            }

            loadTfliteModel()
            loadPreprocessingParams()

            // 注册监听器: 让系统尽可能快发送(FASTEST)，后续再用我们自己的重采样
            sensorManager?.registerListener(this, accSensor, SensorManager.SENSOR_DELAY_FASTEST)
            sensorManager?.registerListener(this, gyrSensor, SensorManager.SENSOR_DELAY_FASTEST)

            Log.i(TAG, "HARInference started. dataSet=$dataSetName, freq=$desiredSamplingRate, window=$windowSize")
        } catch (ex: Exception) {
            Log.e(TAG, "start() failed: ${ex.message}", ex)
            callback.onError("start() failed: ${ex.message}")
        }
    }

    /**
     * 停止：反注册、关闭Interpreter
     */
    fun release() {
        try {
            sensorManager?.unregisterListener(this)
            sensorManager = null
            tfliteInterpreter?.close()
            tfliteInterpreter = null

            synchronized(bufferLock) {
                accBuffer.clear()
                gyroBuffer.clear()
            }
            Log.i(TAG, "HARInference released.")
        } catch (ex: Exception) {
            Log.e(TAG, "release() failed: ${ex.message}", ex)
            callback.onError("release() failed: ${ex.message}")
        }
    }

    // ================================== SensorEventListener ==================================
    override fun onAccuracyChanged(sensor: Sensor?, accuracy: Int) {
        // ignore
    }

    @SuppressLint("DefaultLocale")
    override fun onSensorChanged(event: SensorEvent?) {
        if (event == null) return

        val sensorType = event.sensor?.type ?: return
        val timestampNs = event.timestamp
        val x = event.values.getOrNull(0) ?: 0f
        val y = event.values.getOrNull(1) ?: 0f
        val z = event.values.getOrNull(2) ?: 0f

        synchronized(bufferLock) {
            when (sensorType) {
                Sensor.TYPE_ACCELEROMETER -> {
                    if (timestampNs <= lastAccTimestampNs) {
                        Log.w(TAG, "Received out-of-order acc data: $timestampNs <= $lastAccTimestampNs")
                        return@synchronized
                    }
                    lastAccTimestampNs = timestampNs
                    accBuffer.add(AccData(timestampNs, x, y, z))
                }
                Sensor.TYPE_GYROSCOPE -> {
                    if (timestampNs <= lastGyroTimestampNs) {
                        Log.w(TAG, "Received out-of-order gyro data: $timestampNs <= $lastGyroTimestampNs")
                        return@synchronized
                    }
                    lastGyroTimestampNs = timestampNs
                    gyroBuffer.add(GyroData(timestampNs, x, y, z))
                }
            }

            // 若缓冲足够大 -> 做一次推理
            if (accBuffer.size > windowSize * 2 && gyroBuffer.size > windowSize * 2) {
                processDataAndRunInference()
            }
        }
    }

    // ================================== 数据处理与推断 ==================================

    /**
     * 进行重采样 -> 合并 -> 窗口化 -> 预处理 -> TFLite 推断
     */
    private fun processDataAndRunInference() {
        val localAcc: List<AccData>
        val localGyro: List<GyroData>
        synchronized(bufferLock) {
            localAcc = accBuffer.toList()
            localGyro = gyroBuffer.toList()

            // 使用配置的重叠率
            val keepSize = (windowSize * resampleConfig.bufferOverlap).toInt()
            if (accBuffer.size > keepSize) {
                accBuffer.subList(0, accBuffer.size - keepSize).clear()
            }
            if (gyroBuffer.size > keepSize) {
                gyroBuffer.subList(0, gyroBuffer.size - keepSize).clear()
            }
        }

        // 使用配置的最小样本数
        if (localAcc.size < resampleConfig.minSamples || localGyro.size < resampleConfig.minSamples) {
            Log.d(TAG, "Not enough samples: acc=${localAcc.size}, gyro=${localGyro.size}, required=${resampleConfig.minSamples}")
            return
        }

        // 检查推理间隔
        val currentTime = System.currentTimeMillis()
        if (currentTime - lastInferenceTime < resampleConfig.minInferenceInterval) {
            return
        }

        try {
            // 1. 重采样
            val resampledAcc = reSampleAccData(localAcc, desiredSamplingRate)
            val resampledGyro = reSampleGyroData(localGyro, desiredSamplingRate)

            // 2. 检查重采样结果
            if (resampledAcc.size < windowSize || resampledGyro.size < windowSize) {
                Log.d(TAG, "Insufficient resampled data: acc=${resampledAcc.size}, gyro=${resampledGyro.size}, window=$windowSize")
                return
            }

            // 3. 合并到同一时间轴
            val combinedList = combineResampledData(resampledAcc, resampledGyro)
            if (combinedList.size < windowSize) {
                return
            }

            // 4. 取末端 windowSize 个点
            val finalSegment = combinedList.takeLast(windowSize)

            // 5. 执行推理
            val prediction = runModelInference(finalSegment)
                ?: return

            // 6. 平滑处理
            processPredictionWithSmoothing(prediction)
            lastInferenceTime = currentTime

        } catch (e: Exception) {
            Log.e(TAG, "Inference error:", e)
            callback.onError(e.message ?: "Unknown error")
        }
    }

    /**
     * 执行模型推理
     * @param finalSegment 已经处理好的传感器数据段
     * @return Pair<活动名称, 置信度> 或 null(如果结果无效)
     */
    private fun runModelInference(finalSegment: List<SensorDataAll>): Pair<String, Float>? {
        if (finalSegment.size != windowSize) {
            Log.w(TAG, "Invalid segment size: ${finalSegment.size}, required: $windowSize")
            return null
        }

        // 1. 准备输入数据 [1, windowSize, 6]
        val modelInput = Array(1) { Array(windowSize) { FloatArray(6) } }

        // 2. 填充并标准化数据
        for (i in finalSegment.indices) {
            val data = finalSegment[i]
            modelInput[0][i][0] = (data.accX - accMeanX) / accStdX
            modelInput[0][i][1] = (data.accY - accMeanY) / accStdY
            modelInput[0][i][2] = (data.accZ - accMeanZ) / accStdZ
            modelInput[0][i][3] = (data.gyroX - gyroMeanX) / gyroStdX
            modelInput[0][i][4] = (data.gyroY - gyroMeanY) / gyroStdY
            modelInput[0][i][5] = (data.gyroZ - gyroMeanZ) / gyroStdZ
        }

        // 3. 准备输出缓冲区
        val outputBuffer = Array(1) { FloatArray(activityLabels.size) }

        // 4. 执行推理
        tfliteInterpreter?.run(modelInput, outputBuffer)
            ?: return null

        // 5. 找出最高置信度及其索引
        var maxIdx = 0
        var maxVal = outputBuffer[0][0]
        for (i in outputBuffer[0].indices) {
            if (outputBuffer[0][i] > maxVal) {
                maxVal = outputBuffer[0][i]
                maxIdx = i
            }
        }

        // 6. 检查置信度阈值
        if (maxVal < resampleConfig.confidenceThreshold) {
            Log.d(TAG, "Confidence too low: $maxVal < ${resampleConfig.confidenceThreshold}")
            return null
        }

        val actName = activityLabels.getOrNull(maxIdx) ?: "Unknown"
        return Pair(actName, maxVal)
    }

    /**
     * 使用滑动窗口进行预测平滑
     * @param prediction Pair<活动名称, 置信度>
     */
    private fun processPredictionWithSmoothing(prediction: Pair<String, Float>) {
        synchronized(recentPredictions) {
            if (recentPredictions.size >= resampleConfig.smoothWindowSize) {
                recentPredictions.removeFirst()
            }
            recentPredictions.addLast(prediction)

            // 2. 按活动类型分组并计算平均置信度
            val predictions = recentPredictions.groupBy { it.first }
                .mapValues { (_, values) ->
                    values.map { it.second }.average()
                }

            // 3. 找出最频繁且置信度最高的活动
            // 评分 = 平均置信度 * (出现次数/窗口大小)
            val smoothedPrediction = predictions.maxByOrNull { (activity, confidence) ->
                confidence * recentPredictions.count { it.first == activity }.toFloat() / recentPredictions.size
            }

            // 4. 输出平滑后的结果
            smoothedPrediction?.let { (activity, confidence) ->
                callback.onActivityRecognized(activity, confidence.toFloat())
                Log.d(TAG, "Smoothed inference result: $activity (conf=$confidence)")
            }
        }
    }

    // ====================== 分别重采样加速度/陀螺仪 ======================

    private fun reSampleAccData(accList: List<AccData>, targetHz: Int): List<AccData> {
        if (accList.size < 2) {
            Log.w(TAG, "Too few acc samples: ${accList.size}")
            return emptyList()
        }

        val timeSpanNs = accList.last().timestamp - accList.first().timestamp
        val targetPoints = max(
            (timeSpanNs * targetHz / 1e9).toInt(),
            windowSize
        )

        Log.d(TAG, "Acc resampling: span=${timeSpanNs/1e6}ms, points=$targetPoints")

        val sortedAcc = accList.sortedBy { it.timestamp }
        val startTime = sortedAcc.first().timestamp
        val endTime = sortedAcc.last().timestamp

        // 确保不超过窗口大小
        val numPoints = minOf(targetPoints, windowSize)
        val dtNs = timeSpanNs / (numPoints - 1)  // 实际采样间隔

        val result = mutableListOf<AccData>()
        var t = startTime
        var idx = 0

        // 生成固定数量的点
        repeat(numPoints) {
            // 找到合适的插值区间
            while (idx < sortedAcc.size - 1 && sortedAcc[idx + 1].timestamp < t) {
                idx++
            }

            if (idx >= sortedAcc.size - 1) {
                // 使用最后两个点插值
                idx = sortedAcc.size - 2
            }

            val d1 = sortedAcc[idx]
            val d2 = sortedAcc[idx + 1]
            val ratio = (t - d1.timestamp).toDouble() / (d2.timestamp - d1.timestamp)

            // 线性插值
            val x = d1.x + (d2.x - d1.x) * ratio.toFloat()
            val y = d1.y + (d2.y - d1.y) * ratio.toFloat()
            val z = d1.z + (d2.z - d1.z) * ratio.toFloat()

            result.add(AccData(t, x, y, z))
            t += dtNs
        }

        return result
    }

    private fun reSampleGyroData(gyroList: List<GyroData>, targetHz: Int): List<GyroData> {
        if (gyroList.size < 2) {
            Log.w(TAG, "Too few gyro samples: ${gyroList.size}")
            return emptyList()
        }

        val timeSpanNs = gyroList.last().timestamp - gyroList.first().timestamp
        val targetPoints = max(
            (timeSpanNs * targetHz / 1e9).toInt(),
            windowSize
        )

        Log.d(TAG, "Gyro resampling: span=${timeSpanNs/1e6}ms, points=$targetPoints")

        val sortedGyro = gyroList.sortedBy { it.timestamp }
        val startTime = sortedGyro.first().timestamp
        val endTime = sortedGyro.last().timestamp

        // 确保不超过窗口大小
        val numPoints = minOf(targetPoints, windowSize)
        val dtNs = timeSpanNs / (numPoints - 1)  // 实际采样间隔

        val result = mutableListOf<GyroData>()
        var t = startTime
        var idx = 0

        // 生成固定数量的点
        repeat(numPoints) {
            // 找到合适的插值区间
            while (idx < sortedGyro.size - 1 && sortedGyro[idx + 1].timestamp < t) {
                idx++
            }

            if (idx >= sortedGyro.size - 1) {
                // 使用最后两个点插值
                idx = sortedGyro.size - 2
            }

            val d1 = sortedGyro[idx]
            val d2 = sortedGyro[idx + 1]
            val ratio = (t - d1.timestamp).toDouble() / (d2.timestamp - d1.timestamp)

            // 线性插值
            val x = d1.x + (d2.x - d1.x) * ratio.toFloat()
            val y = d1.y + (d2.y - d1.y) * ratio.toFloat()
            val z = d1.z + (d2.z - d1.z) * ratio.toFloat()

            result.add(GyroData(t, x, y, z))
            t += dtNs
        }

        return result
    }

    // ====================== 合并加速度和陀螺仪到同一时间轴 ======================
    private fun combineResampledData(
        accList: List<AccData>,
        gyroList: List<GyroData>
    ): List<SensorDataAll> {
        if (accList.isEmpty() || gyroList.isEmpty()) return emptyList()

        val merged = mutableListOf<SensorDataAll>()
        // 统一在 startTime~endTime 之间
        val startTime = max(accList.first().timestamp, gyroList.first().timestamp)
        val endTime   = min(accList.last().timestamp, gyroList.last().timestamp)

        // 双指针遍历
        var i = 0
        var j = 0
        while (i < accList.size && j < gyroList.size) {
            val a = accList[i]
            val g = gyroList[j]

            // 添加时间差阈值检查（5ms）
            val timeDiff = abs(a.timestamp - g.timestamp)
            if (timeDiff > 5_000_000L) { // 5ms in nanoseconds
                when {
                    a.timestamp < g.timestamp -> {
                        Log.w(TAG, "Large time gap: acc=${a.timestamp}ns gyro=${g.timestamp}ns (diff=${timeDiff/1e6}ms)")
                        i++
                    }
                    else -> {
                        Log.w(TAG, "Large time gap: gyro=${g.timestamp}ns acc=${a.timestamp}ns (diff=${timeDiff/1e6}ms)")
                        j++
                    }
                }
                continue
            }

            // 修改合并策略：取两者时间戳的平均值
            val mergedTimestamp = (a.timestamp + g.timestamp) / 2
            merged.add(
                SensorDataAll(
                    timestamp = mergedTimestamp,
                    accX = a.x, accY = a.y, accZ = a.z,
                    gyroX = g.x, gyroY = g.y, gyroZ = g.z
                )
            )
            i++; j++
        }
        return merged
    }

    // ================================== 加载 TFLite 模型 ==================================
    @Throws(IOException::class)
    private fun loadTfliteModel() {
        val modelPath = getModelFileName(dataSetName)
        val buffer: MappedByteBuffer = FileUtil.loadMappedFile(context, modelPath)
        val options = Interpreter.Options()
        options.setNumThreads(2) // 可配置
        tfliteInterpreter = Interpreter(buffer, options)
        Log.i(TAG, "TFLite model loaded: $modelPath")
    }

    // ================================== 读取预处理参数 ==================================
    private fun loadPreprocessingParams() {
        val fileName = getPreprocessingParamsFileName(dataSetName)
        try {
            val jsonStr = context.assets.open(fileName).bufferedReader().use { it.readText() }
            val jsonObj = JSONObject(jsonStr)
            val sensors = jsonObj.getJSONObject("sensors")
            val accParams = sensors.getJSONObject("acceleration")
            val gyroParams = sensors.getJSONObject("gyroscope")

            // 分轴加载参数（直接赋值给类成员变量）
            this.accMeanX = accParams.getJSONObject("x").getDouble("mean").toFloat()
            this.accStdX = accParams.getJSONObject("x").getDouble("std").toFloat()
            this.accMeanY = accParams.getJSONObject("y").getDouble("mean").toFloat()
            this.accStdY = accParams.getJSONObject("y").getDouble("std").toFloat()
            this.accMeanZ = accParams.getJSONObject("z").getDouble("mean").toFloat()
            this.accStdZ = accParams.getJSONObject("z").getDouble("std").toFloat()

            this.gyroMeanX = gyroParams.getJSONObject("x").getDouble("mean").toFloat()
            this.gyroStdX = gyroParams.getJSONObject("x").getDouble("std").toFloat()
            this.gyroMeanY = gyroParams.getJSONObject("y").getDouble("mean").toFloat()
            this.gyroStdY = gyroParams.getJSONObject("y").getDouble("std").toFloat()
            this.gyroMeanZ = gyroParams.getJSONObject("z").getDouble("mean").toFloat()
            this.gyroStdZ = gyroParams.getJSONObject("z").getDouble("std").toFloat()

            val isRealWorld = dataSetName.equals("realworld", ignoreCase = true)
            val accUnitScale = if (isRealWorld) 9.80665f else 1f
            val gyroUnitScale = if (isRealWorld) (Math.PI/180).toFloat() else 1f

            // 直接操作类成员变量
            this.accMeanX *= accUnitScale
            this.accStdX *= accUnitScale
            this.accMeanY *= accUnitScale
            this.accStdY *= accUnitScale
            this.accMeanZ *= accUnitScale
            this.accStdZ *= accUnitScale
            this.gyroMeanX *= gyroUnitScale
            this.gyroStdX *= gyroUnitScale
            this.gyroMeanY *= gyroUnitScale
            this.gyroStdY *= gyroUnitScale
            this.gyroMeanZ *= gyroUnitScale
            this.gyroStdZ *= gyroUnitScale

            Log.i(TAG, """
                |Loaded preprocessing params:
                |Acc X: mean=${"%.3f".format(this.accMeanX)}, std=${"%.3f".format(this.accStdX)}
                |Acc Y: mean=${"%.3f".format(this.accMeanY)}, std=${"%.3f".format(this.accStdY)}
                |Acc Z: mean=${"%.3f".format(this.accMeanZ)}, std=${"%.3f".format(this.accStdZ)}
                |Gyro X: mean=${"%.3f".format(this.gyroMeanX)}, std=${"%.3f".format(this.gyroStdX)}
                |Gyro Y: mean=${"%.3f".format(this.gyroMeanY)}, std=${"%.3f".format(this.gyroStdY)}
                |Gyro Z: mean=${"%.3f".format(this.gyroMeanZ)}, std=${"%.3f".format(this.gyroStdZ)}
            """.trimMargin())
        } catch (ex: Exception) {
            Log.w(TAG, "Failed to load preprocess params, using identity normalization", ex)
            // 初始化各轴默认参数
            this.accMeanX = 0f; this.accStdX = 1f
            this.accMeanY = 0f; this.accStdY = 1f
            this.accMeanZ = 0f; this.accStdZ = 1f
            this.gyroMeanX = 0f; this.gyroStdX = 1f
            this.gyroMeanY = 0f; this.gyroStdY = 1f
            this.gyroMeanZ = 0f; this.gyroStdZ = 1f
        }
    }
}

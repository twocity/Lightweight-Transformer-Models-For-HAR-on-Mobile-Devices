package com.twocity.handedness.core

import android.annotation.SuppressLint
import android.content.Context
import android.hardware.*
import android.util.Log
import org.json.JSONObject
import org.tensorflow.lite.Interpreter
import org.tensorflow.lite.support.common.FileUtil
import java.io.IOException
import java.nio.MappedByteBuffer
import java.util.ArrayDeque
import kotlin.math.abs
import kotlin.math.max
import kotlin.math.min

/**
 * HARInference:
 * - 分别采集加速度计、陀螺仪数据到各自缓冲
 * - 分别对加速度计/陀螺仪按指定频率重采样
 * - 在同一时间轴上合并得到 [accX,accY,accZ, gyroX,gyroY,gyroZ]
 * - 根据训练阶段输出的 JSON 做数据标准化
 * - 使用 TFLite 模型进行推断
 *
 * 支持多个数据集: RealWorld, HHAR, UCI, SHL, MotionSense, COMBINED，默认 "UCI"
 */
class HARInference(
    private val context: Context,
    private val callback: HARCallback,
    private val dataSetName: String = "UCI",       // 默认数据集
    private val desiredSamplingRate: Int = 50,     // 目标采样率(Hz)
    private val windowSize: Int = 128,             // 窗口大小
    private val slideStep: Int = 64,               // 窗口滑动步长(可二次扩展)
    private val bufferOverlap: Float = 0.5f,       // 保留一定重叠数据比例
    private val smoothWindowSize: Int = 5,         // 结果平滑窗口大小
    private val confidenceThreshold: Float = 0.8f, // 置信度阈值
    private val minTimeBetweenInferences: Long = 2000  // 推理最小时间间隔(毫秒)
) : SensorEventListener {

    companion object {
        private const val TAG = "HARInference"

        // --------------------------- 活动标签映射 ---------------------------
        private val LABELS_UCI = arrayOf("Walking", "Upstair","Downstair", "Sitting", "Standing", "Lying")

        private val LABELS_HHAR = arrayOf("Sitting", "Standing", "Walking", "Upstairs", "Downstairs", "Biking")
        private val LABELS_REALWORLD = arrayOf(
            "Downstairs", "Upstairs", "Jumping", "Lying", "Running", "Sitting", "Standing", "Walking"
        )
        private val LABELS_MOTIONSENSE = arrayOf("Downstairs", "Upstairs", "Sitting", "Standing", "Walking", "Jogging")
        private val LABELS_SHL = arrayOf("Standing", "Walking", "Runing", "Biking", "Car", "Bus", "Train", "Subway")
        // COMBINED (示例)
        private val LABELS_COMBINED = arrayOf(
            "Walk","Upstair","Downstair","Sit","Stand","Lay","Jump",
            "Run","Bike","Car","Bus","Train","Subway"
        )

        // 根据 dataSetName 返回对应标签数组
        fun getActivityLabelsByDataSet(dsName: String): Array<String> {
            return when (dsName.trim().lowercase()) {
                "hhar" -> LABELS_HHAR
                "realworld" -> LABELS_REALWORLD
                "motionsense" -> LABELS_MOTIONSENSE
                "shl" -> LABELS_SHL
                "combined" -> LABELS_COMBINED
                else -> LABELS_UCI
            }
        }

        // --------------------------- 文件名映射 ---------------------------
        fun getModelFileName(dsName: String): String {
            // 若要针对不同数据集使用不同文件，可在此扩展
            return "har/${dsName.lowercase()}/MobileHART.tflite"
        }

        fun getPreprocessingParamsFileName(dsName: String): String {
            return "har/${dsName.lowercase()}/preprocessing_params.json"
        }
    }

    /**
     * HARCallback:
     * 对外回调接口，用于活动识别结果输出或错误通知
     */
    interface HARCallback {
        fun onActivityRecognized(activity: String, confidence: Float)
        fun onError(error: String)
    }

    // =========================== 传感器相关变量 ===========================
    private var sensorManager: SensorManager? = null
    private var accSensor: Sensor? = null
    private var gyrSensor: Sensor? = null

    // 加速度计缓冲 & 陀螺仪缓冲
    private val accBuffer = mutableListOf<AccData>()
    private val gyroBuffer = mutableListOf<GyroData>()
    private val bufferLock = Any() // 同步锁

    // 数据结构：加速度计
    data class AccData(
        val timestamp: Long,
        val x: Float,
        val y: Float,
        val z: Float
    )

    // 数据结构：陀螺仪
    data class GyroData(
        val timestamp: Long,
        val x: Float,
        val y: Float,
        val z: Float
    )

    // 合并后的统一数据结构
    data class SensorDataAll(
        val timestamp: Long,
        val accX: Float,
        val accY: Float,
        val accZ: Float,
        val gyroX: Float,
        val gyroY: Float,
        val gyroZ: Float
    )

    // =========================== 模型和预处理相关 ===========================
    private var tfliteInterpreter: Interpreter? = null

    // 训练时计算的均值和标准差(Acc & Gyro)
    private var accMean: Float = 0f
    private var accStd: Float = 1f
    private var gyroMean: Float = 0f
    private var gyroStd: Float = 1f

    // 两个传感器的最近时间戳
    private var lastAccTimestampNs: Long = -1L
    private var lastGyroTimestampNs: Long = -1L

    // 根据数据集加载的标签
    private var activityLabels: Array<String> = getActivityLabelsByDataSet(dataSetName)

    // 历史预测，用于平滑
    private val recentPredictions = ArrayDeque<Pair<String, Float>>(smoothWindowSize)

    // 上一次推理时间
    private var lastInferenceTime = 0L

    // Add data class for preprocessing parameters
    private data class PreprocessingParams(
        val accMeanX: Float = 0f,
        val accMeanY: Float = 0f,
        val accMeanZ: Float = 0f,
        val accStdX: Float = 1f,
        val accStdY: Float = 1f,
        val accStdZ: Float = 1f,
        val gyroMeanX: Float = 0f,
        val gyroMeanY: Float = 0f,
        val gyroMeanZ: Float = 0f,
        val gyroStdX: Float = 1f,
        val gyroStdY: Float = 1f,
        val gyroStdZ: Float = 1f
    )

    // Update preprocessing params loading
    private var preprocessingParams = PreprocessingParams()

    /**
     * start():
     *  1) 加载 TFLite 模型
     *  2) 读取预处理参数
     *  3) 注册传感器监听 (FASTEST)，后续使用重采样
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

            // 注册传感器监听，尽快发送，后续再自行重采样
            sensorManager?.registerListener(this, accSensor, SensorManager.SENSOR_DELAY_FASTEST)
            sensorManager?.registerListener(this, gyrSensor, SensorManager.SENSOR_DELAY_FASTEST)

            Log.i(TAG, "HARInference started: dataset=$dataSetName freq=$desiredSamplingRate windowSize=$windowSize")
        } catch (ex: Exception) {
            Log.e(TAG, "start() failed: ${ex.message}", ex)
            callback.onError("start() failed: ${ex.message}")
        }
    }

    /**
     * release():
     *  1) 反注册传感器
     *  2) 关闭 Interpreter
     *  3) 清空缓存
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

    // ================ SensorEventListener 接口实现 ================
    override fun onAccuracyChanged(sensor: Sensor?, accuracy: Int) {
        // 暂不处理
    }

    @SuppressLint("DefaultLocale")
    override fun onSensorChanged(event: SensorEvent?) {
        if (event == null) return

        val sensorType = event.sensor?.type ?: return
        val timestampNs = event.timestamp

        // 分别取 x/y/z，若取不到则视为0
        val x = event.values.getOrNull(0) ?: 0f
        val y = event.values.getOrNull(1) ?: 0f
        val z = event.values.getOrNull(2) ?: 0f

        // 根据传感器类型写入各自 buffer
        synchronized(bufferLock) {
            when (sensorType) {
                Sensor.TYPE_ACCELEROMETER -> {
                    if (lastAccTimestampNs < 0) lastAccTimestampNs = timestampNs
                    accBuffer.add(AccData(timestampNs, x, y, z))
                }
                Sensor.TYPE_GYROSCOPE -> {
                    if (lastGyroTimestampNs < 0) lastGyroTimestampNs = timestampNs
                    gyroBuffer.add(GyroData(timestampNs, x, y, z))
                }
            }

            // 当两个缓冲都超出一定量，则执行推理
            if (accBuffer.size > windowSize * 2 && gyroBuffer.size > windowSize * 2) {
                processDataAndRunInference()
            }
        }
    }

    // =========================== 数据处理与推断 ===========================
    /**
     * processDataAndRunInference():
     * - 复制本地缓存
     * - 根据 overlap 参数保留部分历史数据
     * - 检查样本数是否满足
     * - 分别重采样加速度/陀螺仪
     * - 合并到同一时间轴
     * - 截取最后 [windowSize] 条
     * - 标准化 + 推理 + 平滑输出
     */
    private fun processDataAndRunInference() {
        val localAcc: List<AccData>
        val localGyro: List<GyroData>

        // 1. 拷贝并清理 buffer
        synchronized(bufferLock) {
            localAcc = accBuffer.toList()
            localGyro = gyroBuffer.toList()

            // 根据 bufferOverlap 保留一部分旧数据
            val keepSize = (windowSize * bufferOverlap).toInt()
            if (accBuffer.size > keepSize) {
                accBuffer.subList(0, accBuffer.size - keepSize).clear()
            }
            if (gyroBuffer.size > keepSize) {
                gyroBuffer.subList(0, gyroBuffer.size - keepSize).clear()
            }
        }

        // 2. 检查是否有足够的点
        val minSamples = windowSize
        if (localAcc.size < minSamples || localGyro.size < minSamples) {
            Log.d(TAG, "Not enough samples to infer: acc=${localAcc.size}, gyro=${localGyro.size}, needed=$minSamples")
            return
        }

        // 3. 重采样
        val resampledAcc = reSampleAccData(localAcc, desiredSamplingRate)
        val resampledGyro = reSampleGyroData(localGyro, desiredSamplingRate)

        // 4. 检查重采样结果
        if (resampledAcc.size < windowSize || resampledGyro.size < windowSize) {
            Log.d(
                TAG,
                "Insufficient resampled data: acc=${resampledAcc.size}, gyro=${resampledGyro.size}, window=$windowSize"
            )
            return
        }

        // 5. 合并数据
        val combinedList = combineResampledData(resampledAcc, resampledGyro)
        if (combinedList.size < windowSize) {
            Log.d(TAG, "Combined data < windowSize: ${combinedList.size}")
            return
        }

        // 6. 取末端 windowSize 长度的切片
        val finalSegment = combinedList.takeLast(windowSize)

        // 7. 检查推理时间间隔
        val currentTime = System.currentTimeMillis()
        if (currentTime - lastInferenceTime < minTimeBetweenInferences) {
            Log.d(TAG, "Inference too frequent; skip this round.")
            return
        }

        try {
            // 8. 执行推理
            val prediction = runModelInference(finalSegment) ?: return
            // 9. 平滑处理并输出
            processPredictionWithSmoothing(prediction)

            lastInferenceTime = currentTime
        } catch (e: Exception) {
            Log.e(TAG, "Inference error:", e)
            callback.onError(e.message ?: "Unknown error")
        }
    }

    /**
     * runModelInference():
     *  - 将 finalSegment 标准化并构造成 [1, windowSize, 6]
     *  - 调用 TFLite Interpreter
     *  - 找出最大置信度活动并阈值检测
     */
    private fun runModelInference(finalSegment: List<SensorDataAll>): Pair<String, Float>? {
        if (finalSegment.size != windowSize) {
            Log.w(TAG, "Segment size mismatch: got=${finalSegment.size}, needed=$windowSize")
            return null
        }

        // 添加原始数据日志
        Log.d(TAG, "Raw data sample: ${finalSegment.take(3).map {
            "Acc(${it.accX},${it.accY},${it.accZ}) Gyro(${it.gyroX},${it.gyroY},${it.gyroZ})"
        }}")

        // 添加标准化后数据日志
        Log.d(TAG, "Normalized data sample: ${
            finalSegment.take(3).map { data ->
                "[${(data.accX - preprocessingParams.accMeanX)/preprocessingParams.accStdX}, " +
                        "${(data.gyroX - preprocessingParams.gyroMeanX)/preprocessingParams.gyroStdX}]"
            }
        }")

        // 构建输入
        val modelInput = Array(1) { Array(windowSize) { FloatArray(6) } }

        for ((i, data) in finalSegment.withIndex()) {
            // Normalize each axis independently using the loaded parameters
            modelInput[0][i][0] = (data.accX - preprocessingParams.accMeanX) / preprocessingParams.accStdX
            modelInput[0][i][1] = (data.accY - preprocessingParams.accMeanY) / preprocessingParams.accStdY
            modelInput[0][i][2] = (data.accZ - preprocessingParams.accMeanZ) / preprocessingParams.accStdZ
            modelInput[0][i][3] = (data.gyroX - preprocessingParams.gyroMeanX) / preprocessingParams.gyroStdX
            modelInput[0][i][4] = (data.gyroY - preprocessingParams.gyroMeanY) / preprocessingParams.gyroStdY
            modelInput[0][i][5] = (data.gyroZ - preprocessingParams.gyroMeanZ) / preprocessingParams.gyroStdZ
        }

        // 输出缓冲
        val outputBuffer = Array(1) { FloatArray(activityLabels.size) }
        tfliteInterpreter?.run(modelInput, outputBuffer) ?: return null

        // 找到最大置信度
        var maxIdx = 0
        var maxVal = outputBuffer[0][0]
        for (i in outputBuffer[0].indices) {
            if (outputBuffer[0][i] > maxVal) {
                maxVal = outputBuffer[0][i]
                maxIdx = i
            }
        }

        // 置信度小于阈值则视为无效
        if (maxVal < confidenceThreshold) {
            Log.d(TAG, "Confidence($maxVal) below threshold($confidenceThreshold). Discarding.")
            return null
        }
        val actName = activityLabels.getOrNull(maxIdx) ?: "Unknown"
        return Pair(actName, maxVal)
    }

    /**
     * processPredictionWithSmoothing():
     *  - 维护一个双端队列存最近若干次预测
     *  - 按出现次数和平均置信度做加权输出
     */
    private fun processPredictionWithSmoothing(prediction: Pair<String, Float>) {
        synchronized(recentPredictions) {
            if (recentPredictions.size >= smoothWindowSize) {
                recentPredictions.removeFirst()
            }
            recentPredictions.addLast(prediction)

            // 按活动名分组取平均置信度
            val grouped = recentPredictions.groupBy { it.first }
            val confMap = grouped.mapValues { (_, values) -> values.map { it.second }.average() }

            // 评分 = 平均置信度 * (出现次数 / 队列大小)
            val smoothed = confMap.maxByOrNull { (act, avgConf) ->
                avgConf * recentPredictions.count { it.first == act } / recentPredictions.size.toDouble()
            }

            smoothed?.let { (activity, confidence) ->
                callback.onActivityRecognized(activity, confidence.toFloat())
                Log.d(TAG, "Smoothed inference: activity=$activity, confidence=$confidence")
            }
        }
    }

    // ======================= 重采样: 加速度计 =======================
    private fun reSampleAccData(accList: List<AccData>, targetHz: Int): List<AccData> {
        if (accList.size < 2) {
            Log.w(TAG, "Acc reSampleData: not enough points -> ${accList.size}")
            return emptyList()
        }

        val sortedAcc = accList.sortedBy { it.timestamp }
        val timeSpanNs = sortedAcc.last().timestamp - sortedAcc.first().timestamp
        val targetPoints = max((timeSpanNs * targetHz / 1e9).toInt(), windowSize)

        Log.d(TAG, "Acc resampling: timespanNs=${timeSpanNs}, points=$targetPoints")

        // 不超过 windowSize
        val numPoints = minOf(targetPoints, windowSize)
        if (numPoints < 2) return emptyList()

        val dtNs = timeSpanNs / (numPoints - 1)
        val result = mutableListOf<AccData>()

        var t = sortedAcc.first().timestamp
        var idx = 0

        repeat(numPoints) {
            while (idx < sortedAcc.size - 1 && sortedAcc[idx + 1].timestamp < t) {
                idx++
            }
            if (idx >= sortedAcc.size - 1) {
                idx = sortedAcc.size - 2
            }

            val d1 = sortedAcc[idx]
            val d2 = sortedAcc[idx + 1]
            val ratio = (t - d1.timestamp).toDouble() / (d2.timestamp - d1.timestamp)
            val x = d1.x + (d2.x - d1.x) * ratio.toFloat()
            val y = d1.y + (d2.y - d1.y) * ratio.toFloat()
            val z = d1.z + (d2.z - d1.z) * ratio.toFloat()

            result.add(AccData(t, x, y, z))
            t += dtNs
        }
        return result
    }

    // ======================= 重采样: 陀螺仪 =======================
    private fun reSampleGyroData(gyroList: List<GyroData>, targetHz: Int): List<GyroData> {
        if (gyroList.size < 2) {
            Log.w(TAG, "Gyro reSampleData: not enough points -> ${gyroList.size}")
            return emptyList()
        }

        val sortedGyro = gyroList.sortedBy { it.timestamp }
        val timeSpanNs = sortedGyro.last().timestamp - sortedGyro.first().timestamp
        val targetPoints = max((timeSpanNs * targetHz / 1e9).toInt(), windowSize)

        Log.d(TAG, "Gyro resampling: timespanNs=$timeSpanNs, points=$targetPoints")

        val numPoints = minOf(targetPoints, windowSize)
        if (numPoints < 2) return emptyList()

        val dtNs = timeSpanNs / (numPoints - 1)
        val result = mutableListOf<GyroData>()

        var t = sortedGyro.first().timestamp
        var idx = 0

        repeat(numPoints) {
            while (idx < sortedGyro.size - 1 && sortedGyro[idx + 1].timestamp < t) {
                idx++
            }
            if (idx >= sortedGyro.size - 1) {
                idx = sortedGyro.size - 2
            }

            val d1 = sortedGyro[idx]
            val d2 = sortedGyro[idx + 1]
            val ratio = (t - d1.timestamp).toDouble() / (d2.timestamp - d1.timestamp)
            val x = d1.x + (d2.x - d1.x) * ratio.toFloat()
            val y = d1.y + (d2.y - d1.y) * ratio.toFloat()
            val z = d1.z + (d2.z - d1.z) * ratio.toFloat()

            result.add(GyroData(t, x, y, z))
            t += dtNs
        }
        return result
    }

    // ======================= 合并加速度+陀螺仪 =======================
    private fun combineResampledData(
        accList: List<AccData>,
        gyroList: List<GyroData>
    ): List<SensorDataAll> {
        if (accList.isEmpty() || gyroList.isEmpty()) return emptyList()

        val merged = mutableListOf<SensorDataAll>()
        val startTime = max(accList.first().timestamp, gyroList.first().timestamp)
        val endTime = min(accList.last().timestamp, gyroList.last().timestamp)

        val MAX_TIME_DIFF_NS = 20_000_000L // 放宽到20ms

        var i = 0
        var j = 0
        while (i < accList.size && j < gyroList.size) {
            val a = accList[i]
            val g = gyroList[j]

            // 时间窗口过滤
            if (a.timestamp > endTime || g.timestamp > endTime) break
            if (a.timestamp < startTime) { i++; continue }
            if (g.timestamp < startTime) { j++; continue }

            // 改进的时间匹配逻辑
            when {
                abs(a.timestamp - g.timestamp) <= MAX_TIME_DIFF_NS -> {
                    merged.add(SensorDataAll(
                        (a.timestamp + g.timestamp) / 2,
                        a.x, a.y, a.z, g.x, g.y, g.z
                    ))
                    i++; j++
                }
                a.timestamp < g.timestamp -> {
                    merged.add(SensorDataAll(a.timestamp, a.x, a.y, a.z, 0f, 0f, 0f))
                    i++
                }
                else -> {
                    merged.add(SensorDataAll(g.timestamp, 0f, 0f, 0f, g.x, g.y, g.z))
                    j++
                }
            }
        }
        return merged
    }

    // ================ 加载 TFLite 模型 ================
    @Throws(IOException::class)
    private fun loadTfliteModel() {
        val modelPath = getModelFileName(dataSetName)
        val buffer: MappedByteBuffer = FileUtil.loadMappedFile(context, modelPath)

        val options = Interpreter.Options()
        options.setNumThreads(2) // 默认2线程，可按需求调整

        tfliteInterpreter = Interpreter(buffer, options)
        Log.i(TAG, "TFLite model loaded from: $modelPath")
    }

    // ================ 读取预处理参数 ================
    private fun loadPreprocessingParams() {
        try {
            val fileName = getPreprocessingParamsFileName(dataSetName)
            val jsonStr = context.assets.open(fileName).bufferedReader().use { it.readText() }
            val jsonObj = JSONObject(jsonStr)

            // Get sensors object
            val sensors = jsonObj.getJSONObject("sensors")

            // Get acceleration parameters
            val acc = sensors.getJSONObject("acceleration")
            val gyro = sensors.getJSONObject("gyroscope")

            // 验证每个轴的数据存在
            require(acc.has("x") && acc.has("y") && acc.has("z")) { "Invalid acceleration structure" }
            require(gyro.has("x") && gyro.has("y") && gyro.has("z")) { "Invalid gyroscope structure" }

            // 解析每个轴的参数
            preprocessingParams = PreprocessingParams(
                accMeanX = acc.getJSONObject("x").getDouble("mean").toFloat(),
                accMeanY = acc.getJSONObject("y").getDouble("mean").toFloat(),
                accMeanZ = acc.getJSONObject("z").getDouble("mean").toFloat(),
                accStdX = acc.getJSONObject("x").getDouble("std").toFloat(),
                accStdY = acc.getJSONObject("y").getDouble("std").toFloat(),
                accStdZ = acc.getJSONObject("z").getDouble("std").toFloat(),
                gyroMeanX = gyro.getJSONObject("x").getDouble("mean").toFloat(),
                gyroMeanY = gyro.getJSONObject("y").getDouble("mean").toFloat(),
                gyroMeanZ = gyro.getJSONObject("z").getDouble("mean").toFloat(),
                gyroStdX = gyro.getJSONObject("x").getDouble("std").toFloat(),
                gyroStdY = gyro.getJSONObject("y").getDouble("std").toFloat(),
                gyroStdZ = gyro.getJSONObject("z").getDouble("std").toFloat()
            )

            // 添加参数有效性检查
            if (preprocessingParams.accStdX <= 0.0001f ||
                preprocessingParams.gyroStdX <= 0.0001f) {
                throw IllegalStateException("Invalid standard deviation values")
            }

            Log.i(TAG, "Preprocessing params loaded for $dataSetName: $preprocessingParams")
        } catch (ex: Exception) {
            Log.w(TAG, "Failed to load preprocessing params. Using defaults.", ex)
            preprocessingParams = PreprocessingParams() // Use defaults
        }
    }
}

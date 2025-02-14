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
import kotlin.math.max
import kotlin.math.min

/**
 * HARConfig:
 * 集中管理 HAR 推理相关的所有配置参数
 */
data class HARConfig(
    val dataSetName: String = "MotionSense",     // 默认数据集名称
    val desiredSamplingRate: Int = 50,             // 目标采样率(Hz)
    val windowSize: Int = 128,                     // 窗口大小
    val slideStep: Int = 64,                       // 窗口滑动步长（未使用，可扩展）
    val bufferOverlap: Float = 0.5f,               // 数据缓冲中保留的重叠比例
    val smoothWindowSize: Int = 5,                 // 平滑结果时使用的窗口大小
    val confidenceThreshold: Float = 0.6f,         // 推理置信度阈值
    val minTimeBetweenInferences: Long = 1000      // 推理的最小时间间隔（毫秒）
)

/**
 * HARInference:
 * - 分别采集加速度计、陀螺仪数据到各自缓冲
 * - 使用统一时间轴对加速度计/陀螺仪进行重采样
 * - 在同一时间轴上合并得到 [accX,accY,accZ, gyroX,gyroY,gyroZ]
 * - 根据训练阶段输出的 JSON 做数据标准化
 * - 使用 TFLite 模型进行推断
 *
 * 支持多个数据集: RealWorld, HHAR, UCI, SHL, MotionSense, COMBINED
 */
class HARInference(
    private val context: Context,
    private val callback: HARCallback,
    private val config: HARConfig = HARConfig()
) : SensorEventListener {

    companion object {
        private const val TAG = "HARInference"

        // --------------------------- 活动标签映射 ---------------------------
        private val LABELS_UCI = arrayOf("Walking", "Upstair", "Downstair", "Sitting", "Standing", "Lying")
        private val LABELS_HHAR = arrayOf("Sitting", "Standing", "Walking", "Upstairs", "Downstairs", "Biking")
        private val LABELS_REALWORLD = arrayOf(
            "Downstairs", "Upstairs", "Jumping", "Lying", "Running", "Sitting", "Standing", "Walking"
        )
        private val LABELS_MOTIONSENSE = arrayOf("Downstairs", "Upstairs", "Sitting", "Standing", "Walking", "Jogging")
        private val LABELS_SHL = arrayOf("Standing", "Walking", "Runing", "Biking", "Car", "Bus", "Train", "Subway")
        // COMBINED (示例)
        private val LABELS_COMBINED = arrayOf(
            "Walk", "Upstair", "Downstair", "Sit", "Stand", "Lay", "Jump",
            "Run", "Bike", "Car", "Bus", "Train", "Subway"
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
            // 针对不同数据集使用不同模型文件，可在此扩展
            return "har/${dsName.lowercase()}/MobileHART.tflite"
        }

        fun getPreprocessingParamsFileName(dsName: String): String {
            return "har/${dsName.lowercase()}/dataset_params.json"
        }
    }

    /**
     * HARCallback:
     * 对外回调接口，用于输出活动识别结果或错误通知
     */
    interface HARCallback {
        fun onActivityRecognized(activity: String, confidence: Float)
        fun onError(error: String)
    }

    // =========================== 传感器相关变量 ===========================
    private var sensorManager: SensorManager? = null
    private var accSensor: Sensor? = null
    private var gyrSensor: Sensor? = null

    // 加速度计与陀螺仪数据缓冲
    private val accBuffer = mutableListOf<AccData>()
    private val gyroBuffer = mutableListOf<GyroData>()
    private val bufferLock = Any() // 同步锁

    // 数据结构：加速度计数据
    data class AccData(
        val timestamp: Long,
        val x: Float,
        val y: Float,
        val z: Float
    )

    // 数据结构：陀螺仪数据
    data class GyroData(
        val timestamp: Long,
        val x: Float,
        val y: Float,
        val z: Float
    )

    // 合并后的传感器数据结构
    data class SensorDataAll(
        val timestamp: Long,
        val accX: Float,
        val accY: Float,
        val accZ: Float,
        val gyroX: Float,
        val gyroY: Float,
        val gyroZ: Float
    )

    // =========================== 模型与预处理相关变量 ===========================
    private var tfliteInterpreter: Interpreter? = null

    // 两个传感器最近的时间戳（仅作记录）
    private var lastAccTimestampNs: Long = -1L
    private var lastGyroTimestampNs: Long = -1L

    // 根据数据集加载的活动标签
    private var activityLabels: Array<String> = getActivityLabelsByDataSet(config.dataSetName)

    // 用于平滑处理的历史预测队列
    private val recentPredictions = ArrayDeque<Pair<String, Float>>(config.smoothWindowSize)

    // 上一次推理时间
    private var lastInferenceTime = 0L

    // 预处理参数数据类（均值、标准差）
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

    // 当前预处理参数
    private var preprocessingParams = PreprocessingParams()

    /**
     * start():
     *  1) 加载 TFLite 模型
     *  2) 读取预处理参数
     *  3) 注册传感器监听（使用最快模式采集数据）
     */
    fun start() {
        try {
            sensorManager = context.getSystemService(Context.SENSOR_SERVICE) as SensorManager
            accSensor = sensorManager?.getDefaultSensor(Sensor.TYPE_LINEAR_ACCELERATION)
            gyrSensor = sensorManager?.getDefaultSensor(Sensor.TYPE_GYROSCOPE)

            if (accSensor == null || gyrSensor == null) {
                callback.onError("Accelerometer or Gyroscope not found.")
                return
            }

            loadTfliteModel()
            loadPreprocessingParams()

            // 注册传感器监听，使用最快速模式
            sensorManager?.registerListener(this, accSensor, SensorManager.SENSOR_DELAY_FASTEST)
            sensorManager?.registerListener(this, gyrSensor, SensorManager.SENSOR_DELAY_FASTEST)

            Log.i(
                TAG, "HARInference started: dataset=${config.dataSetName} " +
                        "freq=${config.desiredSamplingRate}Hz windowSize=${config.windowSize}"
            )
        } catch (ex: Exception) {
            Log.e(TAG, "start() failed: ${ex.message}", ex)
            callback.onError("start() failed: ${ex.message}")
        }
    }

    /**
     * release():
     *  1) 注销传感器监听
     *  2) 关闭 TFLite Interpreter
     *  3) 清空数据缓存
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

        // 获取 x, y, z 数据，若不存在则默认0
        val x = event.values.getOrNull(0) ?: 0f
        val y = event.values.getOrNull(1) ?: 0f
        val z = event.values.getOrNull(2) ?: 0f

        // 根据传感器类型将数据添加到对应缓冲中
        synchronized(bufferLock) {
            when (sensorType) {
                Sensor.TYPE_LINEAR_ACCELERATION -> {
                    if (lastAccTimestampNs < 0) lastAccTimestampNs = timestampNs
                    accBuffer.add(AccData(timestampNs, x, y, z))
                }
                Sensor.TYPE_GYROSCOPE -> {
                    if (lastGyroTimestampNs < 0) lastGyroTimestampNs = timestampNs
                    gyroBuffer.add(GyroData(timestampNs, x, y, z))
                }
            }

            // 当两个缓冲中的数据量超过 windowSize * 2 时触发推理
            if (accBuffer.size > config.windowSize * 1 && gyroBuffer.size > config.windowSize * 1) {
                processDataAndRunInference()
            }
        }
    }

    // =========================== 数据处理与推断 ===========================
    /**
     * processDataAndRunInference():
     * - 复制本地缓存，并根据 bufferOverlap 保留部分历史数据
     * - 使用统一时间轴对加速度计和陀螺仪数据进行重采样，生成长度为 windowSize 的序列
     * - 标准化数据后构造成模型输入，调用 TFLite 模型进行推断
     * - 对推断结果进行平滑处理后回调输出
     */
    private fun processDataAndRunInference() {
        val localAcc: List<AccData>
        val localGyro: List<GyroData>

        // 1. 复制并清理缓冲区数据
        synchronized(bufferLock) {
            localAcc = accBuffer.toList()
            localGyro = gyroBuffer.toList()

            // 根据 bufferOverlap 参数保留部分旧数据
            val keepSize = (config.windowSize * config.bufferOverlap).toInt()
            if (accBuffer.size > keepSize) {
                accBuffer.subList(0, accBuffer.size - keepSize).clear()
            }
            if (gyroBuffer.size > keepSize) {
                gyroBuffer.subList(0, gyroBuffer.size - keepSize).clear()
            }
        }

        // 2. 检查是否有足够的采样点
        if (localAcc.isEmpty() || localGyro.isEmpty()) {
            Log.d(TAG, "Not enough samples to infer: acc=${localAcc.size}, gyro=${localGyro.size}")
            return
        }

        // 3. 统一时间轴重采样，生成长度为 windowSize 的数据序列
        val finalSegment = reSampleDataOnCommonTimeline(localAcc, localGyro, config.windowSize)
        if (finalSegment.size != config.windowSize) {
            Log.d(TAG, "Unified resampling produced insufficient data: ${finalSegment.size} vs expected ${config.windowSize}")
            return
        }

        // 4. 检查推理间隔时间
        val currentTime = System.currentTimeMillis()
        if (currentTime - lastInferenceTime < config.minTimeBetweenInferences) {
            Log.d(TAG, "Inference too frequent; skip this round.")
            return
        }

        try {
            val prediction = runModelInference(finalSegment) ?: return
            processPredictionWithSmoothing(prediction)
            lastInferenceTime = currentTime
        } catch (e: Exception) {
            Log.e(TAG, "Inference error:", e)
            callback.onError(e.message ?: "Unknown error")
        }
    }

    /**
     * runModelInference():
     *  - 将 finalSegment 数据标准化并构造成形状 [1, windowSize, 6] 的模型输入
     *  - 调用 TFLite Interpreter 进行推理，并选取最大置信度的结果
     */
    private fun runModelInference(finalSegment: List<SensorDataAll>): Pair<String, Float>? {
        if (finalSegment.size != config.windowSize) {
            Log.w(TAG, "Segment size mismatch: got=${finalSegment.size}, needed=${config.windowSize}")
            return null
        }

        // 构造模型输入，形状 [1, windowSize, 6]
        val modelInput = Array(1) { Array(config.windowSize) { FloatArray(6) } }
        for ((i, data) in finalSegment.withIndex()) {
            modelInput[0][i][0] = (data.accX - preprocessingParams.accMeanX) / preprocessingParams.accStdX
            modelInput[0][i][1] = (data.accY - preprocessingParams.accMeanY) / preprocessingParams.accStdY
            modelInput[0][i][2] = (data.accZ - preprocessingParams.accMeanZ) / preprocessingParams.accStdZ
            modelInput[0][i][3] = (data.gyroX - preprocessingParams.gyroMeanX) / preprocessingParams.gyroStdX
            modelInput[0][i][4] = (data.gyroY - preprocessingParams.gyroMeanY) / preprocessingParams.gyroStdY
            modelInput[0][i][5] = (data.gyroZ - preprocessingParams.gyroMeanZ) / preprocessingParams.gyroStdZ
        }

        // 模型输出缓冲，形状 [1, numLabels]
        val outputBuffer = Array(1) { FloatArray(activityLabels.size) }
        tfliteInterpreter?.run(modelInput, outputBuffer) ?: return null

        // 选取具有最大置信度的活动
        var maxIdx = 0
        var maxVal = outputBuffer[0][0]
        for (i in outputBuffer[0].indices) {
            if (outputBuffer[0][i] > maxVal) {
                maxVal = outputBuffer[0][i]
                maxIdx = i
            }
        }

        // 若最大置信度低于设定阈值，则忽略该推断结果
        if (maxVal < config.confidenceThreshold) {
            Log.d(TAG, "Confidence($maxVal) below threshold(${config.confidenceThreshold}). Discarding.")
            return null
        }
        val actName = activityLabels.getOrNull(maxIdx) ?: "Unknown"
        return Pair(actName, maxVal)
    }

    /**
     * processPredictionWithSmoothing():
     *  - 将最近预测结果存入固定大小的队列中
     *  - 根据各活动的出现次数及平均置信度计算平滑后的最终结果
     */
    private fun processPredictionWithSmoothing(prediction: Pair<String, Float>) {
        synchronized(recentPredictions) {
            if (recentPredictions.size >= config.smoothWindowSize) {
                recentPredictions.removeFirst()
            }
            recentPredictions.addLast(prediction)

            val grouped = recentPredictions.groupBy { it.first }
            val confMap = grouped.mapValues { (_, values) -> values.map { it.second }.average() }
            val smoothed = confMap.maxByOrNull { (act, avgConf) ->
                avgConf * recentPredictions.count { it.first == act } / recentPredictions.size.toDouble()
            }
            smoothed?.let { (activity, confidence) ->
                callback.onActivityRecognized(activity, confidence.toFloat())
                Log.d(TAG, "Smoothed inference: activity=$activity, confidence=$confidence")
            }
        }
    }

    // ======================= 统一时间轴重采样实现 =======================
    /**
     * reSampleDataOnCommonTimeline():
     *  - 对加速度计和陀螺仪数据分别按时间排序
     *  - 计算两者的公共时间区间 [commonStart, commonEnd]
     *  - 根据公共时间区间和 windowSize 生成统一时间戳序列
     *  - 对每个时间戳进行线性插值，合并成 SensorDataAll 数据
     */
    private fun reSampleDataOnCommonTimeline(
        accList: List<AccData>,
        gyroList: List<GyroData>,
        windowSize: Int
    ): List<SensorDataAll> {
        val sortedAcc = accList.sortedBy { it.timestamp }
        val sortedGyro = gyroList.sortedBy { it.timestamp }
        val commonStart = max(sortedAcc.first().timestamp, sortedGyro.first().timestamp)
        val commonEnd = min(sortedAcc.last().timestamp, sortedGyro.last().timestamp)

        if (commonEnd <= commonStart) {
            Log.d(TAG, "No common time interval between accelerometer and gyroscope")
            return emptyList()
        }

        val timeSpanNs = commonEnd - commonStart
        val dt = if (windowSize > 1) timeSpanNs / (windowSize - 1) else timeSpanNs
        val commonTimestamps = List(windowSize) { commonStart + it * dt }
        val resampledData = mutableListOf<SensorDataAll>()

        for (t in commonTimestamps) {
            val accData = interpolateAcc(sortedAcc, t)
            val gyroData = interpolateGyro(sortedGyro, t)
            resampledData.add(
                SensorDataAll(
                    t,
                    accData.x, accData.y, accData.z,
                    gyroData.x, gyroData.y, gyroData.z
                )
            )
        }
        return resampledData
    }

    /**
     * interpolateAcc():
     *  - 对给定的加速度计数据列表，在指定时间 t 处进行线性插值
     */
    private fun interpolateAcc(data: List<AccData>, t: Long): AccData {
        if (t <= data.first().timestamp) return data.first()
        if (t >= data.last().timestamp) return data.last()
        var i = data.indexOfFirst { it.timestamp >= t }
        if (i <= 0) i = 1
        if (i >= data.size) i = data.size - 1
        val d1 = data[i - 1]
        val d2 = data[i]
        val ratio = (t - d1.timestamp).toDouble() / (d2.timestamp - d1.timestamp)
        return AccData(
            t,
            d1.x + ((d2.x - d1.x) * ratio).toFloat(),
            d1.y + ((d2.y - d1.y) * ratio).toFloat(),
            d1.z + ((d2.z - d1.z) * ratio).toFloat()
        )
    }

    /**
     * interpolateGyro():
     *  - 对给定的陀螺仪数据列表，在指定时间 t 处进行线性插值
     */
    private fun interpolateGyro(data: List<GyroData>, t: Long): GyroData {
        if (t <= data.first().timestamp) return data.first()
        if (t >= data.last().timestamp) return data.last()
        var i = data.indexOfFirst { it.timestamp >= t }
        if (i <= 0) i = 1
        if (i >= data.size) i = data.size - 1
        val d1 = data[i - 1]
        val d2 = data[i]
        val ratio = (t - d1.timestamp).toDouble() / (d2.timestamp - d1.timestamp)
        return GyroData(
            t,
            d1.x + ((d2.x - d1.x) * ratio).toFloat(),
            d1.y + ((d2.y - d1.y) * ratio).toFloat(),
            d1.z + ((d2.z - d1.z) * ratio).toFloat()
        )
    }

    // ================ TFLite 模型与预处理参数加载 ================
    @Throws(IOException::class)
    private fun loadTfliteModel() {
        val modelPath = getModelFileName(config.dataSetName)
        val buffer: MappedByteBuffer = FileUtil.loadMappedFile(context, modelPath)
        val options = Interpreter.Options().apply {
            setNumThreads(2) // 默认使用2个线程，可根据需求调整
        }
        tfliteInterpreter = Interpreter(buffer, options)
        Log.i(TAG, "TFLite model loaded from: $modelPath")
    }

    private fun loadPreprocessingParams() {
        try {
            val fileName = getPreprocessingParamsFileName(config.dataSetName)
            val jsonStr = context.assets.open(fileName).bufferedReader().use { it.readText() }
            val jsonObj = JSONObject(jsonStr)
            val sensors = jsonObj.getJSONObject("sensors")
            val acc = sensors.getJSONObject("acceleration")
            val gyro = sensors.getJSONObject("gyroscope")

            require(acc.has("x") && acc.has("y") && acc.has("z")) { "Invalid acceleration structure" }
            require(gyro.has("x") && gyro.has("y") && gyro.has("z")) { "Invalid gyroscope structure" }

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

            if (preprocessingParams.accStdX <= 0.0001f ||
                preprocessingParams.gyroStdX <= 0.0001f) {
                throw IllegalStateException("Invalid standard deviation values")
            }

            Log.i(TAG, "Preprocessing params loaded for ${config.dataSetName}: $preprocessingParams")
        } catch (ex: Exception) {
            Log.w(TAG, "Failed to load preprocessing params. Using defaults.", ex)
            preprocessingParams = PreprocessingParams() // 使用默认预处理参数
        }
    }
}

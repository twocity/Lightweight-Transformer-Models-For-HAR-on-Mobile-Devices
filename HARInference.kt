package com.twocity.handedness.core

import android.annotation.SuppressLint
import android.content.Context
import android.hardware.*
import android.util.Log
import org.tensorflow.lite.Interpreter
import org.tensorflow.lite.support.common.FileUtil
import java.io.IOException
import java.nio.MappedByteBuffer
import kotlin.math.max
import kotlin.math.min
import java.util.ArrayDeque

data class HARConfig(
    val dataSetName: String = "MotionSense",     // 模型训练使用的数据集名称（需与训练配置严格一致）
    val desiredSamplingRate: Int = 50,           // 目标重采样频率（单位Hz，建议与训练时采样率一致）
    val windowSize: Int = 128,                   // 时间窗口大小（必须与训练时的segment_size参数相同）
    val slideStep: Int = 64,                     // 窗口滑动步长（必须与训练时的timeStep参数相同，决定窗口重叠量）
    val bufferOverlap: Float = 0.3f,             // [已废弃] 请使用slideStep控制窗口滑动（保留字段用于兼容旧版本）
    val smoothWindowSize: Int = 15,              // 结果平滑窗口大小（建议值为推理频率的1/3~1/2）
    val confidenceThreshold: Float = 0.4f,      // 置信度阈值（低于此值的结果将被过滤）
    val minTimeBetweenInferences: Long = 1000,    // 最小推理间隔（毫秒，防止高频推理）
    val smoothStrategy: SmoothStrategy = SmoothStrategy.MAJORITY_VOTE
)

// 新增平滑策略枚举
enum class SmoothStrategy {
    MAJORITY_VOTE,  // 多数投票策略：选择窗口期内出现次数最多的活动
    MOVING_AVERAGE, // 滑动平均策略：计算窗口期内的平均置信度
    EXPONENTIAL_SMOOTHING // 指数平滑策略：近期结果具有更高权重（α=0.5）
}

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
        private val LABELS_UCI =
            arrayOf("Walking", "Upstair", "Downstair", "Sitting", "Standing", "Lying")
        private val LABELS_HHAR =
            arrayOf("Sitting", "Standing", "Walking", "Upstairs", "Downstairs", "Biking")
        private val LABELS_REALWORLD = arrayOf(
            "Downstairs",
            "Upstairs",
            "Jumping",
            "Lying",
            "Running",
            "Sitting",
            "Standing",
            "Walking"
        )
        private val LABELS_MOTIONSENSE =
            arrayOf("Downstairs", "Upstairs", "Sitting", "Standing", "Walking", "Jogging")
        private val LABELS_SHL =
            arrayOf("Standing", "Walking", "Runing", "Biking", "Car", "Bus", "Train", "Subway")

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
    }

    /**
     * HARCallback:
     * 对外回调接口，用于输出活动识别结果或错误通知
     */
    interface HARCallback {
        fun onActivityRecognized(activity: String, confidence: Float)
        fun onError(error: String)
    }

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

    private var tfliteInterpreter: Interpreter? = null

    // 两个传感器最近的时间戳（仅作记录）
    private var lastAccTimestampNs: Long = -1L
    private var lastGyroTimestampNs: Long = -1L

    // 根据数据集加载的活动标签
    private var activityLabels: Array<String> = getActivityLabelsByDataSet(config.dataSetName)

    // 上一次推理时间
    private var lastInferenceTime = 0L

    private data class DatasetStats(
        val accMean: Float,
        val accStd: Float,
        val gyroMean: Float,
        val gyroStd: Float
    )

    private val datasetStatsMap = mapOf(
        "UCI" to DatasetStats(
            accMean = 0.304977f,
            accStd = 0.525394f,
            gyroMean = -0.000896f,
            gyroStd = 0.348772f
        ),
        "MotionSense" to DatasetStats(
            accMean = 0.026396f,
            accStd = 0.418189f,
            gyroMean = 0.011328f,
            gyroStd = 1.125449f
        ),
        "HHAR" to DatasetStats(
            accMean = 0.962f,
            accStd = 3.662f,
            gyroMean = 0.003f,
            gyroStd = 2.449f
        ),
        "RealWorld" to DatasetStats(
            accMean = 0.183f,
            accStd = 2.836f,
            gyroMean = 0.024f,
            gyroStd = 2.897f
        ),
        "SHL" to DatasetStats(
            accMean = 0.063f,
            accStd = 1.587f,
            gyroMean = 0.008f,
            gyroStd = 1.812f
        ),
        "COMBINED" to DatasetStats(
            accMean = 0.294f,
            accStd = 2.173f,
            gyroMean = 0.007f,
            gyroStd = 1.942f
        )
    )

    // 新增平滑缓冲区
    private val predictionHistory = ArrayDeque<Pair<String, Float>>()
    private val smoothLock = Any()

    init {
        // 验证窗口参数有效性
        require(config.windowSize > 0) { "Window size must be positive" }
        require(config.slideStep > 0) { "Slide step must be positive" }
        require(config.slideStep <= config.windowSize) {
            "Slide step(${config.slideStep}) cannot exceed window size(${config.windowSize})"
        }

        // 打印关键参数（调试用）
        Log.i(TAG, """
            |Window params:
            |  Size: ${config.windowSize} 
            |  Slide: ${config.slideStep}
            |  Overlap: ${config.windowSize - config.slideStep}
        """.trimMargin())
    }

    /**
     * start():
     *  1) 加载 TFLite 模型
     *  2) 注册传感器监听（使用最快模式采集数据）
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

            synchronized(smoothLock) {
                predictionHistory.clear()
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
                    // 单位转换逻辑
                    val (accX, accY, accZ) = when (config.dataSetName.uppercase()) {
                        "UCI" -> Triple(x/9.80665f, y/9.80665f, z/9.80665f)
                        else -> Triple(x, y, z) // MotionSense 等数据集保持 m/s²
                    }
                    accBuffer.add(AccData(timestampNs, accX, accY, accZ))
                    Log.v("RAW_DATA", "Acc raw: $accX, $accY, $accZ (g)")
                }

                Sensor.TYPE_GYROSCOPE -> {
                    if (lastGyroTimestampNs < 0) lastGyroTimestampNs = timestampNs
                    gyroBuffer.add(GyroData(timestampNs, x, y, z))
                    Log.v("RAW_DATA", "Gyro raw: $x, $y, $z (rad/s)")
                }
            }

            // 当两个缓冲中的数据量超过 windowSize * 2 时触发推理
            if (accBuffer.size > config.windowSize * 1 && gyroBuffer.size > config.windowSize * 1) {
                processDataAndRunInference()
            }
        }
    }

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

        // 修改后的数据保留逻辑（基于 slideStep）
        synchronized(bufferLock) {
            localAcc = accBuffer.toList()
            localGyro = gyroBuffer.toList()

            // 根据 slideStep 计算保留样本数（转换为时间间隔）
            val keepSamples = max(0, config.windowSize - config.slideStep)

            if (accBuffer.size > keepSamples) {
                accBuffer.subList(0, accBuffer.size - keepSamples).clear()
            }
            if (gyroBuffer.size > keepSamples) {
                gyroBuffer.subList(0, gyroBuffer.size - keepSamples).clear()
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
            Log.d(
                TAG,
                "Unified resampling produced insufficient data: ${finalSegment.size} vs expected ${config.windowSize}"
            )
            return
        }

        // 4. 检查推理间隔时间
        val currentTime = System.currentTimeMillis()
        if (currentTime - lastInferenceTime < config.minTimeBetweenInferences) {
//            Log.d(TAG, "Inference too frequent; skip this round.")
            return
        }

        try {
            val prediction = runModelInference(finalSegment) ?: return
            processPrediction(prediction)
            lastInferenceTime = currentTime
        } catch (e: Exception) {
            Log.e(TAG, "Inference error:", e)
            callback.onError(e.message ?: "Unknown error")
        }

        Log.d("RESAMPLED", finalSegment.joinToString {
            "t=${it.timestamp} acc(${it.accX},${it.accY},${it.accZ})"
        })
    }

    /**
     * runModelInference():
     * 使用Z-score标准化处理数据
     */
    private fun runModelInference(finalSegment: List<SensorDataAll>): Pair<String, Float>? {
        if (finalSegment.size != config.windowSize) {
            Log.w(
                TAG,
                "Segment size mismatch: got=${finalSegment.size}, needed=${config.windowSize}"
            )
            return null
        }

        // 构造模型输入，形状 [1, windowSize, 6]
        val modelInput = Array(1) { Array(config.windowSize) { FloatArray(6) } }

        for ((i, data) in finalSegment.withIndex()) {
            val accValues = floatArrayOf(data.accX, data.accY, data.accZ)
            val gyroValues = floatArrayOf(data.gyroX, data.gyroY, data.gyroZ)

            // 分别对加速度和陀螺仪数据进行标准化
            standardizeData(accValues, true)
            standardizeData(gyroValues, false)

//            Log.d(TAG, "Inference data: acc=${accValues.contentToString()}, gyro=${gyroValues.contentToString()}")

            // 填充到模型输入
            accValues.forEachIndexed { j, v -> modelInput[0][i][j] = v }
            gyroValues.forEachIndexed { j, v -> modelInput[0][i][j + 3] = v }
        }

        Log.d("MODEL_INPUT", modelInput[0].joinToString { it.contentToString() })

        // 模型输出缓冲，形状 [1, numLabels]
        val outputBuffer = Array(1) { FloatArray(activityLabels.size) }
        tfliteInterpreter?.run(modelInput, outputBuffer) ?: return null


        // 选取具有最大置信度的活动
        var maxIdx = 0
        var maxVal = outputBuffer[0][0]
        for (i in outputBuffer[0].indices) {
            Log.d(
                TAG,
                "Inference result: activity=${activityLabels[i]}, confidence=${outputBuffer[0][i]}"
            )
            if (outputBuffer[0][i] > maxVal) {
                maxVal = outputBuffer[0][i]
                maxIdx = i
            }
        }


        // 若最大置信度低于设定阈值，则忽略该推断结果
        if (maxVal < config.confidenceThreshold) {
            Log.d(
                TAG,
                "Confidence($maxVal) below threshold(${config.confidenceThreshold}). Discarding."
            )
            return null
        }

        val actName = activityLabels.getOrNull(maxIdx) ?: "Unknown"
        return Pair(actName, maxVal)
    }

    private fun processPrediction(prediction: Pair<String, Float>) {
        synchronized(smoothLock) {
            // 1. 维护固定大小的历史记录
            if (predictionHistory.size >= config.smoothWindowSize) {
                predictionHistory.removeFirst()
            }
            predictionHistory.addLast(prediction)

            // 2. 应用平滑策略
            val finalPrediction = when (config.smoothStrategy) {
                SmoothStrategy.MAJORITY_VOTE -> applyMajorityVote()
                SmoothStrategy.MOVING_AVERAGE -> applyMovingAverage()
                SmoothStrategy.EXPONENTIAL_SMOOTHING -> applyExponentialSmoothing()
            }

            // 3. 输出最终结果
            if (finalPrediction.second >= config.confidenceThreshold) {
                Log.d(TAG, "Smoothed activity: =>${finalPrediction.first}, confidence=${finalPrediction.second}")
                callback.onActivityRecognized(finalPrediction.first, finalPrediction.second)
            }
        }
    }

    // 多数投票策略
    private fun applyMajorityVote(): Pair<String, Float> {
        /**
         * 多数投票策略实现：
         * 1. 统计窗口期内各活动出现的频次
         * 2. 选择出现次数最多的活动
         * 3. 置信度计算为：出现次数 / 窗口大小
         */
        val frequencyMap = mutableMapOf<String, Int>()
        predictionHistory.forEach {
            frequencyMap[it.first] = frequencyMap.getOrDefault(it.first, 0) + 1
        }
        val maxEntry = frequencyMap.maxByOrNull { it.value }!!
        return Pair(maxEntry.key, maxEntry.value.toFloat() / predictionHistory.size)
    }

    // 滑动平均策略
    private fun applyMovingAverage(): Pair<String, Float> {
        /**
         * 滑动平均策略实现：
         * 1. 累加各活动的置信度得分
         * 2. 选择总分最高的活动
         * 3. 最终置信度为：总分 / 窗口大小
         */
        val scoreMap = mutableMapOf<String, Float>()
        predictionHistory.forEach {
            scoreMap[it.first] = scoreMap.getOrDefault(it.first, 0f) + it.second
        }
        val maxEntry = scoreMap.maxByOrNull { it.value }!!
        return Pair(maxEntry.key, maxEntry.value / predictionHistory.size)
    }

    // 指数平滑策略（α=0.5）
    private fun applyExponentialSmoothing(): Pair<String, Float> {
        /**
         * 指数平滑策略实现（α=0.5）：
         * 1. 权重系数按时间指数衰减：w_i = 0.5^(N-i)
         * 2. 计算各活动的加权总分
         * 3. 最终置信度为：加权总分 / 总权重
         */
        val weights = List(predictionHistory.size) { Math.pow(0.5, (predictionHistory.size - it).toDouble()).toFloat() }
        val totalWeight = weights.sum()

        val weightedScores = mutableMapOf<String, Float>()
        predictionHistory.forEachIndexed { index, (activity, score) ->
            weightedScores[activity] = weightedScores.getOrDefault(activity, 0f) + score * weights[index]
        }

        val maxEntry = weightedScores.maxByOrNull { it.value }!!
        return Pair(maxEntry.key, maxEntry.value / totalWeight)
    }

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
        val commonTimestamps = generateSequence(commonStart) { it + config.slideStep * 1_000_000L }
            .takeWhile { it <= commonEnd }
            .take(config.windowSize)
            .toList()

        // 修复后的时间戳不足处理逻辑
        if (commonTimestamps.size < config.windowSize) {
            val missingCount = config.windowSize - commonTimestamps.size
            val adjustedTimestamps = commonTimestamps.toMutableList()

            // 生成补充时间戳（保持原有逻辑）
            var current = commonEnd + config.slideStep * 1_000_000L
            repeat(missingCount) {
                adjustedTimestamps.add(current)
                current += config.slideStep * 1_000_000L
            }

            // 对全部时间戳进行插值处理（新增）
            return adjustedTimestamps.map { t ->
                SensorDataAll(
                    t,
                    interpolateAcc(sortedAcc, t).let { it.x },  // 加速度X
                    interpolateAcc(sortedAcc, t).let { it.y },  // 加速度Y
                    interpolateAcc(sortedAcc, t).let { it.z },  // 加速度Z
                    interpolateGyro(sortedGyro, t).let { it.x }, // 陀螺仪X
                    interpolateGyro(sortedGyro, t).let { it.y }, // 陀螺仪Y
                    interpolateGyro(sortedGyro, t).let { it.z }  // 陀螺仪Z
                )
            }
        }

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

    @Throws(IOException::class)
    private fun loadTfliteModel() {
        val modelPath = getModelFileName(config.dataSetName)
        val buffer: MappedByteBuffer = FileUtil.loadMappedFile(context, modelPath)
        val options = Interpreter.Options().apply {
            numThreads = 2
        }
        tfliteInterpreter = Interpreter(buffer, options)
        Log.i(TAG, "TFLite model loaded from: $modelPath")
    }

    private fun standardizeData(values: FloatArray, isAcc: Boolean) {
        val stats = datasetStatsMap[config.dataSetName] ?: return
        val (mean, std) = if (isAcc) {
            Pair(stats.accMean, stats.accStd)
        } else {
            Pair(stats.gyroMean, stats.gyroStd)
        }

        Log.d("STANDARDIZE", "Raw: ${values.contentToString()} -> Mean: $mean, Std: $std")

        for (i in values.indices) {
            values[i] = (values[i] - mean) / std
        }

        Log.d("STANDARDIZE", "Result: ${values.contentToString()}")
    }
}

package com.zenbyte.core.ml

import java.io.InputStream
import java.nio.ByteBuffer
import java.nio.ByteOrder
import kotlin.math.exp
import kotlin.math.ln
import kotlin.math.sqrt

/**
 * TF-IDF(char_wb) + 선형 SVM 추론의 순수 Kotlin 구현 (exp-0011).
 *
 * exp-0010의 제약 하 벤치마크에서 fastText를 성능(F1 0.805 vs 0.788)과
 * 크기(3.08MB vs 14.04MB) 양쪽에서 능가하여 배포 모델로 채택되었다.
 *
 * scikit-learn 파이프라인을 그대로 재현한다:
 *   소문자화 → 연속 공백 정규화 → 단어 경계 char n-gram → 어휘 조회
 *   → sublinear tf (1+ln c) × idf → L2 정규화 → 선형 결정함수 → sigmoid
 *
 * 네이티브 라이브러리 없음. 입력 문장과 추론 결과는 어디에도 저장하지 않는다 (Zero-Persistence).
 * 동치성은 TfidfSvmClassifierTest(테스트 벡터 1,000문장, 오차 < 1e-4)로 자동 검증된다.
 */
class TfidfSvmClassifier private constructor(
    private val minN: Int,
    private val maxN: Int,
    private val sublinearTf: Boolean,
    val threshold: Float,
    private val intercept: Float,
    private val coefScale: Float,
    private val idf: FloatArray,
    private val coefQ: ByteArray,
    private val vocab: HashMap<String, Int>,
) {
    companion object {
        fun load(stream: InputStream): TfidfSvmClassifier {
            val buf = ByteBuffer.wrap(stream.readBytes()).order(ByteOrder.LITTLE_ENDIAN)
            require(buf.int == 0x5653425A) { "Not a ZBSV file" } // 'ZBSV' little-endian
            val version = buf.int
            require(version == 1) { "Unsupported ZBSV version $version" }
            val minN = buf.int
            val maxN = buf.int
            val nTerms = buf.int
            val sublinear = buf.get().toInt() != 0
            buf.get()                       // useIdf (항상 1)
            buf.short                       // padding
            val threshold = buf.float
            val intercept = buf.float
            val coefScale = buf.float

            val idf = FloatArray(nTerms)
            buf.asFloatBuffer().get(idf)
            buf.position(buf.position() + nTerms * 4)
            val coefQ = ByteArray(nTerms)
            buf.get(coefQ)

            val vocab = HashMap<String, Int>(nTerms * 2)
            val scratch = ByteArray(1024)
            for (i in 0 until nTerms) {
                val len = buf.short.toInt() and 0xFFFF
                buf.get(scratch, 0, len)
                vocab[String(scratch, 0, len, Charsets.UTF_8)] = i
            }
            return TfidfSvmClassifier(minN, maxN, sublinear, threshold, intercept,
                coefScale, idf, coefQ, vocab)
        }

        /** sklearn `_white_spaces = re.compile(r"\s\s+")` — **2개 이상** 연속 공백만 단일 공백으로. */
        private val MULTI_SPACE = Regex("\\s\\s+")
    }

    /**
     * sklearn `_char_wb_ngrams` 재현.
     * 단어를 공백으로 패딩한 뒤 슬라이딩 윈도우로 n-gram을 뽑되,
     * 패딩된 단어가 n보다 짧으면 전체를 1회만 세고 n 루프를 끝낸다.
     */
    private fun forEachNgram(text: String, action: (String) -> Unit) {
        val normalized = MULTI_SPACE.replace(text.lowercase(), " ")
        for (raw in normalized.split(' ', '\t', '\n', '\r')) {
            if (raw.isEmpty()) continue
            val w = " $raw "
            val wLen = w.length
            for (n in minN..maxN) {
                var offset = 0
                action(w.substring(offset, minOf(offset + n, wLen)))
                while (offset + n < wLen) {
                    offset++
                    action(w.substring(offset, offset + n))
                }
                if (offset == 0) break // 짧은 단어는 1회만 세고 종료
            }
        }
    }

    /** 선형 결정함수 값 (부호가 판정, 크기가 확신도). */
    private fun decision(text: String): Float {
        val counts = HashMap<Int, Int>()
        forEachNgram(text) { ng ->
            val idx = vocab[ng]
            if (idx != null) counts[idx] = (counts[idx] ?: 0) + 1
        }
        if (counts.isEmpty()) return intercept

        // tf-idf 값 계산 → L2 정규화 → 계수와 내적
        var normSq = 0.0
        val values = DoubleArray(counts.size)
        val indices = IntArray(counts.size)
        var i = 0
        for ((idx, c) in counts) {
            val tf = if (sublinearTf) 1.0 + ln(c.toDouble()) else c.toDouble()
            val v = tf * idf[idx]
            values[i] = v
            indices[i] = idx
            normSq += v * v
            i++
        }
        val norm = sqrt(normSq)
        if (norm == 0.0) return intercept

        var acc = 0.0
        for (k in indices.indices) {
            acc += (values[k] / norm) * (coefQ[indices[k]] * coefScale)
        }
        return (acc + intercept).toFloat()
    }

    /** P(독성) ∈ [0,1]. 입력·결과 모두 호출자 스택 밖에 잔류하지 않는다. */
    fun probToxic(text: String): Float = (1.0 / (1.0 + exp(-decision(text).toDouble()))).toFloat()

    /** 전송 전 경고 판정 (임계값은 모델 파일에 내장, val 셋에서 선택됨). */
    fun isToxic(text: String): Boolean = probToxic(text) >= threshold
}

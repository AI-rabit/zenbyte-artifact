package com.zenbyte.core.ml

import java.io.InputStream
import java.nio.ByteBuffer
import java.nio.ByteOrder
import kotlin.math.exp
import kotlin.math.ln
import kotlin.math.sqrt

/**
 * Pure-Kotlin implementation of TF-IDF (char_wb) + linear SVM inference
 * (exp-0011).
 *
 * In the constrained benchmark of exp-0010 this model beat fastText on both
 * accuracy (F1 0.805 vs 0.788) and size (3.08MB vs 14.04MB), and was adopted as
 * the deployed model.
 *
 * It reproduces the scikit-learn pipeline exactly:
 *   lowercase → collapse runs of whitespace → word-boundary char n-grams →
 *   vocabulary lookup → sublinear tf (1+ln c) × idf → L2 normalization →
 *   linear decision function → sigmoid
 *
 * No native library. Neither the input sentence nor the inference result is
 * stored anywhere (zero-persistence). Equivalence is checked against the test
 * vectors bundled under equivalence/ (1,000 sentences, max Δp < 1e-4).
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
            buf.get()                       // useIdf (always 1)
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

        /** sklearn `_white_spaces = re.compile(r"\s\s+")` — collapses runs of **two or more** spaces into one. */
        private val MULTI_SPACE = Regex("\\s\\s+")
    }

    /**
     * Reproduces sklearn's `_char_wb_ngrams`.
     * Pads each word with spaces and slides a window over it, except that when
     * the padded word is shorter than n, the whole word is counted once and the
     * n loop ends.
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
                if (offset == 0) break // a short word is counted once, then we stop
            }
        }
    }

    /** The linear decision value (its sign is the verdict, its magnitude the confidence). */
    private fun decision(text: String): Float {
        val counts = HashMap<Int, Int>()
        forEachNgram(text) { ng ->
            val idx = vocab[ng]
            if (idx != null) counts[idx] = (counts[idx] ?: 0) + 1
        }
        if (counts.isEmpty()) return intercept

        // compute tf-idf values → L2 normalize → dot with the coefficients
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

    /** P(toxic) ∈ [0,1]. Neither the input nor the result outlives the caller's stack. */
    fun probToxic(text: String): Float = (1.0 / (1.0 + exp(-decision(text).toDouble()))).toFloat()

    /** The pre-send warning verdict (the threshold is embedded in the model file, selected on the val set). */
    fun isToxic(text: String): Boolean = probToxic(text) >= threshold
}

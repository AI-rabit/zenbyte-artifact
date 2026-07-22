package com.zenbyte.data.repository

import android.content.Context
import com.zenbyte.core.ml.TfidfSvmClassifier
import com.zenbyte.domain.repository.ToxicityRepository
import dagger.hilt.android.qualifiers.ApplicationContext
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext
import javax.inject.Inject
import javax.inject.Singleton
import com.zenbyte.core.SafeLog

/**
 * Runs on-device inference by lazily loading the ZBSV model (TF-IDF char
 * n-grams + linear SVM) from assets.
 *
 * exp-0011: in the constrained benchmark (exp-0010) this model beat fastText on
 * both accuracy (F1 0.805 vs 0.788) and size (3.08MB vs 14.04MB), and so
 * replaced it as the deployed model.
 *
 * The model is loaded once on the first check and stays resident in RAM for the
 * lifetime of the process (about 4MB). Neither the inspected sentence nor the
 * verdict is stored, and if loading fails the feature disables itself quietly
 * rather than interfering with sending.
 */
@Singleton
class DefaultToxicityRepository @Inject constructor(
    @ApplicationContext private val context: Context,
) : ToxicityRepository {

    @Volatile
    private var classifier: TfidfSvmClassifier? = null

    @Volatile
    private var loadFailed = false

    private suspend fun classifier(): TfidfSvmClassifier? {
        classifier?.let { return it }
        if (loadFailed) return null
        return withContext(Dispatchers.IO) {
            synchronized(this@DefaultToxicityRepository) {
                classifier ?: runCatching {
                    context.assets.open(MODEL_ASSET).use { TfidfSvmClassifier.load(it) }
                }.onSuccess { classifier = it }
                    .onFailure {
                        loadFailed = true
                        SafeLog.w(TAG, "toxicity model failed to load — feature disabled")
                    }
                    .getOrNull()
            }
        }
    }

    override suspend fun isToxic(text: String): Boolean {
        if (text.isBlank()) return false
        val model = classifier() ?: return false
        return withContext(Dispatchers.Default) { model.isToxic(text) }
    }

    private companion object {
        const val MODEL_ASSET = "toxicity_model.zbsv"
        const val TAG = "ToxicityRepository"
    }
}

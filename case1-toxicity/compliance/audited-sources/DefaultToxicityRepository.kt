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
 * assets의 ZBSV 모델(TF-IDF char n-gram + 선형 SVM)을 지연 로드해 온디바이스 추론을 수행한다.
 *
 * exp-0011: 제약 하 벤치마크(exp-0010) 결과 fastText를 성능(F1 0.805 vs 0.788)과
 * 크기(3.08MB vs 14.04MB) 양쪽에서 능가하여 배포 모델로 교체되었다.
 *
 * 모델은 첫 검사 시 1회 로드되어 프로세스 수명 동안 RAM에 상주한다 (약 4MB).
 * 검사 문장·결과는 저장하지 않으며, 로드 실패 시 조용히 비활성화되어 전송을 방해하지 않는다.
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
                        SafeLog.w(TAG, "독성 탐지 모델 로드 실패 — 기능 비활성화")
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

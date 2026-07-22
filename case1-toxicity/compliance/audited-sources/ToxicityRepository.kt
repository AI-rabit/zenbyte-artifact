package com.zenbyte.domain.repository

/**
 * On-device toxic-language detection (exp-0004).
 *
 * Zero-persistence contract: an implementation leaves neither the inspected
 * sentence nor the verdict anywhere — not on the network, not on disk, not in
 * logs. Inference begins and ends in device RAM.
 */
interface ToxicityRepository {
    /** True when the sentence is judged toxic. False when no model is bundled or loading failed (never blocks sending). */
    suspend fun isToxic(text: String): Boolean
}

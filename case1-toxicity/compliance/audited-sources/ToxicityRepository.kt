package com.zenbyte.domain.repository

/**
 * 온디바이스 독성 표현 탐지 (exp-0004).
 *
 * Zero-Persistence 계약: 구현체는 검사 대상 문장과 판정 결과를 어디에도(네트워크·디스크·로그)
 * 남기지 않는다. 추론은 전적으로 기기 RAM 안에서 끝난다.
 */
interface ToxicityRepository {
    /** 문장이 독성으로 판정되면 true. 모델 미탑재/로드 실패 시 false (전송을 막지 않는다). */
    suspend fun isToxic(text: String): Boolean
}

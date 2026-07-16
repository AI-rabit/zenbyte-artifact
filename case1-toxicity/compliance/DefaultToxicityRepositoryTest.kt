package com.zenbyte.data.repository

import android.content.Context
import androidx.test.core.app.ApplicationProvider
import java.io.File
import kotlinx.coroutines.runBlocking
import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertTrue
import org.junit.Before
import org.junit.Test
import org.junit.runner.RunWith
import org.robolectric.RobolectricTestRunner
import org.robolectric.annotation.Config

/**
 * exp-0004 런타임 무기록 증명 (디스크 레벨).
 *
 * 추론을 반복 실행한 전후로 앱 데이터 디렉토리(filesDir, cacheDir, shared_prefs)의
 * 파일 목록·크기·내용이 **바이트 단위로 불변**임을 확인한다. 즉 검사 문장도, 판정 결과도,
 * 어떤 캐시도 디스크에 떨어지지 않는다.
 */
@RunWith(RobolectricTestRunner::class)
@Config(sdk = [34], application = android.app.Application::class)
class DefaultToxicityRepositoryTest {

    private lateinit var context: Context
    private lateinit var repository: DefaultToxicityRepository

    @Before
    fun setup() {
        context = ApplicationProvider.getApplicationContext()
        repository = DefaultToxicityRepository(context)
    }

    /** 데이터 디렉토리 스냅샷: 상대경로 → (크기, 내용 해시) */
    private fun snapshot(): Map<String, Pair<Long, Int>> {
        val roots = listOfNotNull(context.filesDir, context.cacheDir, context.dataDir)
        return roots.flatMap { root ->
            root.walkTopDown().filter { it.isFile }.map { f: File ->
                f.absolutePath to (f.length() to f.readBytes().contentHashCode())
            }
        }.toMap()
    }

    @Test
    fun `추론 전후 앱 데이터 디렉토리가 바이트 단위로 불변이다`() = runBlocking {
        val before = snapshot()

        repeat(3) {
            repository.isToxic("ㅅㅂ 개같은 상황이네")
            repository.isToxic("좋은 아침입니다. 오늘 회의는 10시입니다")
            repository.isToxic("")
        }

        val after = snapshot()
        val added = after.keys - before.keys
        val removed = before.keys - after.keys
        val changed = before.keys.intersect(after.keys).filter { before[it] != after[it] }

        assertTrue("추론이 파일을 생성함: $added", added.isEmpty())
        assertTrue("추론이 파일을 삭제함: $removed", removed.isEmpty())
        assertTrue("추론이 파일을 변경함: $changed", changed.isEmpty())
    }

    @Test
    fun `모델이 실제로 독성 문장과 정상 문장을 구분한다`() = runBlocking {
        // 동작점 임계값(0.375) 기준. 회귀 시 즉시 감지되도록 대표 문장만 확인한다.
        assertTrue("독성 문장을 놓침", repository.isToxic("ㅅㅂ 개같은 상황이네"))
        assertFalse("정상 문장을 오탐", repository.isToxic("좋은 아침입니다. 오늘 회의는 10시입니다"))
    }

    @Test
    fun `빈 문장은 검사하지 않는다`() = runBlocking {
        assertFalse(repository.isToxic(""))
        assertFalse(repository.isToxic("   "))
    }
}

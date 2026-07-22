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
 * exp-0004 runtime zero-persistence proof (disk level).
 *
 * Confirms that the file listing, sizes and contents of the app's data
 * directories (filesDir, cacheDir, shared_prefs) are **byte-for-byte unchanged**
 * before and after inference is run repeatedly. That is: neither the inspected
 * sentence, nor the verdict, nor any cache reaches the disk.
 *
 * The Korean sentences below are test inputs, not prose — the classifier is a
 * Korean-language model, so they are kept as-is.
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

    /** Snapshot of the data directories: relative path → (size, content hash) */
    private fun snapshot(): Map<String, Pair<Long, Int>> {
        val roots = listOfNotNull(context.filesDir, context.cacheDir, context.dataDir)
        return roots.flatMap { root ->
            root.walkTopDown().filter { it.isFile }.map { f: File ->
                f.absolutePath to (f.length() to f.readBytes().contentHashCode())
            }
        }.toMap()
    }

    @Test
    fun `app data directories are byte-identical before and after inference`() = runBlocking {
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

        assertTrue("inference created files: $added", added.isEmpty())
        assertTrue("inference deleted files: $removed", removed.isEmpty())
        assertTrue("inference modified files: $changed", changed.isEmpty())
    }

    @Test
    fun `the model actually separates toxic from ordinary sentences`() = runBlocking {
        // Judged at the deployed threshold (0.475). Only representative sentences
        // are checked, so that a regression shows up immediately.
        assertTrue("missed a toxic sentence", repository.isToxic("ㅅㅂ 개같은 상황이네"))
        assertFalse("flagged an ordinary sentence", repository.isToxic("좋은 아침입니다. 오늘 회의는 10시입니다"))
    }

    @Test
    fun `blank sentences are not inspected`() = runBlocking {
        assertFalse(repository.isToxic(""))
        assertFalse(repository.isToxic("   "))
    }
}

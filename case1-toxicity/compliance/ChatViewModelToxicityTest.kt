package com.zenbyte.presentation.viewmodel

import androidx.lifecycle.SavedStateHandle
import com.zenbyte.domain.model.User
import com.zenbyte.domain.repository.CryptoRepository
import com.zenbyte.domain.repository.ToxicityRepository
import com.zenbyte.domain.repository.UserRepository
import com.zenbyte.domain.usecase.SendMessageUseCase
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.ExperimentalCoroutinesApi
import kotlinx.coroutines.flow.emptyFlow
import kotlinx.coroutines.test.StandardTestDispatcher
import kotlinx.coroutines.test.advanceUntilIdle
import kotlinx.coroutines.test.resetMain
import kotlinx.coroutines.test.runTest
import kotlinx.coroutines.test.setMain
import org.junit.After
import org.junit.Assert.assertEquals
import org.junit.Assert.assertNull
import org.junit.Before
import org.junit.Test
import org.junit.runner.RunWith
import org.mockito.kotlin.any
import org.mockito.kotlin.doReturn
import org.mockito.kotlin.eq
import org.mockito.kotlin.mock
import org.mockito.kotlin.never
import org.mockito.kotlin.stub
import org.mockito.kotlin.verify
import org.mockito.kotlin.whenever
import org.robolectric.RobolectricTestRunner
import org.robolectric.annotation.Config

/**
 * exp-0004 runtime zero-persistence proof (send-path level).
 *
 * What it proves:
 *  1) when a sentence is judged toxic, **nothing goes to the network** before the
 *     user confirms (SendMessageUseCase is never invoked)
 *  2) if the user chooses "send anyway", the **original text alone** is sent —
 *     no verdict and no probability rides along in the payload or the metadata
 *  3) on dismiss, the pending sentence is destroyed at once (nothing left in RAM)
 *
 * The Korean strings are test payloads and are kept as-is: they exercise
 * multibyte UTF-8 round-tripping through the send path, which an ASCII
 * placeholder would not. ("욕설이 담긴 문장" = "a sentence containing profanity",
 * "좋은 아침입니다" = "good morning", "친구" = "friend".)
 */
@OptIn(ExperimentalCoroutinesApi::class)
@RunWith(RobolectricTestRunner::class)
@Config(sdk = [34])
class ChatViewModelToxicityTest {

    private val dispatcher = StandardTestDispatcher()

    private lateinit var sendMessageUseCase: SendMessageUseCase
    private lateinit var toxicityRepository: ToxicityRepository
    private lateinit var userRepository: UserRepository
    private lateinit var cryptoRepository: CryptoRepository

    private val partner = User(id = "partner-key", alias = "친구")

    @Before
    fun setup() {
        Dispatchers.setMain(dispatcher)
        sendMessageUseCase = mock()
        toxicityRepository = mock()
        cryptoRepository = mock()
        userRepository = mock {
            onBlocking { findUserById(any()) } doReturn partner
        }
    }

    @After
    fun tearDown() = Dispatchers.resetMain()

    private fun viewModel(): ChatViewModel {
        val panelRepo = mock<com.zenbyte.domain.repository.TacticalPanelRepository> {
            on { getPanels(any()) } doReturn emptyFlow()
        }
        return ChatViewModel(
            application = mock<com.zenbyte.ZenbyteApplication>(),
            sendMessageUseCase = sendMessageUseCase,
            sendGroupMessageUseCase = mock(),
            messageDispatchQueue = mock(),
            cryptoRepository = cryptoRepository,
            userRepository = userRepository,
            groupRepository = mock(),
            blockedGroupRepository = mock(),
            getSubscriptionStatusUseCase = mock(),
            sendReadReceiptUseCase = mock(),
            reportUserUseCase = mock(),
            createTacticalPanelUseCase = mock(),
            updateTacticalPanelUseCase = mock(),
            syncTacticalPanelUseCase = mock(),
            tacticalPanelRepository = panelRepo,
            blockListSyncManager = mock(),
            toxicityRepository = toxicityRepository,
            json = kotlinx.serialization.json.Json,
            savedStateHandle = SavedStateHandle(mapOf("chatId" to "partner-key", "chatType" to "DIRECT")),
        )
    }

    @Test
    fun `nothing reaches the network before the user confirms a toxic verdict`() = runTest(dispatcher) {
        toxicityRepository.stub { onBlocking { isToxic(any()) } doReturn true }
        val vm = viewModel()

        vm.sendMessage("욕설이 담긴 문장")
        advanceUntilIdle()

        verify(sendMessageUseCase, never()).invoke(any(), any(), any(), any())
        assertEquals("욕설이 담긴 문장", vm.toxicWarningText.value)
    }

    @Test
    fun `send anyway transmits the original text only - the verdict does not ride along`() = runTest(dispatcher) {
        toxicityRepository.stub { onBlocking { isToxic(any()) } doReturn true }
        val vm = viewModel()

        vm.sendMessage("욕설이 담긴 문장")
        advanceUntilIdle()
        vm.sendPendingMessageAnyway()
        advanceUntilIdle()

        // original bytes only, type "text", metadata null — there is no field a
        // toxicity score or verdict could occupy
        verify(sendMessageUseCase).invoke(
            eq("욕설이 담긴 문장".toByteArray(Charsets.UTF_8)),
            eq(partner),
            eq("text"),
            eq(null),
        )
        assertNull(vm.toxicWarningText.value) // pending sentence destroyed at once
    }

    @Test
    fun `dismissing sends nothing and destroys the pending sentence`() = runTest(dispatcher) {
        toxicityRepository.stub { onBlocking { isToxic(any()) } doReturn true }
        val vm = viewModel()

        vm.sendMessage("욕설이 담긴 문장")
        advanceUntilIdle()
        vm.dismissToxicWarning()
        advanceUntilIdle()

        verify(sendMessageUseCase, never()).invoke(any(), any(), any(), any())
        assertNull(vm.toxicWarningText.value)
    }

    @Test
    fun `an ordinary sentence is sent straight through without a warning`() = runTest(dispatcher) {
        toxicityRepository.stub { onBlocking { isToxic(any()) } doReturn false }
        val vm = viewModel()

        vm.sendMessage("좋은 아침입니다")
        advanceUntilIdle()

        assertNull(vm.toxicWarningText.value)
        verify(sendMessageUseCase).invoke(any(), eq(partner), eq("text"), eq(null))
    }
}

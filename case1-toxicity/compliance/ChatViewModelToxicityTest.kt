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
 * exp-0004 런타임 무기록 증명 (전송 경로 레벨).
 *
 * 증명하려는 것:
 *  1) 독성 판정 시 사용자 확인 전까지 **네트워크 전송이 일어나지 않는다** (SendMessageUseCase 미호출)
 *  2) 사용자가 "그대로 보내기"를 택하면 **원문 그대로** 전송된다 — 판정 결과·확률이
 *     페이로드나 메타데이터에 실려 나가지 않는다
 *  3) 취소 시 대기 문장이 즉시 파기된다 (RAM에도 잔류 없음)
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
    fun `독성 판정 시 확인 전에는 네트워크로 나가지 않는다`() = runTest(dispatcher) {
        toxicityRepository.stub { onBlocking { isToxic(any()) } doReturn true }
        val vm = viewModel()

        vm.sendMessage("욕설이 담긴 문장")
        advanceUntilIdle()

        verify(sendMessageUseCase, never()).invoke(any(), any(), any(), any())
        assertEquals("욕설이 담긴 문장", vm.toxicWarningText.value)
    }

    @Test
    fun `그대로 보내기 선택 시 원문만 전송된다 - 판정 결과는 실리지 않는다`() = runTest(dispatcher) {
        toxicityRepository.stub { onBlocking { isToxic(any()) } doReturn true }
        val vm = viewModel()

        vm.sendMessage("욕설이 담긴 문장")
        advanceUntilIdle()
        vm.sendPendingMessageAnyway()
        advanceUntilIdle()

        // 원문 바이트만, 타입 "text", 메타데이터 null — 독성 점수/판정이 새어나갈 자리가 없다
        verify(sendMessageUseCase).invoke(
            eq("욕설이 담긴 문장".toByteArray(Charsets.UTF_8)),
            eq(partner),
            eq("text"),
            eq(null),
        )
        assertNull(vm.toxicWarningText.value) // 대기 문장 즉시 파기
    }

    @Test
    fun `취소 시 전송되지 않고 대기 문장이 파기된다`() = runTest(dispatcher) {
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
    fun `정상 문장은 경고 없이 곧바로 전송된다`() = runTest(dispatcher) {
        toxicityRepository.stub { onBlocking { isToxic(any()) } doReturn false }
        val vm = viewModel()

        vm.sendMessage("좋은 아침입니다")
        advanceUntilIdle()

        assertNull(vm.toxicWarningText.value)
        verify(sendMessageUseCase).invoke(any(), eq(partner), eq("text"), eq(null))
    }
}

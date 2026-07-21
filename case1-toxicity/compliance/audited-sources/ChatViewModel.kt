package com.zenbyte.presentation.viewmodel

import android.app.Application
import androidx.lifecycle.SavedStateHandle
import androidx.lifecycle.viewModelScope
import android.util.Base64
import com.zenbyte.domain.model.User
import com.zenbyte.ZenbyteApplication
import com.zenbyte.core.ErrorEvent
import com.zenbyte.core.secureWipe
import com.zenbyte.domain.model.ChatType
import com.zenbyte.domain.model.BlockedGroupInfo
import com.zenbyte.domain.model.Message
import com.zenbyte.domain.repository.CryptoRepository
import com.zenbyte.domain.repository.BlockedGroupRepository
import com.zenbyte.domain.repository.GroupRepository
import com.zenbyte.domain.repository.UserRepository
import com.zenbyte.domain.usecase.GetSubscriptionStatusUseCase
import com.zenbyte.domain.usecase.MessageDispatchQueue
import com.zenbyte.domain.usecase.SendGroupMessageUseCase
import com.zenbyte.domain.usecase.SendMessageUseCase
import com.zenbyte.domain.usecase.SendReadReceiptUseCase
import com.zenbyte.domain.usecase.ReportUserUseCase
import dagger.hilt.android.lifecycle.HiltViewModel
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.flow.update
import kotlinx.coroutines.launch
import javax.inject.Inject
import java.net.URLDecoder
import java.nio.charset.StandardCharsets
import android.Manifest
import android.content.pm.PackageManager
import com.zenbyte.presentation.util.MGRSConverter
import com.zenbyte.R

// ✅ Panel Imports
import com.zenbyte.domain.usecase.panel.CreateTacticalPanelUseCase
import com.zenbyte.domain.usecase.panel.UpdateTacticalPanelUseCase
import com.zenbyte.domain.usecase.panel.SyncTacticalPanelUseCase
import com.zenbyte.domain.repository.TacticalPanelRepository
import com.zenbyte.domain.model.TacticalPanelState
import kotlinx.coroutines.flow.SharingStarted
import kotlinx.coroutines.flow.stateIn
import kotlinx.serialization.json.Json
import kotlinx.serialization.encodeToString
import com.zenbyte.core.SafeLog

@HiltViewModel
class ChatViewModel @Inject constructor(
    private val application: Application, // Inject Application context
    private val sendMessageUseCase: SendMessageUseCase,
    private val sendGroupMessageUseCase: SendGroupMessageUseCase,
    private val messageDispatchQueue: MessageDispatchQueue, // ✅ Re-inject MessageDispatchQueue
    private val cryptoRepository: CryptoRepository,
    private val userRepository: UserRepository,
    private val groupRepository: GroupRepository,
    private val blockedGroupRepository: BlockedGroupRepository, // Inject new repository
    private val getSubscriptionStatusUseCase: GetSubscriptionStatusUseCase,
    private val sendReadReceiptUseCase: SendReadReceiptUseCase, // Inject SendReadReceiptUseCase
    private val reportUserUseCase: ReportUserUseCase,
    // ✅ Panel UseCases & Repo
    private val createTacticalPanelUseCase: CreateTacticalPanelUseCase,
    private val updateTacticalPanelUseCase: UpdateTacticalPanelUseCase,
    private val syncTacticalPanelUseCase: SyncTacticalPanelUseCase,
    private val tacticalPanelRepository: TacticalPanelRepository,
    private val blockListSyncManager: com.zenbyte.domain.usecase.BlockListSyncManager, // ✅ Inject BlockListSyncManager
    private val toxicityRepository: com.zenbyte.domain.repository.ToxicityRepository, // ✅ 온디바이스 독성 탐지
    private val json: Json,
    savedStateHandle: SavedStateHandle
) : BaseViewModel() { // Inherit from BaseViewModel

    val chatId: String = try {
        val rawId = savedStateHandle.get<String>("chatId")
        if (rawId != null) {
            URLDecoder.decode(rawId, StandardCharsets.UTF_8.toString())
        } else {
            // For direct chat (chat_direct route), get from Application
            (application as? ZenbyteApplication)?.selectedFriendId ?: ""
        }
    } catch (e: Exception) {
        savedStateHandle.get<String>("chatId") ?: ""
    }

    val chatType: ChatType = try {
        savedStateHandle.get<String>("chatType")?.let {
            ChatType.valueOf(it.uppercase())
        } ?: ChatType.DIRECT
    } catch (e: Exception) {
        ChatType.DIRECT
    }

    private val _chatTitle = MutableStateFlow("")
    val chatTitle: StateFlow<String> = _chatTitle.asStateFlow()

    private val _messages = MutableStateFlow<List<Message>>(emptyList())
    val messages: StateFlow<List<Message>> = _messages.asStateFlow()

    private val _isSending = MutableStateFlow(false)
    val isSending: StateFlow<Boolean> = _isSending.asStateFlow()

    private val _isGroupCreator = MutableStateFlow(false)
    val isGroupCreator: StateFlow<Boolean> = _isGroupCreator.asStateFlow()

    // ✅ Tactical Panels State (Observed by UI)
    val tacticalPanels: StateFlow<List<TacticalPanelState>> =
        tacticalPanelRepository.getPanels(chatId)
            .stateIn(viewModelScope, SharingStarted.WhileSubscribed(5000), emptyList())

    val myPublicKeyString: String by lazy {
        try {
            Base64.encodeToString(cryptoRepository.getPublicKey().encoded, Base64.NO_WRAP)
        } catch (e: Exception) {
            ""
        }
    }

    private val userAliasCache = mutableMapOf<String, String>()

    fun getAliasForUserId(userId: String): String {
        return userAliasCache.getOrElse(userId) { "알 수 없는 상대" }
    }

    private fun loadUserAlias(userId: String) {
        viewModelScope.launch {
            try {
                val user = userRepository.findUserById(userId)
                if (user != null) {
                    userAliasCache[userId] = user.alias
                }
            } catch (e: Exception) {
                // Ignore
            }
        }
    }

    init {
        viewModelScope.launch {
            if (chatId.isBlank()) {
                sendError(ErrorEvent.Snackbar("잘못된 채팅 정보입니다."))
                return@launch
            }

            try {
                // ✅ CRITICAL: Notify MessageDispatchQueue that UI is ready to process messages
                messageDispatchQueue.notifyUiReady()

                // Set this chat as currently open to prevent marking as unread
                (application as? ZenbyteApplication)?.setCurrentChat(chatId)

                when (chatType) {
                    ChatType.DIRECT -> {
                        val user = userRepository.findUserById(chatId)
                        _chatTitle.value = user?.alias ?: "알 수 없는 상대"
                        if (user != null) {
                            userAliasCache[user.id] = user.alias
                        }
                    }
                    ChatType.GROUP -> {
                        val group = groupRepository.getGroup(chatId)
                        _chatTitle.value = group?.name ?: "그룹 채팅"
                        // Check if current user is the group creator
                        _isGroupCreator.value = group?.creatorId == myPublicKeyString
                        // Group members are loaded when messages arrive
                    }
                    ChatType.READ_RECEIPT -> {
                        // Read receipts are not a chat type to be displayed in ChatViewModel
                        // This case should ideally not be reached for chatType
                        // If it is, it means an invalid chatType was passed.
                        sendError(ErrorEvent.Snackbar("잘못된 채팅 유형입니다."))
                    }
                }

                (application as? ZenbyteApplication)?.clearUnreadCount(chatId) // Mark current chat as read when entering

                // Clear blue read receipt indicator when entering chat (1:1 only)
                if (chatType == ChatType.DIRECT) {
                    (application as? ZenbyteApplication)?.clearReadReceiptFor(chatId)
                }

                // ✅ New: Observe messages from ZenbyteApplication's currentChatMessages
                (application as? ZenbyteApplication)?.currentChatMessages?.collect { newMessages ->
                    // ✅ Sync Tactical Panels
                    // Process ALL panel messages to reconstruct state accurately
                    newMessages.filter { 
                        it.type == "panel_create" || it.type == "panel_update" || it.type == "panel_delete"
                    }.forEach { panelMessage ->
                        syncTacticalPanelUseCase(panelMessage)
                    }

                    // ✅ Filter out panel messages from UI list
                    // System messages should not be displayed as chat bubbles
                    val displayableMessages = newMessages.filterNot { 
                        it.type == "panel_create" || it.type == "panel_update" || it.type == "panel_delete"
                    }

                    _messages.value = displayableMessages // Update local messages state

                    // Read receipt logic:
                    // Check the last message in the list. If it's new and from another user, send a read receipt.
                    val lastMessage = newMessages.lastOrNull()
                    if (chatType == ChatType.DIRECT && lastMessage != null && lastMessage.senderId != myPublicKeyString) {
                        // Ensure it's a text or image message, not a read receipt itself
                        if (lastMessage.type == "text" || lastMessage.type == "image") {
                            viewModelScope.launch {
                                kotlinx.coroutines.delay(1000L) // 1 second delay
                                sendReadReceiptUseCase(recipient = User(id = lastMessage.senderId, alias = ""))
                                SafeLog.d("ChatViewModel", "✅ Read receipt sent for message: ${lastMessage.id}")
                            }
                        }
                    }

                    // For group chats, extract and cache sender aliases from loaded messages
                    if (chatType == ChatType.GROUP) {
                        newMessages.forEach { message ->
                            val senderAlias = message.metadata?.get("senderAlias")
                            if (senderAlias != null && message.senderId.isNotEmpty()) {
                                userAliasCache[message.senderId] = senderAlias
                            }
                        }
                    }
                }

            } catch (e: Exception) {
                sendError(ErrorEvent.Snackbar("채팅 정보를 불러오는 데 실패했습니다: ${e.message}"))
            }
        }
    }

    /**
     * 독성 경고 대기 중인 문장 (null이면 경고 없음).
     * 경고는 안내일 뿐 차단이 아니다 — 사용자는 [sendPendingMessageAnyway]로 그대로 보낼 수 있다.
     * 검사 문장은 이 StateFlow(RAM) 밖으로 나가지 않으며, 전송/취소 즉시 비운다.
     */
    private val _toxicWarningText = MutableStateFlow<String?>(null)
    val toxicWarningText: StateFlow<String?> = _toxicWarningText.asStateFlow()

    /**
     * 전송 요청 진입점. 전송 직전 로컬(온디바이스) 추론으로 독성 여부를 검사하고,
     * 판정 시 경고를 띄운 뒤 사용자의 결정을 기다린다. 추론 결과는 서버로 보내지 않는다.
     */
    fun sendMessage(text: String) {
        if (_isSending.value) return

        viewModelScope.launch {
            if (toxicityRepository.isToxic(text)) {
                _toxicWarningText.value = text
                return@launch
            }
            dispatchMessage(text)
        }
    }

    /** 경고를 무시하고 그대로 전송한다 (사용자 최종 결정권). */
    fun sendPendingMessageAnyway() {
        val text = _toxicWarningText.value ?: return
        _toxicWarningText.value = null
        viewModelScope.launch { dispatchMessage(text) }
    }

    /** 경고 후 전송을 취소한다 (문장은 입력창에 남고, 대기 문장은 즉시 파기). */
    fun dismissToxicWarning() {
        _toxicWarningText.value = null
    }

    private fun dispatchMessage(text: String) {
        if (_isSending.value) return

        viewModelScope.launch {
            _isSending.value = true
            try {
                // Create the message object with appropriate metadata
                val metadata = when (chatType) {
                    ChatType.GROUP -> mapOf("groupId" to chatId)
                    ChatType.DIRECT -> mapOf("recipientId" to chatId) // 일대일: recipientId 포함
                    ChatType.READ_RECEIPT -> null // Read receipts don't need specific metadata here
                }

                val message = Message(
                    id = java.util.UUID.randomUUID(),
                    content = text.toByteArray(Charsets.UTF_8),
                    senderId = myPublicKeyString,
                    type = "text",
                    metadata = metadata,
                    timestamp = java.util.Date()
                )

                // Add to session cache (ZenbyteApplication will handle updating currentChatMessages)
                (application as? ZenbyteApplication)?.addMessageToCache(chatId, message)

                when (chatType) {
                    ChatType.DIRECT -> {
                        val partner = userRepository.findUserById(chatId) ?: throw IllegalStateException("User not found")

                        if (partner.id == myPublicKeyString) {
                            throw IllegalStateException("자기 자신에게 메시지를 보낼 수 없습니다!")
                        }

                        sendMessageUseCase(text.toByteArray(Charsets.UTF_8), partner, "text", null)
                    }
                    ChatType.GROUP -> {
                        sendGroupMessageUseCase(text, chatId)
                    }
                    ChatType.READ_RECEIPT -> {
                        // Should not happen, as READ_RECEIPT is not a type for sending user messages
                        sendError(ErrorEvent.Snackbar("읽음 확인 채팅 유형으로는 메시지를 보낼 수 없습니다."))
                    }
                }

                // Update last message time for friend list sorting
                (application as? ZenbyteApplication)?.updateLastMessageTime(chatId)
            } catch (e: Exception) {
                sendError(ErrorEvent.Snackbar("메시지 전송에 실패했습니다: ${e.message}"))
            } finally {
                _isSending.value = false
            }
        }
    }

    // ✅ Tactical Location Sharing
    fun sendCurrentLocation(context: android.content.Context) {
        if (androidx.core.content.ContextCompat.checkSelfPermission(
                context,
                Manifest.permission.ACCESS_FINE_LOCATION
            ) != PackageManager.PERMISSION_GRANTED
        ) {
            sendError(ErrorEvent.Snackbar("위치 권한이 필요합니다."))
            return
        }

        viewModelScope.launch {
            try {
                val fusedLocationClient = com.google.android.gms.location.LocationServices.getFusedLocationProviderClient(context)
                fusedLocationClient.lastLocation.addOnSuccessListener { location ->
                    if (location != null) {
                        val mgrsCoordinate = MGRSConverter.toMGRS(location.latitude, location.longitude)
                        val googleMapsLink = "https://www.google.com/maps/search/?api=1&query=${location.latitude},${location.longitude}"
                        val messageText = "📍 MGRS: $mgrsCoordinate\nGoogle Maps: $googleMapsLink"

                        // 2. Create Metadata
                        // ⚠️ [무기록 원칙] 좌표는 절대 metadata에 넣지 않는다 — WebSocket 봉투에서
                        // 암호화되는 것은 payload뿐이고 metadata는 평문으로 서버를 통과한다
                        // (오프라인 수신자면 pendingBuffer에 최대 24시간 평문 체류). 좌표는 위의
                        // messageText(암호화되는 본문)에만 실리고, 수신 측은 복호화된 본문에서 파싱한다.
                        // 발신자는 온디바이스 위치 API만 사용하며 외부(지도 서버 등) 접속 흔적을 남기지 않는다.
                        val metadata = when (chatType) {
                            ChatType.GROUP -> mapOf("groupId" to chatId)
                            ChatType.DIRECT -> mapOf("recipientId" to chatId)
                            else -> emptyMap()
                        }

                        // 3. Create Message Object
                        val message = Message(
                            id = java.util.UUID.randomUUID(),
                            content = messageText.toByteArray(Charsets.UTF_8), // 텍스트 메시지로 전송
                            senderId = myPublicKeyString,
                            type = "text", // New Type
                            metadata = metadata,
                            timestamp = java.util.Date()
                        )

                        // 4. Update UI & Cache
                        (application as? ZenbyteApplication)?.addMessageToCache(chatId, message)

                        // 5. Send Message
                        viewModelScope.launch {
                            try {
                                when (chatType) {
                                    ChatType.DIRECT -> {
                                        val partner = userRepository.findUserById(chatId)
                                        if (partner != null) {
                                            sendMessageUseCase(messageText.toByteArray(Charsets.UTF_8), partner, "text", metadata)
                                        }
                                    }
                                    ChatType.GROUP -> {
                                        val groupMetadata = metadata.toMutableMap()
                                        // senderAlias and creatorId are handled by SendGroupMessageUseCase
                                        sendGroupMessageUseCase(
                                            content = messageText.toByteArray(Charsets.UTF_8),
                                            groupId = chatId,
                                            type = "group_text",
                                            metadata = groupMetadata
                                        )
                                    }
                                    else -> {}
                                }
                                (application as? ZenbyteApplication)?.updateLastMessageTime(chatId)
                            } catch (e: Exception) {
                                sendError(ErrorEvent.Snackbar("위치 전송 실패: ${e.message}"))
                            }
                        }
                    } else {
                        sendError(ErrorEvent.Snackbar("위치 정보를 가져올 수 없습니다."))
                    }
                }.addOnFailureListener {
                    sendError(ErrorEvent.Snackbar("위치 조회 실패: ${it.message}"))
                }
            } catch (e: Exception) {
                sendError(ErrorEvent.Snackbar("위치 서비스 오류: ${e.message}"))
            }
        }
    }

    // ✅ Tactical Panel Functions
    fun createChecklist(title: String, items: List<String>) {
        viewModelScope.launch {
            try {
                createTacticalPanelUseCase(chatId, title, items)
            } catch (e: Exception) {
                sendError(ErrorEvent.Snackbar("체크리스트 생성 실패: ${e.message}"))
            }
        }
    }

    fun createRollCall() {
        viewModelScope.launch {
            try {
                createTacticalPanelUseCase.createRollCall(chatId)
            } catch (e: Exception) {
                sendError(ErrorEvent.Snackbar("출석부 생성 실패: ${e.message}"))
            }
        }
    }

    /**
     * 놓쳤을 수 있는 메시지를 상대방/그룹에게 다시 요청합니다.
     */
    fun requestResend() {
        viewModelScope.launch {
            try {
                val resendMessageText = application.applicationContext.getString(R.string.request_resend_message)
                val resendMessageBytes = resendMessageText.toByteArray(Charsets.UTF_8)

                // Create a message object to be displayed locally
                val message = Message(
                    id = java.util.UUID.randomUUID(),
                    content = resendMessageBytes,
                    senderId = myPublicKeyString,
                    type = "text",
                    metadata = when (chatType) {
                        ChatType.GROUP -> mapOf("groupId" to chatId)
                        else -> mapOf("recipientId" to chatId)
                    },
                    timestamp = java.util.Date()
                )

                // Add to local cache immediately to update UI
                (application as? ZenbyteApplication)?.addMessageToCache(chatId, message)

                // Send the message over the network
                when (chatType) {
                    ChatType.DIRECT -> {
                        val partner = userRepository.findUserById(chatId)
                        if (partner != null) {
                            sendMessageUseCase(resendMessageBytes, partner, "text", null)
                        }
                    }
                    ChatType.GROUP -> {
                        sendGroupMessageUseCase(resendMessageText, chatId)
                    }
                    else -> {
                        // READ_RECEIPT etc. are not applicable
                    }
                }
                sendError(ErrorEvent.Toast("메시지 재전송을 요청했습니다."))
            } catch (e: Exception) {
                sendError(ErrorEvent.Snackbar("요청에 실패했습니다: ${e.message}"))
            }
        }
    }

    fun togglePanelItem(panelId: String, itemId: String, isChecked: Boolean) {
        viewModelScope.launch {
            try {
                updateTacticalPanelUseCase(chatId, panelId, itemId, isChecked)
            } catch (e: Exception) {
                sendError(ErrorEvent.Snackbar("상태 업데이트 실패: ${e.message}"))
            }
        }
    }

    fun deletePanel(panelId: String) {
        viewModelScope.launch {
            try {
                // Construct JSON content: { "chatId": "...", "panelId": "..." }
                val deleteInfo = mapOf("chatId" to chatId, "panelId" to panelId)
                val contentJson = json.encodeToString(deleteInfo)
                val contentBytes = contentJson.toByteArray(Charsets.UTF_8)

                // Create local message for sync
                val message = Message(
                    id = java.util.UUID.randomUUID(),
                    content = contentBytes,
                    senderId = myPublicKeyString,
                    type = "panel_delete",
                    metadata = if (chatType == ChatType.GROUP) mapOf("groupId" to chatId) else null,
                    timestamp = java.util.Date()
                )

                // Add to local cache
                (application as? ZenbyteApplication)?.addMessageToCache(chatId, message)
                
                // Send to network
                if (chatType == ChatType.GROUP) {
                    val group = groupRepository.getGroup(chatId)
                    val metadata = mapOf("groupId" to chatId)
                    group?.members?.forEach { memberId ->
                        if (memberId != myPublicKeyString) {
                            val recipient = User(id = memberId, alias = "")
                            sendMessageUseCase(contentBytes, recipient, "panel_delete", metadata)
                        }
                    }
                } else {
                    val partner = userRepository.findUserById(chatId)
                    if (partner != null) {
                        val metadata = mapOf("recipientId" to chatId)
                        sendMessageUseCase(contentBytes, partner, "panel_delete", metadata)
                    }
                }
            } catch (e: Exception) {
                sendError(ErrorEvent.Snackbar("패널 삭제 실패: ${e.message}"))
            }
        }
    }

    fun reportUser(reason: String) {
        viewModelScope.launch {
            try {
                reportUserUseCase(chatId, reason)
                sendError(ErrorEvent.Snackbar("신고가 접수되었습니다."))
            } catch (e: Exception) {
                sendError(ErrorEvent.Snackbar("신고 접수에 실패했습니다: ${e.message}"))
            }
        }
    }

    fun isGroupCreator(): Boolean {
        return _isGroupCreator.value
    }

    fun leaveGroup(onGroupLeft: () -> Unit) {
        if (chatType != ChatType.GROUP) return
        viewModelScope.launch {
            try {
                val group = groupRepository.getGroup(chatId)
                if (group != null) {
                    val blockedInfo = BlockedGroupInfo(groupId = group.id, creatorId = group.creatorId)
                    blockedGroupRepository.add(blockedInfo)
                    blockListSyncManager.sync() // ✅ 차단 목록 동기화 (서버 사이드 필터링 적용)
                }
                // Delete group locally after adding to block list
                groupRepository.deleteGroup(chatId)
                onGroupLeft()
            } catch (e: Exception) {
                sendError(ErrorEvent.Snackbar("그룹을 나가는 데 실패했습니다: ${e.message}"))
            }
        }
    }

    /**
     * 그룹을 해체합니다 (방장만 가능)
     *
     * 동작:
     * 1. 모든 그룹 멤버에게 "group_disbanded" 메시지 전송
     * 2. 로컬에서 그룹 키 삭제
     * 3. 로컬에서 그룹 데이터 삭제
     * 4. 채팅 화면 종료
     */
    fun disbandGroup(onGroupDisbanded: () -> Unit) {
        if (chatType != ChatType.GROUP) return
        viewModelScope.launch {
            try {
                // 1. 그룹 정보 가져오기
                val group = groupRepository.getGroup(chatId) ?: run {
                    sendError(ErrorEvent.Snackbar("그룹을 찾을 수 없습니다"))
                    return@launch
                }

                // 2. 방장의 프로필 정보 가져오기
                val myProfile = userRepository.getMyProfile()
                val senderAlias = myProfile.alias

                // 3. "group_disbanded" 메시지 생성 및 전송
                val content = chatId.toByteArray(Charsets.UTF_8)

                // 모든 멤버에게 개별적으로 전송 (그룹 메시지와 동일한 metadata 형식)
                group.members.forEach { memberId ->
                    val recipient = User(id = memberId, alias = "")
                    sendMessageUseCase(
                        content = content,
                        to = recipient,
                        type = "group_disbanded",
                        metadata = mapOf(
                            "groupId" to chatId,
                            "senderAlias" to senderAlias,
                            "creatorId" to group.creatorId
                        )
                    )
                }

                // 4. 로컬에서 그룹 키 삭제
                cryptoRepository.deleteGroupKey(chatId)

                // 5. 로컬에서 그룹 삭제
                groupRepository.deleteGroup(chatId)

                // 6. 채팅 화면 종료
                onGroupDisbanded()

                SafeLog.i("ChatViewModel", "그룹 해체 완료: $chatId")
            } catch (e: Exception) {
                SafeLog.e("ChatViewModel", "그룹 해체 실패: ${e.message}", e)
                sendError(ErrorEvent.Snackbar("그룹 해체에 실패했습니다: ${e.message}"))
            }
        }
    }

    fun clearChatData() {
        _messages.value.forEach { message ->
            message.content.secureWipe()
        }
        _messages.value = emptyList() // Clear the list itself
    }

    @Suppress("UNUSED_PARAMETER")
    fun markMessageAsViewed(_id: String) {
        // This is a placeholder. In a real app, you would update the message state.
        // Since we are practicing zero-persistence, we might just update a transient state in memory.
    }

    /**
     * 메시지 각인 토글 (Zero-Persistence with deferred shredding)
     * 각인된 메시지는 채팅방을 나가도 파쇄되지 않고, 앱 종료 시까지 유지됨
     */
    fun toggleImprintMessage(messageId: java.util.UUID) {
        viewModelScope.launch {
            val app = application as? ZenbyteApplication ?: return@launch

            // 현재 각인 상태 확인
            val isCurrentlyImprinted = app.isMessageImprinted(chatId, messageId)

            if (isCurrentlyImprinted) {
                // 각인 해제
                app.unimprintMessage(chatId, messageId)
            } else {
                // 각인
                app.imprintMessage(chatId, messageId)
            }

            // UI 즉시 업데이트: 기존 리스트에서 해당 메시지만 업데이트 (중복 방지)
            _messages.update { currentMessages ->
                currentMessages.map { message ->
                    if (message.id == messageId) {
                        // 캐시에서 업데이트된 메시지 가져오기 (deepCopy된 객체)
                        app.getMessagesFromCache(chatId).find { it.id == messageId } ?: message
                    } else {
                        message
                    }
                }
            }

            SafeLog.d(
                "ChatViewModel",
                "📌 메시지 각인 토글: messageId=$messageId, isImprinted=${!isCurrentlyImprinted}"
            )
        }
    }

    override fun onCleared() {
        super.onCleared()
        // Clear current chat when leaving chat room
        (application as? ZenbyteApplication)?.setCurrentChat(null)

        // 각인 기능: 각인되지 않은 메시지만 파쇄
        // 각인된 메시지는 앱 종료 시까지 유지 (Zero-Persistence with deferred shredding)
        (application as? ZenbyteApplication)?.clearNonImprintedMessagesFromCache(chatId)
    }
}

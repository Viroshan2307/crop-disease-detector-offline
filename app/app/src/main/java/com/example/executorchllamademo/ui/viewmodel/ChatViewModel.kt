/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

package com.example.executorchllamademo.ui.viewmodel

import android.app.ActivityManager
import android.app.Application
import android.content.Context
import android.net.Uri
import android.util.Log
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateListOf
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.setValue
import androidx.lifecycle.AndroidViewModel
import com.example.executorchllamademo.BackendType
import com.example.executorchllamademo.AppSettings
import com.example.executorchllamademo.DemoSharedPreferences
import com.example.executorchllamademo.ETImage
import com.example.executorchllamademo.LeafClassifier
import com.example.executorchllamademo.ETLogging
import com.example.executorchllamademo.Message
import com.example.executorchllamademo.MessageType
import com.example.executorchllamademo.ModelConfiguration
import com.example.executorchllamademo.ModelType
import com.example.executorchllamademo.ModelUtils
import com.example.executorchllamademo.PromptFormat
import com.example.executorchllamademo.ModuleSettings
import com.google.gson.Gson
import com.google.gson.GsonBuilder
import com.google.gson.InstanceCreator
import com.google.gson.reflect.TypeToken
import org.json.JSONException
import org.json.JSONObject
import org.pytorch.executorch.ExecutorchRuntimeException
import org.pytorch.executorch.extension.llm.LlmCallback
import org.pytorch.executorch.extension.llm.LlmModule
import java.io.IOException
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.file.Files
import java.nio.file.Paths
import java.util.concurrent.Executor
import java.util.concurrent.Executors

class ChatViewModel(application: Application) : AndroidViewModel(application), LlmCallback {

    private val _messages = mutableStateListOf<Message>()
    val messages: List<Message> = _messages

    var inputText by mutableStateOf("")
    var isModelReady by mutableStateOf(false)
    var isGenerating by mutableStateOf(false)
    var adviceLang by mutableStateOf("en")
    var thinkMode by mutableStateOf(false)
    var showMediaSelector by mutableStateOf(false)
    var ramUsage by mutableStateOf("0 MB")
    var showMediaButtons by mutableStateOf(false)
    var supportsImageInput by mutableStateOf(false)
    var supportsAudioInput by mutableStateOf(false)

    // Thinking mode state: tracks whether we're inside a <think>...</think> block
    private var isInThinkingBlock = false

    private val _selectedImages = mutableStateListOf<Uri>()
    val selectedImages: List<Uri> = _selectedImages

    // Dialog states
    var showModelLoadErrorDialog by mutableStateOf(false)
    var modelLoadError by mutableStateOf("")

    // LoRA mode state
    var isLoraMode by mutableStateOf(false)
        private set
    var availableModels by mutableStateOf<List<ModelConfiguration>>(emptyList())
        private set
    var activeModelId by mutableStateOf("")
        private set

    // Map of loaded LlmModules by model ID for LoRA mode
    private val loadedModules = mutableMapOf<String, LlmModule>()

    private var module: LlmModule? = null
    private var resultMessage: Message? = null
    private var lastBubbleUpdateMs: Long = 0L
    private val demoSharedPreferences = DemoSharedPreferences(application)
    private var currentSettingsFields = ModuleSettings()
    private var appSettings = AppSettings()
    private var promptID = 0
    private var sawStartHeaderId = false
    private var audioFileToPrefill: String? = null
    private var shouldAddSystemPrompt = true

    private val executor: Executor = Executors.newSingleThreadExecutor()
    private val contentResolver = application.contentResolver

    init {
        // Check for clear chat history flag BEFORE loading saved messages
        val moduleSettings = demoSharedPreferences.getModuleSettings()
        appSettings = demoSharedPreferences.getAppSettings()
        if (moduleSettings.isClearChatHistory) {
            // Clear the flag and don't load messages
            // Keep isLoadModel flag so model still loads in checkAndLoadSettings()
            demoSharedPreferences.removeExistingMessages()
            demoSharedPreferences.saveModuleSettings(moduleSettings.copy(isClearChatHistory = false))
            // Don't update currentSettingsFields here - let checkAndLoadSettings() handle it
            // so it detects the change and loads the model if needed
        } else {
            loadSavedMessages()
        }
    }

    private fun loadSavedMessages() {
        val appSettings = demoSharedPreferences.getAppSettings()
        // Only load saved messages if saveChatHistory is enabled
        if (!appSettings.saveChatHistory) {
            // Clear any existing saved messages since saving is disabled
            demoSharedPreferences.removeExistingMessages()
            return
        }

        val existingMsgJSON = demoSharedPreferences.getSavedMessages()
        if (existingMsgJSON.isNotEmpty()) {
            // Use InstanceCreator so that Gson calls the Message constructor (which
            // assigns default values like id = UUID). Without this, Gson uses
            // Unsafe.allocateInstance() and fields missing from old JSON become null.
            val gson = GsonBuilder()
                .registerTypeAdapter(Message::class.java, InstanceCreator<Message> {
                    Message("", false, MessageType.TEXT, 0)
                })
                .create()
            val type = object : TypeToken<ArrayList<Message>>() {}.type
            val savedMessages: ArrayList<Message>? = gson.fromJson(existingMsgJSON, type)
            savedMessages?.let {
                _messages.addAll(it)
                promptID = _messages.maxOfOrNull { msg -> msg.promptID }?.plus(1) ?: 0
            }
        }
    }

    fun saveMessages() {
        val appSettings = demoSharedPreferences.getAppSettings()
        // Only save messages if saveChatHistory is enabled
        if (appSettings.saveChatHistory) {
            demoSharedPreferences.addMessages(_messages.toList())
        } else {
            // Make sure no messages are persisted
            demoSharedPreferences.removeExistingMessages()
        }
    }

    private val systemPromptMessage = "To get started, select your desired model and tokenizer from the top right corner"

    fun checkAndLoadSettings() {
        val updatedSettingsFields = demoSharedPreferences.getModuleSettings()
        appSettings = demoSharedPreferences.getAppSettings()
        val isUpdated = currentSettingsFields != updatedSettingsFields
        val isLoadModel = updatedSettingsFields.isLoadModel
        // Update LoRA mode state
        isLoraMode = updatedSettingsFields.isLoraMode
        availableModels = updatedSettingsFields.models
        activeModelId = updatedSettingsFields.activeModelId

        if (isUpdated) {
            val settingsAfterClear = checkForClearChatHistory(updatedSettingsFields)

            if (isLoadModel) {
                // Update local copy BEFORE checking media capabilities
                val settingsWithLoadFlagCleared = settingsAfterClear.copy(isLoadModel = false)
                currentSettingsFields = settingsWithLoadFlagCleared
                demoSharedPreferences.saveModuleSettings(settingsWithLoadFlagCleared)

                // Update media capabilities after settings are updated
                setBackendMode(settingsAfterClear.backendType)

                if (isLoraMode && settingsAfterClear.hasModels()) {
                    // LoRA mode: Load all configured models
                    loadLoraModels(settingsAfterClear)
                } else {
                    // Legacy single-model mode
                    loadLocalModelAndParameters(
                        settingsAfterClear.modelFilePath,
                        settingsAfterClear.tokenizerFilePath,
                        settingsAfterClear.dataPath,
                        settingsAfterClear.temperature.toFloat()
                    )
                }
            } else {
                currentSettingsFields = settingsAfterClear.copy()
                // Update media capabilities after settings are updated
                setBackendMode(settingsAfterClear.backendType)
                if (module == null && loadedModules.isEmpty()) {
                    addSystemMessage(systemPromptMessage)
                }
            }
        } else {
            // Settings unchanged, but still update media capabilities for current settings
            setBackendMode(updatedSettingsFields.backendType)
            val modelPath = updatedSettingsFields.modelFilePath
            val tokenizerPath = updatedSettingsFields.tokenizerFilePath
            if (modelPath.isEmpty() || tokenizerPath.isEmpty()) {
                if (!isLoraMode || !updatedSettingsFields.hasModels()) {
                    addSystemMessage(systemPromptMessage)
                }
            }
        }
    }

    /**
     * Loads all models configured in LoRA mode.
     */
    private fun loadLoraModels(settings: ModuleSettings) {
        Thread {
            val sharedDataPath = settings.getEffectiveDataPath()
            
            // Build detailed loading message with args for each model
            val loadingDetails = StringBuilder("Loading ${settings.models.size} LoRA model(s):\n")
            settings.models.forEachIndexed { index, modelConfig ->
                if (modelConfig.isValid()) {
                    val dataFiles = mutableListOf<String>()
                    if (sharedDataPath.isNotEmpty()) {
                        dataFiles.add(sharedDataPath)
                    }
                    dataFiles.addAll(modelConfig.adapterFilePaths)
                    
                    loadingDetails.append("\n${index + 1}. ${modelConfig.displayName}\n")
                    loadingDetails.append("   PTE: ${modelConfig.modelFilePath.substringAfterLast('/')}\n")
                    loadingDetails.append("   Tokenizer: ${modelConfig.tokenizerFilePath.substringAfterLast('/')}\n")
                    loadingDetails.append("   Data files: ${if (dataFiles.isEmpty()) "none" else dataFiles.joinToString(", ") { it.substringAfterLast('/') }}\n")
                }
            }
            
            val modelLoadingMessage = Message(loadingDetails.toString(), false, MessageType.SYSTEM, 0)
            _messages.add(modelLoadingMessage)
            isModelReady = false

            var loadedCount = 0
            var firstLoadedModelId: String? = null

            for (modelConfig in settings.models) {
                if (!modelConfig.isValid()) continue

                try {
                    // Build dataFiles list: foundation PTD + adapter PTDs
                    val dataFiles = mutableListOf<String>()
                    if (sharedDataPath.isNotEmpty()) {
                        dataFiles.add(sharedDataPath)
                    }
                    dataFiles.addAll(modelConfig.adapterFilePaths)

                    val dataFilesLog = if (dataFiles.isEmpty()) "no data files" else dataFiles.joinToString(", ")
                    ETLogging.getInstance().log(
                        "LoRA: Loading model ${modelConfig.displayName} with tokenizer ${modelConfig.tokenizerFilePath}, data files: $dataFilesLog"
                    )

                    val runStartTime = System.currentTimeMillis()
                    val llmModule = LlmModule(
                        ModelUtils.getModelCategory(modelConfig.modelType, modelConfig.backendType),
                        modelConfig.modelFilePath,
                        modelConfig.tokenizerFilePath,
                        modelConfig.temperature.toFloat(),
                        dataFiles
                    )

                    llmModule.load()
                    val loadDuration = System.currentTimeMillis() - runStartTime

                    // Store in map
                    loadedModules[modelConfig.id] = llmModule
                    loadedCount++

                    if (firstLoadedModelId == null) {
                        firstLoadedModelId = modelConfig.id
                    }

                    ETLogging.getInstance().log(
                        "LoRA: Loaded ${modelConfig.displayName} in ${loadDuration.toFloat() / 1000} sec"
                    )
                } catch (e: ExecutorchRuntimeException) {
                    ETLogging.getInstance().log("LoRA: Failed to load ${modelConfig.displayName}: ${e.message}")
                    _messages.add(Message("Failed to load ${modelConfig.displayName}: ${e.message}", false, MessageType.SYSTEM, 0))
                }
            }

            _messages.remove(modelLoadingMessage)

            if (loadedCount > 0) {
                // Set the active module
                val activeId = if (settings.activeModelId.isNotEmpty() && loadedModules.containsKey(settings.activeModelId)) {
                    settings.activeModelId
                } else {
                    firstLoadedModelId ?: ""
                }

                activeModelId = activeId
                module = loadedModules[activeId]

                val activeModelName = settings.getModelById(activeId)?.displayName ?: "Unknown"
                _messages.add(Message(
                    "Successfully loaded $loadedCount model(s). Active: $activeModelName. Use the switch button to change models.",
                    false, MessageType.SYSTEM, 0
                ))
                isModelReady = true
            } else {
                _messages.add(Message("No models loaded. Please check your configuration.", false, MessageType.SYSTEM, 0))
                isModelReady = false
            }
        }.start()
    }

    /**
     * Switches to a different model in LoRA mode.
     * Creates a new LlmModule with the selected PTE and the same PTD.
     */
    fun switchToModel(modelId: String) {
        if (!isLoraMode) return
        if (isGenerating) {
            addSystemMessage("Cannot switch models while generating. Please wait or stop generation.")
            return
        }

        val modelConfig = currentSettingsFields.getModelById(modelId)
        if (modelConfig == null) {
            addSystemMessage("Model not found.")
            return
        }

        // Check if model is already loaded
        if (loadedModules.containsKey(modelId)) {
            // Just switch to the already loaded module
            module = loadedModules[modelId]
            activeModelId = modelId

            // Update settings with new active model
            currentSettingsFields = currentSettingsFields.setActiveModel(modelId)
            demoSharedPreferences.saveModuleSettings(currentSettingsFields)

            addSystemMessage("Switched to ${modelConfig.displayName}")
            ETLogging.getInstance().log("LoRA: Switched to already loaded model ${modelConfig.displayName}")
        } else {
            // Need to load the model first
            Thread {
                val sharedDataPath = currentSettingsFields.getEffectiveDataPath()
                addSystemMessage("Loading ${modelConfig.displayName}...")
                isModelReady = false

                try {
                    // Build dataFiles list: foundation PTD + adapter PTDs
                    val dataFiles = mutableListOf<String>()
                    if (sharedDataPath.isNotEmpty()) {
                        dataFiles.add(sharedDataPath)
                    }
                    dataFiles.addAll(modelConfig.adapterFilePaths)

                    val runStartTime = System.currentTimeMillis()
                    val llmModule = LlmModule(
                        ModelUtils.getModelCategory(modelConfig.modelType, modelConfig.backendType),
                        modelConfig.modelFilePath,
                        modelConfig.tokenizerFilePath,
                        modelConfig.temperature.toFloat(),
                        dataFiles
                    )

                    llmModule.load()
                    val loadDuration = System.currentTimeMillis() - runStartTime

                    // Store and switch
                    loadedModules[modelId] = llmModule
                    module = llmModule
                    activeModelId = modelId

                    // Update settings
                    currentSettingsFields = currentSettingsFields.setActiveModel(modelId)
                    demoSharedPreferences.saveModuleSettings(currentSettingsFields)

                    addSystemMessage("Switched to ${modelConfig.displayName} (loaded in ${loadDuration.toFloat() / 1000} sec)")
                    ETLogging.getInstance().log("LoRA: Loaded and switched to ${modelConfig.displayName} in ${loadDuration.toFloat() / 1000} sec")
                    isModelReady = true
                } catch (e: ExecutorchRuntimeException) {
                    addSystemMessage("Failed to load ${modelConfig.displayName}: ${e.message}")
                    ETLogging.getInstance().log("LoRA: Failed to load ${modelConfig.displayName}: ${e.message}")
                    isModelReady = loadedModules.isNotEmpty()
                }
            }.start()
        }
    }

    private fun setBackendMode(backendType: BackendType) {
        // Media buttons visibility depends on backend (MediaTek doesn't support media)
        val backendSupportsMedia = when (backendType) {
            BackendType.XNNPACK, BackendType.QUALCOMM, BackendType.VULKAN -> true
            BackendType.MEDIATEK -> false
        }
        updateMediaCapabilities(backendSupportsMedia)
    }

    private fun updateMediaCapabilities(backendSupportsMedia: Boolean) {
        // In LoRA mode, use foundation model type; otherwise use regular model type
        val modelType = if (currentSettingsFields.isLoraMode) {
            currentSettingsFields.foundationModelType
        } else {
            currentSettingsFields.modelType
        }
        
        supportsImageInput = backendSupportsMedia
        supportsAudioInput = backendSupportsMedia && modelType.supportsAudio()
        showMediaButtons = supportsImageInput || supportsAudioInput
        ETLogging.getInstance().log("updateMediaCapabilities: modelType=$modelType, supportsImage=${modelType.supportsImage()}, supportsAudio=${modelType.supportsAudio()}, showMediaButtons=$showMediaButtons")
    }

    private fun getCapabilityDescription(modelType: ModelType): String {
        return when {
            modelType.supportsImage() && modelType.supportsAudio() ->
                "You can send text, images, or audio for inference."
            modelType.supportsImage() ->
                "You can send text or images for inference."
            modelType.supportsAudio() ->
                "You can send text or audio for inference."
            else ->
                "You can send text for inference."
        }
    }

    private fun checkForClearChatHistory(updatedSettingsFields: ModuleSettings): ModuleSettings {
        if (updatedSettingsFields.isClearChatHistory) {
            _messages.clear()
            demoSharedPreferences.removeExistingMessages()
            val clearedSettings = updatedSettingsFields.copy(isClearChatHistory = false)
            demoSharedPreferences.saveModuleSettings(clearedSettings)
            module?.resetContext()
            shouldAddSystemPrompt = true
            promptID = 0
            return clearedSettings
        }
        return updatedSettingsFields
    }

    private fun loadLocalModelAndParameters(
        modelFilePath: String,
        tokenizerFilePath: String,
        dataPath: String,
        temperature: Float
    ) {
        Thread {
            setLocalModel(modelFilePath, tokenizerFilePath, dataPath, temperature)
        }.start()
    }

    private fun setLocalModel(
        modelPath: String,
        tokenizerPath: String,
        dataPath: String,
        temperature: Float
    ) {
        val modelLoadingMessage = Message("Loading model...", false, MessageType.SYSTEM, 0)
        ETLogging.getInstance().log(
            "Loading model $modelPath with tokenizer $tokenizerPath data path $dataPath"
        )

        isModelReady = false
        _messages.add(modelLoadingMessage)

        val runStartTime = System.currentTimeMillis()
        module = if (dataPath.isEmpty()) {
            LlmModule(
                ModelUtils.getModelCategory(
                    currentSettingsFields.modelType,
                    currentSettingsFields.backendType
                ),
                modelPath,
                tokenizerPath,
                temperature
            )
        } else {
            LlmModule(
                ModelUtils.getModelCategory(
                    currentSettingsFields.modelType,
                    currentSettingsFields.backendType
                ),
                modelPath,
                tokenizerPath,
                temperature,
                dataPath
            )
        }

        var loadDuration = System.currentTimeMillis() - runStartTime
        var modelInfo: String

        var loadSuccess = false
        try {
            module?.load()
            val pteName = modelPath.substringAfterLast('/')
            val tokenizerName = tokenizerPath.substringAfterLast('/')
            val capabilityText = getCapabilityDescription(currentSettingsFields.modelType)
            modelInfo = "Successfully loaded model. $pteName and tokenizer $tokenizerName in ${loadDuration.toFloat() / 1000} sec. $capabilityText"

            if (currentSettingsFields.modelType == ModelType.LLAVA_1_5) {
                val llavaPresetPrompt = PromptFormat.getLlavaPresetPrompt()
                ETLogging.getInstance().log("Llava start prefill prompt: $llavaPresetPrompt")
                module?.prefillPrompt(llavaPresetPrompt)
                ETLogging.getInstance().log("Llava completes prefill prompt")
            }
            loadSuccess = true
        } catch (e: ExecutorchRuntimeException) {
            modelInfo = "Model load failure: ${e.message}"
            loadDuration = 0
            modelLoadError = modelInfo
            showModelLoadErrorDialog = true
        }

        val modelLoadedMessage = Message(modelInfo, false, MessageType.SYSTEM, 0)
        val modelLoggingInfo = "Model path: $modelPath\n" +
                "Tokenizer path: $tokenizerPath\n" +
                "Backend: ${currentSettingsFields.backendType}\n" +
                "ModelType: ${ModelUtils.getModelCategory(currentSettingsFields.modelType, currentSettingsFields.backendType)}\n" +
                "Temperature: $temperature\n" +
                "Model loaded time: $loadDuration ms"
        ETLogging.getInstance().log("Load complete. $modelLoggingInfo")

        isModelReady = loadSuccess
        _messages.remove(modelLoadingMessage)
        _messages.add(modelLoadedMessage)
    }

    fun updateMemoryUsage() {
        val context = getApplication<Application>()
        val memoryInfo = ActivityManager.MemoryInfo()
        val activityManager = context.getSystemService(Context.ACTIVITY_SERVICE) as? ActivityManager
            ?: return
        activityManager.getMemoryInfo(memoryInfo)
        val totalMem = memoryInfo.totalMem / (1024 * 1024)
        val availableMem = memoryInfo.availMem / (1024 * 1024)
        val usedMem = totalMem - availableMem
        ramUsage = "${usedMem}MB"
    }

    fun toggleThinkMode() {
        thinkMode = !thinkMode
        val thinkingModeText = if (thinkMode) "on" else "off"
        _messages.add(Message("Thinking mode is $thinkingModeText", false, MessageType.SYSTEM, 0))
    }

    fun toggleMediaSelector() {
        showMediaSelector = !showMediaSelector
    }

    private fun knowledgeText(label: String, lang: String): String? {
        return try {
            val txt = getApplication<Application>().assets.open("diseases.json")
                .bufferedReader(Charsets.UTF_8).use { it.readText() }
            val e = JSONObject(txt).optJSONObject(label) ?: return null
            if (lang == "ta") {
                e.getString("name_ta") + "\n\n" +
                        "\u0B85\u0BB1\u0BBF\u0B95\u0BC1\u0BB1\u0BBF\u0B95\u0BB3\u0BCD: " + e.getString("symptoms_ta") + "\n\n" +
                        "\u0B95\u0BBE\u0BB0\u0BA3\u0BAE\u0BCD: " + e.getString("cause_ta") + "\n\n" +
                        "\u0B9A\u0BBF\u0B95\u0BBF\u0B9A\u0BCD\u0B9A\u0BC8: " + e.getString("treatment_ta") + "\n\n" +
                        "\u0BA4\u0B9F\u0BC1\u0BAA\u0BCD\u0BAA\u0BC1: " + e.getString("prevention_ta")
            } else {
                e.getString("name_en") + "\n\n" +
                        "Symptoms: " + e.getString("symptoms_en") + "\n\n" +
                        "Cause: " + e.getString("cause_en") + "\n\n" +
                        "Treatment: " + e.getString("treatment_en") + "\n\n" +
                        "Prevention: " + e.getString("prevention_en")
            }
        } catch (ex: Exception) {
            null
        }
    }
    private fun kbField(label: String, field: String): String? {
        return try {
            val txt = getApplication<Application>().assets.open("diseases.json")
                .bufferedReader(Charsets.UTF_8).use { it.readText() }
            JSONObject(txt).optJSONObject(label)?.optString(field)
        } catch (ex: Exception) {
            null
        }
    }
    private var leafClassifier: LeafClassifier? = null

    private fun scanLeaf(uri: Uri) {
        if (isGenerating) return
        _messages.add(Message(uri.toString(), true, MessageType.IMAGE, 0))
        _messages.add(Message("Scanning leaf... Tip: one leaf, close up, good light.", false, MessageType.SYSTEM, 0))
        executor.execute {
            try {
                if (leafClassifier == null) {
                    leafClassifier = LeafClassifier("/data/local/tmp/vision.pte")
                }
                val t0 = System.currentTimeMillis()
                val r = leafClassifier!!.classify(contentResolver, uri)
                val ms = System.currentTimeMillis() - t0
                val pct = (r.confidence * 100).toInt()
                val name = r.label.replace("___", " ").replace("_", " ")
                val top3 = r.top3.joinToString("; ") {
                    it.first.replace("___", " ")
                        .replace("_", " ") + " " + (it.second * 100).toInt() + "%"
                }
                ETLogging.getInstance().log("Leaf scan: " + top3 + " in " + ms + " ms")
                if (r.isSure()) {
                    addSystemMessage("Detected: " + name + " (" + pct + "%), vision time " + ms + " ms")
                    knowledgeText(r.label, adviceLang)?.let { addSystemMessage(it) }
                    if (r.label.contains("healthy", ignoreCase = true)) {
                        addSystemMessage(if (adviceLang == "ta") "\u0BA8\u0BCB\u0BAF\u0BCD \u0B8E\u0BA4\u0BC1\u0BB5\u0BC1\u0BAE\u0BCD \u0B95\u0BA3\u0BCD\u0B9F\u0BB1\u0BBF\u0BAF\u0BAA\u0BCD\u0BAA\u0B9F\u0BB5\u0BBF\u0BB2\u0BCD\u0BB2\u0BC8. \u0B85\u0BB5\u0BCD\u0BB5\u0BAA\u0BCD\u0BAA\u0BCB\u0BA4\u0BC1 \u0BAE\u0BC1\u0BB4\u0BC1\u0B9A\u0BCD \u0B9A\u0BC6\u0B9F\u0BBF\u0BAF\u0BC8\u0BAF\u0BC1\u0BAE\u0BCD \u0BAA\u0BBE\u0BB0\u0BC1\u0B99\u0BCD\u0B95\u0BB3\u0BCD." else "No disease found. Check the whole plant now and then.")
                    } else if (isModelReady) {
                        module?.resetContext()
                        val what = (kbField(r.label, "symptoms_en") ?: "") + " " + (kbField(
                            r.label,
                            "cause_en"
                        ) ?: "")
                        addSystemMessage("Below, the on-device AI (Llama) explains the disease in simple words. For treatment and prevention, follow the verified advice above.")
                        inputText =
                            "The plant disease is called " + name + ". Explain in simple words, in under 50 words, what it looks like and what causes it. Use this exact name and do not invent other names. Use ONLY the facts below. Do not give any treatment or advice.\n\nFACTS:\n" + what
                        sendMessage(name)
                    } else {
                        addSystemMessage("Llama is not loaded, so no summary can be written. The verified advice above still applies.")
                    }
                } else {
                    addSystemMessage("Not sure (" + pct + "% for " + name + "). Retake the photo: one leaf, close up, good light. Top guesses: " + top3)
                }
            } catch (e: Exception) {
                addSystemMessage(if (e.message == "NOT_A_LEAF") "This does not look like a leaf. Retake the photo: one leaf, close up, good light." else "Leaf scan failed: " + e.message)
            }
        }
    }

    fun addImage(uri: Uri) {
        if (currentSettingsFields.modelType != ModelType.LLAVA_1_5 &&
            currentSettingsFields.modelType != ModelType.GEMMA_3
        ) {
            scanLeaf(uri)
            return
        }
        if (_selectedImages.size < MAX_NUM_OF_IMAGES) {
            _selectedImages.add(uri)
            prefillImageIfNeeded()
        }
    }

    fun removeImage(uri: Uri) {
        _selectedImages.remove(uri)
    }

    fun clearImages() {
        _selectedImages.clear()
    }

    fun setAudioFile(path: String) {
        audioFileToPrefill = path
        _messages.add(Message("Selected audio: $path", false, MessageType.SYSTEM, 0))
    }

    private fun prefillImageIfNeeded() {
        if (currentSettingsFields.modelType == ModelType.LLAVA_1_5 ||
            currentSettingsFields.modelType == ModelType.GEMMA_3
        ) {
            val processedImageList = getProcessedImagesForModel(_selectedImages)
            if (processedImageList.isNotEmpty()) {
                _messages.add(
                    Message("Starting image prefill.", false, MessageType.SYSTEM, 0)
                )
                executor.execute {
                    android.os.Process.setThreadPriority(android.os.Process.THREAD_PRIORITY_MORE_FAVORABLE)
                    ETLogging.getInstance().log("Starting runnable prefill image")
                    val img = processedImageList[0]
                    ETLogging.getInstance().log("Starting prefill image")
                    if (currentSettingsFields.modelType == ModelType.LLAVA_1_5) {
                        module?.prefillImages(
                            img.getInts(),
                            img.width,
                            img.height,
                            ModelUtils.VISION_MODEL_IMAGE_CHANNELS
                        )
                    } else if (currentSettingsFields.modelType == ModelType.GEMMA_3) {
                        val gemmaPreImagePrompt = PromptFormat.getGemmaPreImagePrompt()
                        ETLogging.getInstance().log("Gemma prefill pre-image prompt: $gemmaPreImagePrompt")
                        module?.prefillPrompt(gemmaPreImagePrompt)
                        module?.prefillImages(
                            img.getFloats(),
                            img.width,
                            img.height,
                            ModelUtils.VISION_MODEL_IMAGE_CHANNELS
                        )
                    }
                }
            }
        }
    }

    private fun getInputImageSideSize(): Int {
        return when (currentSettingsFields.modelType) {
            ModelType.LLAVA_1_5 -> 336
            ModelType.GEMMA_3 -> 896
            else -> throw IllegalArgumentException("Unsupported model type: ${currentSettingsFields.modelType}")
        }
    }

    private fun getProcessedImagesForModel(uris: List<Uri>): List<ETImage> {
        val imageList = mutableListOf<ETImage>()
        uris.forEach { uri ->
            imageList.add(ETImage(contentResolver, uri, getInputImageSideSize()))
        }
        return imageList
    }

    private fun replyLooksBad(reply: String, disease: String): Boolean {
        val trimmed = reply.trim()
        val words = Regex("[\\p{L}\\p{N}]+")
            .findAll(trimmed.lowercase(java.util.Locale.ROOT))
            .map { it.value }.toList()

        if (trimmed.length < 40 || trimmed.length > 700) return true
        if (words.size < 8 || words.size > 90) return true
        if (words.toSet().size.toDouble() / words.size < 0.45) return true
        if (words.windowed(3).any { it[0] == it[1] && it[1] == it[2] }) return true

        val repeatedPhrases = words.windowed(3)
            .groupingBy { it.joinToString(" ") }.eachCount()
        if (repeatedPhrases.values.any { it >= 3 }) return true

        val normalizedReply = trimmed.replace(Regex("\\s+"), " ")
        val normalizedDisease = disease.trim().replace(Regex("\\s+"), " ")
        val diseasePattern = Regex(
            "(?<![\\p{L}\\p{N}])" + Regex.escape(normalizedDisease) +
                "(?![\\p{L}\\p{N}])",
            RegexOption.IGNORE_CASE
        )
        return !diseasePattern.containsMatchIn(normalizedReply)
    }

    fun sendMessage() {
        sendMessage("")
    }

    private fun sendMessage(expectedDisease: String) {
        if (inputText.trim().isEmpty()) return
        if (!isModelReady || isGenerating) return

        // Add selected images to chat
        for (imageURI in _selectedImages) {
            _messages.add(Message(imageURI.toString(), true, MessageType.IMAGE, 0))
        }

        val rawPrompt = inputText
        val finalPrompt: String

        if (currentSettingsFields.modelType == ModelType.LLAVA_1_5 && _selectedImages.isNotEmpty()) {
            finalPrompt = PromptFormat.getLlavaMultimodalUserPrompt()
                .replace(PromptFormat.USER_PLACEHOLDER, rawPrompt)
        } else if (currentSettingsFields.modelType == ModelType.GEMMA_3 && _selectedImages.isNotEmpty()) {
            finalPrompt = PromptFormat.getGemmaMultimodalUserPrompt()
                .replace(PromptFormat.USER_PLACEHOLDER, rawPrompt)
        } else {
            finalPrompt = (if (shouldAddSystemPrompt) currentSettingsFields.getFormattedSystemPrompt() else "") +
                    currentSettingsFields.getFormattedUserPrompt(rawPrompt, thinkMode)
        }
        shouldAddSystemPrompt = false

        // Add user message
        _messages.add(Message(rawPrompt, true, MessageType.TEXT, promptID))
        inputText = ""

        // Create result message placeholder
        resultMessage = Message("", false, MessageType.TEXT, promptID)
        isInThinkingBlock = false
        _messages.add(resultMessage!!)

        // Clear selected images after adding to chat
        _selectedImages.clear()
        showMediaSelector = false
        promptID++

        // Run generation
        executor.execute {
            android.os.Process.setThreadPriority(android.os.Process.THREAD_PRIORITY_MORE_FAVORABLE)
            ETLogging.getInstance().log("starting runnable generate()")
            isGenerating = true

            var explanationFailed = false
            val generateStartTime = System.currentTimeMillis()
            if (ModelUtils.getModelCategory(
                    currentSettingsFields.modelType,
                    currentSettingsFields.backendType
                ) == ModelUtils.VISION_MODEL
            ) {
                if (currentSettingsFields.modelType == ModelType.VOXTRAL && audioFileToPrefill != null) {
                    prefillVoxtralAudio(audioFileToPrefill!!, finalPrompt)
                    audioFileToPrefill = null
                    ETLogging.getInstance().log("Running vision model inference.. prompt=(empty after audio prefill)")
                    module?.generate("", appSettings.maxSeqLen, this, false)
                } else {
                    ETLogging.getInstance().log("Running vision model inference.. prompt=$finalPrompt")
                    module?.generate(finalPrompt, appSettings.maxSeqLen, this, false)
                }
            } else if (currentSettingsFields.modelType == ModelType.LLAMA_GUARD_3) {
                val llamaGuardPromptForClassification =
                    PromptFormat.getFormattedLlamaGuardPrompt(rawPrompt)
                ETLogging.getInstance().log("Running inference.. prompt=$llamaGuardPromptForClassification")
                module?.generate(
                    llamaGuardPromptForClassification,
                    llamaGuardPromptForClassification.length + 64,
                    this,
                    false
                )
            } else {
                ETLogging.getInstance().log("Running inference.. prompt=$finalPrompt")
                try {
                    module?.generate(finalPrompt, appSettings.maxSeqLen, this, false)
                } catch (e: Exception) {
                    if (expectedDisease.isNotEmpty()) {
                        explanationFailed = true
                        ETLogging.getInstance().log("Leaf explanation generation failed")
                    } else {
                        addSystemMessage("Llama could not write the explanation. The verified advice above still applies.")
                    }
                }
            }

            val generateDuration = System.currentTimeMillis() - generateStartTime
            resultMessage?.let { msg ->
                msg.totalGenerationTime = generateDuration
                val rejectExplanation = expectedDisease.isNotEmpty() &&
                    (explanationFailed || replyLooksBad(msg.text, expectedDisease))
                val index = _messages.indexOfLast { it === msg }
                if (index >= 0) {
                    val updated = if (rejectExplanation) {
                        Log.w("LeafCheck", "rejected reply (" + msg.text.trim().split(Regex("\\s+")).size + " words): " + msg.text)
                        ETLogging.getInstance().log(
                            "Leaf explanation rejected by output checker: $expectedDisease"
                        )
                        Message(
                            "Llama could not write a clear explanation. The verified advice above still applies.",
                            msg.isSent,
                            msg.messageType,
                            msg.promptID,
                            msg.timestamp,
                            msg.id
                        ).also {
                            it.tokensPerSecond = msg.tokensPerSecond
                            it.totalGenerationTime = msg.totalGenerationTime
                        }
                    } else {
                        if (expectedDisease.isNotEmpty()) {
                            ETLogging.getInstance().log(
                                "Leaf explanation passed output checker: $expectedDisease"
                            )
                        }
                        msg.copy()
                    }
                    _messages[index] = updated
                    resultMessage = updated
                }
            }
            isGenerating = false
            ETLogging.getInstance().log("Inference completed")
        }
    }

    fun stopGeneration() {
        Log.i("ChatViewModel", "stopGeneration called")
        module?.stop()
    }

    private fun prefillVoxtralAudio(audioFeaturePath: String, textPrompt: String) {
        try {
            val byteData = Files.readAllBytes(Paths.get(audioFeaturePath))
            val buffer = ByteBuffer.wrap(byteData).order(ByteOrder.LITTLE_ENDIAN)
            val floatCount = byteData.size / java.lang.Float.BYTES
            val floats = FloatArray(floatCount)

            for (i in 0 until floatCount) {
                floats[i] = buffer.float
            }
            val bins = 128
            val frames = 3000
            val batchSize = floatCount / (bins * frames)
            val preAudioPrompt = "<s>[INST][BEGIN_AUDIO]"
            val postAudioPrompt = "$textPrompt[/INST]"
            ETLogging.getInstance().log("Voxtral prefill pre-audio prompt: $preAudioPrompt")
            module?.prefillPrompt(preAudioPrompt)
            module?.prefillAudio(floats, batchSize, bins, frames)
            ETLogging.getInstance().log("Voxtral prefill post-audio prompt: $postAudioPrompt")
            module?.prefillPrompt(postAudioPrompt)
        } catch (e: IOException) {
            Log.e("AudioPrefill", "Audio file error")
        }
    }

    fun dismissModelLoadErrorDialog() {
        showModelLoadErrorDialog = false
    }

    fun addSystemMessage(text: String) {
        if (!isDuplicateSystemMessage(text)) {
            _messages.add(Message(text, false, MessageType.SYSTEM, 0))
        }
    }

    private fun isDuplicateSystemMessage(text: String): Boolean {
        if (_messages.isEmpty()) return false
        val lastMessage = _messages.last()
        return lastMessage.messageType == MessageType.SYSTEM && text == lastMessage.text
    }

    // LlmCallback implementation
    override fun onResult(result: String) {
        var processedResult = result

        if (processedResult == PromptFormat.getStopToken(currentSettingsFields.modelType)) {
            module?.stop()
            return
        }

        // Thinking mode state machine: intercept <think> / </think> tags
        if (processedResult == "<think>") {
            isInThinkingBlock = true
            return
        }
        if (processedResult == "</think>") {
            isInThinkingBlock = false
            return
        }

        processedResult = PromptFormat.replaceSpecialToken(currentSettingsFields.modelType, processedResult)

        if (currentSettingsFields.modelType == ModelType.LLAMA_3 &&
            processedResult == "<|start_header_id|>"
        ) {
            sawStartHeaderId = true
        }
        if (currentSettingsFields.modelType == ModelType.LLAMA_3 &&
            processedResult == "<|end_header_id|>"
        ) {
            sawStartHeaderId = false
            return
        }
        if (sawStartHeaderId) {
            return
        }

        if (isInThinkingBlock) {
            // Skip leading newlines in thinking content
            val keepThinking = !(processedResult == "\n" || processedResult == "\n\n") ||
                    resultMessage?.thinkingContent?.isNotEmpty() == true
            if (keepThinking) {
                resultMessage?.appendThinkingText(processedResult)
            }
        } else {
            val keepResult = !(processedResult == "\n" || processedResult == "\n\n") ||
                    resultMessage?.text?.isNotEmpty() == true
            if (keepResult) {
                resultMessage?.appendText(processedResult)
            }
        }

        // Redraw the chat bubble at most every 150 ms: fewer text-layout passes while Llama streams.
        val nowMs = System.currentTimeMillis()
        if (nowMs - lastBubbleUpdateMs < 150) return
        lastBubbleUpdateMs = nowMs
        val index = _messages.indexOfLast { it === resultMessage }
        if (index >= 0) {
            val updated = resultMessage!!.copy()
            _messages[index] = updated
            resultMessage = updated
        }
    }

    override fun onStats(stats: String) {
        resultMessage?.let { msg ->
            var tps = 0f
            try {
                val jsonObject = JSONObject(stats)
                val numGeneratedTokens = jsonObject.getInt("generated_tokens")
                val inferenceEndMs = jsonObject.getInt("inference_end_ms")
                val promptEvalEndMs = jsonObject.getInt("prompt_eval_end_ms")
                tps = numGeneratedTokens.toFloat() / (inferenceEndMs - promptEvalEndMs) * 1000
            } catch (e: JSONException) {
                Log.e("LLM", "Error parsing JSON: ${e.message}")
            }
            msg.tokensPerSecond = tps
            // Create a new reference to trigger recomposition under strong skipping mode
            val index = _messages.indexOfLast { it === msg }
            if (index >= 0) {
                val updated = msg.copy()
                _messages[index] = updated
                resultMessage = updated
            }
        }
    }

    companion object {
        private const val MAX_NUM_OF_IMAGES = 5
    }
}

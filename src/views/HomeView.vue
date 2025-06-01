<script setup lang="ts">
import { ref, onMounted, onUnmounted } from "vue";
// 引入图标
import {
  ArrowUpTrayIcon,
  MicrophoneIcon,
  StopCircleIcon,
  DocumentTextIcon,
  ExclamationCircleIcon,
} from "@heroicons/vue/24/solid";
// 引入自定义组件
import BaseButton from "../components/BaseButton.vue";
import StatusIndicator from "../components/StatusIndicator.vue";

// 核心 API invoke 用于调用 Rust 命令
import { invoke } from "@tauri-apps/api/core";
// 导入 dialog 插件的 open 函数用于文件选择
import { open } from "@tauri-apps/plugin-dialog";
// 导入 event 模块的 listen 函数用于监听拖放等事件
import { listen } from "@tauri-apps/api/event";

// 定义应用可能的状态类型
type AppState =
  | "idle"
  | "selecting"
  | "recording"
  | "processing"
  | "success"
  | "error";
// 当前应用状态
const currentState = ref<AppState>("idle");
// 存储转录结果
const transcription = ref<string>("");
// 存储错误信息
const errorMessage = ref<string | null>(null);
// 存储当前选定或录制的文件名 (用于显示)
const selectedFileName = ref<string | null>(null);
// 跟踪文件是否正在拖拽进入窗口 (用于拖放视觉效果)
const isDragging = ref(false);

// --- Tauri API 调用与处理函数 ---

// 处理 "选择文件" 按钮点击或拖放区域点击
async function handleSelectFile() {
  // 如果正在录音或处理中，则不响应点击
  if (currentState.value === "recording" || currentState.value === "processing")
    return;

  currentState.value = "selecting"; // 进入选择中状态
  selectedFileName.value = null;
  errorMessage.value = null;
  transcription.value = "";
  console.log("Tauri v2: 使用 Dialog 插件打开文件对话框...");
  try {
    // 调用 Dialog 插件的 open API
    const result = await open({
      multiple: false, // 只允许选择单个文件
      directory: false, // 不允许选择目录
      filters: [
        {
          // 定义文件类型过滤器
          name: "Audio",
          extensions: ["wav", "mp3", "m4a", "ogg", "flac"], // 支持的音频格式
        },
      ],
    });

    // 处理返回结果
    if (result) {
      // result 在 single 模式下是文件路径字符串或 null
      const filePath = Array.isArray(result) ? result[0] : result; // 处理兼容 multiple:true 的情况
      if (filePath) {
        // 从路径中提取文件名用于显示
        selectedFileName.value = filePath.split(/[\\/]/).pop() ?? filePath;
        console.log("Tauri v2: 文件已选择:", filePath);
        await processAudio(filePath); // 调用音频处理函数
      } else {
        // 理论上 filePath 不会是 null 且 result 不为 null，但做个防御
        currentState.value = "idle";
      }
    } else {
      // 用户点击了取消按钮
      currentState.value = "idle";
      console.log("Tauri v2: 用户取消文件选择");
    }
  } catch (err) {
    // 处理 Dialog 插件可能抛出的错误
    console.error("Tauri v2: Dialog 插件出错:", err);
    errorMessage.value = `文件选择失败: ${err instanceof Error ? err.message : String(err)}`;
    currentState.value = "error";
  } finally {
    // 确保如果操作未进入 processing，最终状态回到 idle
    if (currentState.value === "selecting") {
      currentState.value = "idle";
    }
  }
}

// 处理录音按钮的点击事件 (开始/停止切换)
async function handleRecord() {
  // 如果正在选择文件或处理中，则不响应
  if (currentState.value === "selecting" || currentState.value === "processing")
    return;

  if (currentState.value === "recording") {
    await stopRecording();
  } else {
    await startRecording();
  }
}

// 调用后端开始录音的命令
async function startRecording() {
  currentState.value = "recording";
  selectedFileName.value = "实时录音"; // 显示操作类型
  errorMessage.value = null;
  transcription.value = "";
  console.log("占位符：调用后端 start_recording 命令...");
  try {
    // --- 实际调用 Rust 后端命令 ---
    await invoke("start_recording");
    console.log("占位符：后端 start_recording 命令调用成功 (模拟)");
    // 注意：实际应用中，录音状态可能需要通过事件从后端更新
  } catch (err) {
    console.error("占位符：调用 start_recording 命令失败:", err);
    errorMessage.value = `开始录音失败: ${err instanceof Error ? err.message : String(err)}`;
    currentState.value = "error";
  }
}

// 调用后端停止录音的命令
async function stopRecording() {
  console.log("调用后端 stop_recording 命令...");
  currentState.value = "processing"; // 进入处理状态
  try {
    // --- 实际调用 Rust 后端命令 ---
    // 假设后端停止录音后，会返回录音文件的路径
    const resultPath = await invoke<string>("stop_recording");
    console.log("后端 stop_recording 命令成功，返回路径:", resultPath);
    if (resultPath) {
      await processAudio(resultPath); // 处理返回的录音文件
    } else {
      throw new Error("后端未返回有效的录音文件路径");
    }

    // --- 模拟 ---
    // const mockRecordingPath = "/mock/path/to/recording.wav";
    // console.log('占位符：模拟录音停止，文件：', mockRecordingPath);
    // await processAudio(mockRecordingPath);
  } catch (err) {
    console.error("调用 stop_recording 或后续处理失败:", err);
    errorMessage.value = `停止录音或处理失败: ${err instanceof Error ? err.message : String(err)}`;
    currentState.value = "error";
  }
}

// 调用后端处理音频文件 (转录) 的命令
async function processAudio(filePath: string) {
  currentState.value = "processing"; // 确保处于处理状态
  errorMessage.value = null;
  // transcription.value = ''; // 可选：处理前清空旧结果，或保留以显示加载状态
  console.log(`调用后端 transcribe_audio 命令处理文件： ${filePath}`);
  try {
    // --- 实际调用 Rust 后端命令 ---
    const result = await invoke<string>("transcribe_audio", {
      filePath: filePath,
    }); // 传递参数
    console.log("后端 transcribe_audio 命令成功，结果：", result);
    transcription.value = result; // 显示转录结果
    currentState.value = "success"; // 设置状态为成功

    // --- 模拟 ---
    // await new Promise(resolve => setTimeout(resolve, 2500));
    // const mockResult = `这是从 "${selectedFileName.value || '未知文件'}" 转录的模拟文本。\n时间: ${new Date().toLocaleTimeString()}`;
    // transcription.value = mockResult;
    // currentState.value = 'success';
  } catch (err) {
    console.error("调用 transcribe_audio 命令失败:", err);
    errorMessage.value = `处理音频失败: ${err instanceof Error ? err.message : String(err)}`;
    currentState.value = "error";
  }
}

// --- 文件拖放事件监听 (使用 Tauri 内置事件) ---
let unlistenDrop: (() => void) | null = null;
let unlistenHover: (() => void) | null = null;
let unlistenCancel: (() => void) | null = null;

// 在组件挂载时设置监听器
onMounted(async () => {
  try {
    // 监听文件悬停在窗口上方的事件
    unlistenHover = await listen<string[]>("tauri://drag-over", () => {
      // 只有在非录音/处理状态下才响应拖放悬停
      if (
        currentState.value !== "recording" &&
        currentState.value !== "processing"
      ) {
        isDragging.value = true; // 更新状态以显示视觉反馈
      }
    });

    // 监听文件在窗口上方释放 (放下) 的事件
    unlistenDrop = await listen<string[]>("tauri://drag-drop", (event) => {
      isDragging.value = false; // 移除视觉反馈
      // 只有在非录音/处理状态下才处理文件
      if (
        currentState.value !== "recording" &&
        currentState.value !== "processing"
      ) {
        const filePath = event.payload[0]; // 获取第一个拖放文件的路径
        if (filePath) {
          // 简单地基于文件扩展名检查文件类型 (后端最好做更严格的检查)
          const lowerPath = filePath.toLowerCase();
          if (
            lowerPath.endsWith(".wav") ||
            lowerPath.endsWith(".mp3") ||
            lowerPath.endsWith(".m4a") ||
            lowerPath.endsWith(".ogg") ||
            lowerPath.endsWith(".flac")
          ) {
            selectedFileName.value = filePath.split(/[\\/]/).pop() ?? filePath; // 显示文件名
            processAudio(filePath); // 使用真实路径处理
          } else {
            errorMessage.value = "请拖放有效的音频文件";
            currentState.value = "error";
            console.warn("拖放的文件类型不被支持:", filePath);
          }
        }
      }
    });

    // 监听拖放操作被取消的事件 (例如文件拖出窗口)
    unlistenCancel = await listen<null>("tauri://drag-leave", () => {
      isDragging.value = false; // 移除视觉反馈
    });
  } catch (error) {
    console.error("设置 Tauri 文件拖放监听器失败 (v2):", error);
    // 这里可以考虑通知用户拖放功能可能无法使用
  }
});

// 在组件卸载时清理监听器，防止内存泄漏
onUnmounted(() => {
  unlistenDrop?.();
  unlistenHover?.();
  unlistenCancel?.();
});
</script>

<template>
  <div
    class="flex flex-col items-center min-h-screen pt-16 pb-8 px-4 space-y-6 bg-apple-gray-100 dark:bg-apple-gray-850"
  >
    <h1
      class="text-2xl font-semibold text-apple-gray-950 dark:text-apple-gray-50"
    >
      Whisper ASR 转录
    </h1>

    <div
      @click="handleSelectFile"
      class="w-full max-w-lg p-6 border-2 border-dashed rounded-lg transition-colors duration-200 text-center space-y-3"
      :class="{
        'border-apple-gray-300 dark:border-apple-gray-700': !isDragging,
        'border-apple-blue bg-apple-blue/10 dark:bg-apple-blue/20': isDragging,
        'cursor-pointer hover:border-apple-gray-400 dark:hover:border-apple-gray-600':
          currentState !== 'recording' &&
          currentState !== 'processing' &&
          !isDragging,
        'opacity-60 cursor-not-allowed':
          currentState === 'recording' || currentState === 'processing',
      }"
    >
      <ArrowUpTrayIcon
        class="w-10 h-10 mx-auto text-apple-gray-500 dark:text-apple-gray-400"
      />
      <p class="text-apple-gray-600 dark:text-apple-gray-300 text-sm">
        拖放音频文件到此区域，或
        <span class="text-apple-blue font-medium">点击选择文件</span>
      </p>
      <p
        v-if="selectedFileName && currentState !== 'idle'"
        class="text-xs text-apple-gray-500 dark:text-apple-gray-400 mt-1 truncate px-2"
      >
        当前: {{ selectedFileName }}
      </p>
    </div>

    <div class="flex items-center w-full max-w-xs">
      <hr
        class="flex-grow border-t border-apple-gray-200 dark:border-apple-gray-700"
      />
      <span class="px-3 text-apple-gray-500 dark:text-apple-gray-400 text-xs"
        >或</span
      >
      <hr
        class="flex-grow border-t border-apple-gray-200 dark:border-apple-gray-700"
      />
    </div>

    <div class="flex flex-col items-center">
      <BaseButton
        @click="handleRecord"
        :disabled="
          currentState === 'processing' || currentState === 'selecting'
        "
        :variant="currentState === 'recording' ? 'danger' : 'primary'"
        rounded="full"
        size="lg"
        class="shadow-md hover:shadow-lg"
      >
        <template #icon>
          <StopCircleIcon v-if="currentState === 'recording'" class="w-5 h-5" />
          <MicrophoneIcon v-else class="w-5 h-5" />
        </template>
        {{ currentState === "recording" ? "停止录音" : "开始录音" }}
      </BaseButton>
    </div>

    <div class="h-6 pt-1 pb-1">
      <StatusIndicator :state="currentState" />
    </div>

    <div
      v-if="currentState === 'error' && errorMessage"
      class="w-full max-w-lg p-3 bg-red-100/80 dark:bg-apple-red/20 border border-apple-red/50 dark:border-apple-red/40 rounded-md text-red-700 dark:text-red-200 flex items-start space-x-2 text-sm"
    >
      <ExclamationCircleIcon
        class="w-4 h-4 flex-shrink-0 mt-0.5 text-apple-red"
      />
      <span>{{ errorMessage }}</span>
    </div>

    <div
      v-if="currentState === 'success' || currentState === 'processing'"
      class="w-full max-w-lg p-4 bg-white dark:bg-apple-gray-800 rounded-lg shadow-sm border border-apple-gray-200 dark:border-apple-gray-700"
    >
      <h2
        class="text-base font-medium text-apple-gray-700 dark:text-apple-gray-200 mb-2 flex items-center space-x-2"
      >
        <DocumentTextIcon class="w-4 h-4" />
        <span>转录结果</span>
      </h2>
      <textarea
        readonly
        :value="transcription"
        class="w-full h-40 p-2 bg-apple-gray-100 dark:bg-apple-gray-700 rounded-md text-apple-gray-900 dark:text-apple-gray-100 focus:outline-none focus:ring-1 focus:ring-apple-blue/50 resize-none font-mono text-xs leading-relaxed"
        placeholder="转录文本将显示在这里..."
      ></textarea>
      <p
        v-if="currentState === 'processing'"
        class="text-xs text-apple-gray-500 dark:text-apple-gray-400 mt-2 animate-pulse text-center"
      >
        {{ transcription ? "正在处理..." : "正在处理音频..." }}
      </p>
    </div>
  </div>
</template>

<script setup lang="ts">
import { onMounted, onUnmounted, ref } from "vue";
// 导入 Tauri webviewWindow API 用于主题处理
import { getCurrentWindow, type Theme } from "@tauri-apps/api/Window";

// 当前主题状态 ('light' | 'dark' | null)
const currentTheme = ref<Theme | null>(null);

// 应用主题变化的函数
function applyTheme(theme: Theme | null) {
  const root = document.documentElement; // 获取 <html> 元素
  if (theme === "dark") {
    root.classList.add("dark"); // 添加 dark 类启用暗色模式
    console.log("应用暗色主题 (Tauri v2)");
  } else {
    root.classList.remove("dark"); // 移除 dark 类启用亮色模式
    console.log("应用亮色主题 (Tauri v2)");
  }
  currentTheme.value = theme; // 更新 Vue ref 状态
}

// 组件挂载后执行的逻辑
onMounted(async () => {
  // 获取当前的 Tauri 窗口实例
  const currentWindow = getCurrentWindow();

  try {
    // 获取并应用初始系统主题
    const initialTheme = await currentWindow.theme();
    applyTheme(initialTheme);

    // 监听系统主题变化事件
    const unlisten = await currentWindow.onThemeChanged(
      ({ payload: theme }) => {
        console.log(`系统主题变更为: ${theme}`);
        applyTheme(theme); // 当系统主题改变时，自动应用新主题
      },
    );

    onUnmounted(() => {
      unlisten();
    });
  } catch (error) {
    // 如果获取或监听失败，提供一个默认值并打印错误
    console.error("获取或监听系统主题失败 (Tauri v2):", error);
    applyTheme("light"); // 默认回退到亮色主题
  }
});

// 可选的手动切换主题逻辑 (当前未使用)
// function toggleTheme() {
//   const newTheme = currentTheme.value === 'dark' ? 'light' : 'dark';
//   applyTheme(newTheme);
// }
</script>

<template>
  <div class="min-h-screen font-sans">
    <router-view v-slot="{ Component }">
      <transition name="fade" mode="out-in">
        <component :is="Component" />
      </transition>
    </router-view>
  </div>
</template>

<style>
/* 定义页面切换的淡入淡出效果 */
.fade-enter-active,
.fade-leave-active {
  transition: opacity 0.2s ease;
}

.fade-enter-from,
.fade-leave-to {
  opacity: 0;
}

/* 确保 html 和 body 高度充满，背景色过渡更平滑 */
html,
body {
  height: 100%;
  margin: 0;
  padding: 0;
  /* 应用背景色过渡动画 */
  transition-property: background-color, color; /* 指定要过渡的属性 */
  transition-timing-function: ease-in-out; /* 过渡曲线 */
  transition-duration: 300ms; /* 过渡时间 */
}

/* 自定义 Webkit 滚动条样式，使其更像 macOS */
::-webkit-scrollbar {
  width: 6px; /* 滚动条宽度 */
  height: 6px; /* 水平滚动条高度 */
}

::-webkit-scrollbar-track {
  background: transparent; /* 轨道背景透明 */
}

::-webkit-scrollbar-thumb {
  background-color: rgba(136, 136, 136, 0.5); /* 滚动条滑块颜色 (半透明灰色) */
  border-radius: 10px; /* 圆角 */
  border: 1px solid transparent; /* 透明边框，防止背景裁剪 */
  background-clip: content-box; /* 让背景色在边框内 */
}

::-webkit-scrollbar-thumb:hover {
  background-color: rgba(100, 100, 100, 0.7); /* 悬停时颜色加深 */
}
</style>

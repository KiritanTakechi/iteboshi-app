<script setup lang="ts">
import { computed } from "vue";
// 引入所需图标
import {
  CheckCircleIcon,
  ArrowPathIcon,
  XCircleIcon,
  MicrophoneIcon,
  InformationCircleIcon,
} from "@heroicons/vue/24/solid";

// 定义应用状态类型 (与 HomeView 保持一致)
type AppState =
  | "idle"
  | "selecting"
  | "recording"
  | "processing"
  | "success"
  | "error";

// 定义组件接收的属性
interface Props {
  state: AppState; // 当前状态
}
// 接收父组件传递的 state 属性
const props = defineProps<Props>();

// 根据当前状态计算显示的文本、图标、颜色和动画效果
const statusInfo = computed(() => {
  switch (props.state) {
    case "selecting":
      // 选择中状态: 蓝色，信息图标，图标旋转
      return {
        text: "选择文件中...",
        icon: InformationCircleIcon,
        color: "text-apple-blue",
        pulseIcon: true,
      };
    case "recording":
      // 录音中状态: 红色，麦克风图标，文本脉动
      return {
        text: "正在录音...",
        icon: MicrophoneIcon,
        color: "text-apple-red",
        pulseIcon: false,
        pulseText: true,
      };
    case "processing":
      // 处理中状态: 橙色，箭头图标，图标旋转
      return {
        text: "正在处理...",
        icon: ArrowPathIcon,
        color: "text-apple-orange",
        pulseIcon: true,
      };
    case "success":
      // 成功状态: 绿色，勾选图标
      return {
        text: "处理完成",
        icon: CheckCircleIcon,
        color: "text-apple-green",
      };
    case "error":
      // 错误状态: 红色，叉号图标
      return { text: "出现错误", icon: XCircleIcon, color: "text-apple-red" };
    case "idle":
    default: // 默认空闲状态: 灰色，无图标
      return {
        text: "等待操作",
        icon: null,
        color: "text-apple-gray-500 dark:text-apple-gray-400",
      };
  }
});
</script>

<template>
  <div
    v-if="statusInfo.text !== '等待操作'"
    class="flex items-center space-x-1.5 text-xs"
    :class="[statusInfo.color, { 'animate-pulse': statusInfo.pulseText }]"
  >
    <component
      v-if="statusInfo.icon"
      :is="statusInfo.icon"
      class="w-4 h-4 flex-shrink-0"
      :class="{ 'animate-spin': statusInfo.pulseIcon }"
    />
    <span class="font-medium">{{ statusInfo.text }}</span>
  </div>
  <div v-else class="text-xs" :class="statusInfo.color">
    {{ statusInfo.text }}
  </div>
</template>

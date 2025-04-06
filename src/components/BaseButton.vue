<script setup lang="ts">
import { computed, useSlots } from "vue";

interface Props {
  variant?: "primary" | "secondary" | "danger" | "plain";
  size?: "sm" | "md" | "lg";
  rounded?: "none" | "sm" | "md" | "lg" | "full";
  disabled?: boolean;
  fullWidth?: boolean;
}

const props = withDefaults(defineProps<Props>(), {
  variant: "primary",
  size: "md",
  // 修正：设置一个标准的 Tailwind 圆角类作为默认值
  // 'sm' 对应 v4 的 rounded-sm (即 0.125rem)
  // 'md' 对应 v4 的 rounded-md (即 0.375rem)
  // 如果想默认接近 6px，'md' 可能更接近
  rounded: "md", // 或者 'sm' 如果更喜欢小的默认圆角
  disabled: false,
  fullWidth: false,
});

const slots = useSlots();

const buttonClasses = computed(() => {
  const base = `inline-flex items-center justify-center font-medium
                   focus:outline-none focus:ring-2 focus:ring-apple-blue/60 focus:ring-offset-1 dark:focus:ring-offset-black
                   transition-all duration-150 ease-in-out
                   disabled:opacity-60 disabled:cursor-not-allowed`;

  let variantClasses = "";
  // ... (variant logic 不变) ...
  switch (props.variant) {
    case "secondary":
      variantClasses = `bg-apple-gray-200 dark:bg-apple-gray-700 hover:bg-apple-gray-300 dark:hover:bg-apple-gray-600 text-apple-gray-950 dark:text-apple-gray-50 shadow-xs hover:shadow-sm`;
      break;
    case "danger":
      variantClasses = `bg-apple-red hover:bg-red-600 dark:hover:bg-red-500 text-white shadow-sm hover:shadow-md`;
      break;
    case "plain":
      variantClasses = `bg-transparent text-apple-blue hover:text-apple-blue-darker dark:hover:text-apple-blue-lighter`;
      break;
    case "primary":
    default:
      variantClasses = `bg-apple-blue hover:bg-opacity-85 active:bg-apple-blue-darker text-white shadow-sm hover:shadow-md`;
      break;
  }

  let sizeClasses = "";
  // ... (size logic 不变) ...
  switch (props.size) {
    case "sm":
      sizeClasses = "px-3 py-1 text-xs";
      break;
    case "lg":
      sizeClasses = "px-5 py-2.5 text-base";
      break;
    case "md":
    default:
      sizeClasses = "px-4 py-1.5 text-sm";
      break;
  }

  const iconSpacing = slots.icon && slots.default ? "space-x-1.5" : "";

  // 修正：roundedMap 只包含有效的 Tailwind 类
  const roundedMap = {
    none: "rounded-none",
    sm: "rounded-sm", // v4 的 rounded-sm (0.125rem)
    md: "rounded-md", // v4 的 rounded-md (0.375rem)
    lg: "rounded-lg", // v4 的 rounded-lg (0.5rem)
    full: "rounded-full",
    // 移除非 Tailwind 类的映射
    // apple: 'rounded-apple', // 移除
    // 'apple-sm': 'rounded-apple-sm', // 移除
    // 'apple-lg': 'rounded-apple-lg', // 移除
  };
  // 确保 props.rounded 的值是 map 中的 key
  const roundedClasses = props.rounded ? roundedMap[props.rounded] : "";

  const widthClasses = props.fullWidth ? "w-full" : "";

  return [
    base,
    variantClasses,
    sizeClasses,
    roundedClasses,
    iconSpacing,
    widthClasses,
  ];
});
</script>

<template>
  <button :class="buttonClasses" :disabled="props.disabled">
    <span v-if="slots.icon" :class="{ 'mr-1.5': slots.default }">
      <slot name="icon"></slot>
    </span>
    <span v-if="slots.default">
      <slot></slot>
    </span>
  </button>
</template>

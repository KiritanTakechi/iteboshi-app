import { createRouter, createWebHistory } from "vue-router";
import HomeView from "../views/HomeView.vue"; // 引入主视图组件

// 创建路由实例
const router = createRouter({
  // 使用 Web History 模式 (Tauri 推荐)
  history: createWebHistory(import.meta.env.BASE_URL),
  // 定义路由规则
  routes: [
    {
      path: "/", // 根路径
      name: "home", // 路由名称
      component: HomeView, // 对应的组件
    },
    // 未来可以添加其他路由配置
    // {
    //   path: '/settings',
    //   name: 'settings',
    //   component: () => import('../views/SettingsView.vue') // 懒加载示例
    // }
  ],
});

// 导出路由实例
export default router;

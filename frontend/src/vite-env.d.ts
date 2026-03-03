import { defineComponent } from 'vue'

declare module '*.vue' {
  const component: defineComponent<{}, {}, any>
  export default component
}



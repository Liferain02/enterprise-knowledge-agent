#!/usr/bin/env python
"""
进一步测试 Mem0 检索结果
"""
import asyncio
import sys
import os

# 设置代理
os.environ["http_proxy"] = "http://127.0.0.1:7897"
os.environ["https_proxy"] = "http://127.0.0.1:7897"

# 添加项目路径
sys.path.insert(0, "/share/home/lifr/workspace/code/enterprise-knowledge-agent")

async def test_mem0_detail():
    """测试 Mem0 检索详情"""
    print("=" * 60)
    print("测试 Mem0 检索详情...")
    print("=" * 60)
    
    try:
        from src.agent.memory.mem0_manager import get_mem0_manager
        
        # 获取 Mem0 管理器
        mem0_manager = get_mem0_manager(provider="qwen")
        
        # 测试检索 - 打印完整返回
        print("\n检索 '红色书籍' 相关记忆...")
        search_results = await mem0_manager.search(
            query="红色书籍",
            user_id="test_user_001",
            limit=5
        )
        
        print(f"\n检索结果 (共 {len(search_results)} 条):")
        print("-" * 40)
        for i, mem in enumerate(search_results):
            print(f"\n结果 {i+1}:")
            for key, value in mem.items():
                print(f"  {key}: {value}")
        
        # 测试格式化记忆
        print("\n" + "=" * 60)
        print("测试格式化记忆:")
        print("=" * 60)
        formatted = mem0_manager.format_memories_for_context(search_results)
        print(formatted)
        
        print("\n✅ 详细测试完成!")
        return True
            
    except Exception as e:
        print(f"\n❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    result = asyncio.run(test_mem0_detail())
    sys.exit(0 if result else 1)
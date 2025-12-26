"""
测试阿里云API配置和基本功能
"""
import os
from dotenv import load_dotenv

# 加载环境变量
load_dotenv()

def test_env_variables():
    """测试环境变量配置"""
    print("=" * 50)
    print("测试环境变量配置")
    print("=" * 50)
    
    api_key = os.getenv("DASHSCOPE_API_KEY")
    if api_key:
        print(f"✓ DASHSCOPE_API_KEY: {api_key[:10]}...{api_key[-4:]}")
    else:
        print("✗ DASHSCOPE_API_KEY 未设置")
        return False
    
    embedding_model = os.getenv("EMBEDDING_MODEL", "text-embedding-v3")
    print(f"✓ EMBEDDING_MODEL: {embedding_model}")
    
    llm_model = os.getenv("LLM_MODEL", "qwen-plus")
    print(f"✓ LLM_MODEL: {llm_model}")
    
    threshold = os.getenv("RELEVANCE_THRESHOLD", "0.5")
    print(f"✓ RELEVANCE_THRESHOLD: {threshold}")
    
    return True

def test_embedding():
    """测试Embedding功能"""
    print("\n" + "=" * 50)
    print("测试Embedding功能")
    print("=" * 50)
    
    try:
        from langchain_openai import OpenAIEmbeddings
        from config import DASHSCOPE_API_KEY, DASHSCOPE_BASE_URL, EMBEDDING_MODEL
        
        embeddings = OpenAIEmbeddings(
            model=EMBEDDING_MODEL,
            openai_api_key=DASHSCOPE_API_KEY,
            openai_api_base=DASHSCOPE_BASE_URL
        )
        
        # 测试生成embedding
        test_text = "这是一个测试文本"
        print(f"测试文本: {test_text}")
        
        result = embeddings.embed_query(test_text)
        print(f"✓ Embedding生成成功")
        print(f"  维度: {len(result)}")
        print(f"  前5个值: {result[:5]}")
        
        return True
        
    except Exception as e:
        print(f"✗ Embedding测试失败: {e}")
        return False

def test_llm():
    """测试LLM功能"""
    print("\n" + "=" * 50)
    print("测试LLM功能")
    print("=" * 50)
    
    try:
        from langchain_openai import ChatOpenAI
        from config import (
            DASHSCOPE_API_KEY, DASHSCOPE_BASE_URL, LLM_MODEL,
            LLM_TEMPERATURE, LLM_TOP_P, LLM_MAX_TOKENS
        )
        
        llm = ChatOpenAI(
            model=LLM_MODEL,
            openai_api_key=DASHSCOPE_API_KEY,
            openai_api_base=DASHSCOPE_BASE_URL,
            temperature=LLM_TEMPERATURE,
            top_p=LLM_TOP_P,
            max_tokens=LLM_MAX_TOKENS
        )
        
        # 测试生成回答
        test_prompt = "请用一句话介绍你自己。"
        print(f"测试提示: {test_prompt}")
        
        response = llm.invoke(test_prompt)
        print(f"✓ LLM调用成功")
        print(f"  回答: {response.content}")
        
        return True
        
    except Exception as e:
        print(f"✗ LLM测试失败: {e}")
        return False

def test_document_splitter():
    """测试文档切割器"""
    print("\n" + "=" * 50)
    print("测试文档切割器")
    print("=" * 50)
    
    try:
        from document_splitter import create_hr_splitter
        
        splitter = create_hr_splitter()
        
        # 测试文本
        test_text = """一、薪资管理制度

1. 薪资构成
员工薪资由基本工资、绩效奖金、津贴补贴等组成。

2. 薪资发放
每月15日发放上月薪资。

二、考勤管理制度

1. 打卡要求
员工需每日上下班打卡。

2. 请假流程
请假需提前申请，经主管批准后生效。"""
        
        print(f"测试文本长度: {len(test_text)} 字符")
        
        chunks = splitter.split_text(test_text)
        print(f"✓ 文档切割成功")
        print(f"  切割块数: {len(chunks)}")
        for i, chunk in enumerate(chunks, 1):
            print(f"\n  块 {i} ({len(chunk)} 字符):")
            print(f"  {chunk[:100]}...")
        
        return True
        
    except Exception as e:
        print(f"✗ 文档切割器测试失败: {e}")
        return False

def test_relevance_evaluator():
    """测试相关性评估器"""
    print("\n" + "=" * 50)
    print("测试相关性评估器")
    print("=" * 50)
    
    try:
        from relevance_evaluator import create_relevance_evaluator
        from langchain.schema import Document
        
        evaluator = create_relevance_evaluator()
        
        # 创建测试文档
        docs = [
            Document(page_content="薪资发放时间", metadata={'score': 0.8}),
            Document(page_content="考勤管理制度", metadata={'score': 0.6}),
            Document(page_content="培训发展计划", metadata={'score': 0.3}),
        ]
        
        query = "薪资什么时候发放？"
        print(f"测试查询: {query}")
        
        result = evaluator.evaluate(query, docs)
        print(f"✓ 相关性评估成功")
        print(f"  是否相关: {result['is_relevant']}")
        print(f"  最高分数: {result['max_score']:.3f}")
        print(f"  平均分数: {result['avg_score']:.3f}")
        print(f"  相关文档数: {result['relevant_count']}/{len(docs)}")
        
        return True
        
    except Exception as e:
        print(f"✗ 相关性评估器测试失败: {e}")
        return False

def main():
    """运行所有测试"""
    print("\n" + "=" * 50)
    print("阿里云API配置和功能测试")
    print("=" * 50 + "\n")
    
    results = []
    
    # 1. 测试环境变量
    results.append(("环境变量配置", test_env_variables()))
    
    # 2. 测试Embedding
    results.append(("Embedding功能", test_embedding()))
    
    # 3. 测试LLM
    results.append(("LLM功能", test_llm()))
    
    # 4. 测试文档切割器
    results.append(("文档切割器", test_document_splitter()))
    
    # 5. 测试相关性评估器
    results.append(("相关性评估器", test_relevance_evaluator()))
    
    # 汇总结果
    print("\n" + "=" * 50)
    print("测试结果汇总")
    print("=" * 50)
    
    for name, result in results:
        status = "✓ 通过" if result else "✗ 失败"
        print(f"{name}: {status}")
    
    total = len(results)
    passed = sum(1 for _, r in results if r)
    print(f"\n总计: {passed}/{total} 通过")
    
    if passed == total:
        print("\n🎉 所有测试通过！系统配置正确。")
    else:
        print("\n⚠️ 部分测试失败，请检查配置。")

if __name__ == "__main__":
    main()

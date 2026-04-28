from dotenv import load_dotenv
import os

# 必须在import之前加载环境变量
load_dotenv()

from agent import DeepResearchAgent
import time
import asyncio


def main():
    print("====== 🚀 深度研究助手启动 ======")
    researcher = DeepResearchAgent()
    topic = "卷积神经网络的最新进展"

    # 这一步会直接在终端里跑出结果，根本不需要浏览器和前端！
    report = researcher.run(topic=topic)
    print("\n最终报告:\n", report)


if __name__ == "__main__":
    start_time = time.time()
    print(f"程序启动，当前时间为{start_time}")
    main()
    end_time = time.time()
    print(f"程序结束，当前时间为{end_time}")
    print(f"共用时间为{end_time - start_time}秒")


import json
from flask import Flask, request, Response, jsonify
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
chat_model = ChatOpenAI(
    model="qwen2.5-fuchanke",  # 本地模型路径 vllm部署后model调用的名字
    api_key="EMPTY",  # 本地调用不需要 API key
    openai_api_base="http://localhost:20202/v1"  # 本地 API 基础地址22222为你vllm部署的时候自定义的端口号
)
app = Flask(__name__)
@app.route('/chat', methods=['POST'])
def chat_api():
    # 从请求中获取数据
    data = request.json
    request_id = data.get('request_id')
    phone_number = data.get('phone_number')
    query = data.get('query')
    logger.info("正在处理请求: %s", request_id)
    logger.info("正在处理用户: %s", phone_number)
    logger.info("正在处理问题: %s", query)
    if not request_id or not phone_number or not query:
        return jsonify({"error": "Missing required fields"}), 400
    # 创建一个生成器来逐块返回数据
    def generate_response():
        for chunk in chat_model.stream([HumanMessage(content=query)]):
            json_chunk = json.dumps({
                "request_id": request_id,
                "phone_number": phone_number,
                "response": chunk.content  
            })
            yield json_chunk + '\n'  # 每个 JSON 块后面加上换行符
    # 使用 Flask 的 Response 对象来返回流式响应
    return Response(generate_response(), content_type='application/json')
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=42356, threaded=True)

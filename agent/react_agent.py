from langchain_core.messages import SystemMessage, ToolMessage
from langgraph.graph import StateGraph, MessagesState, START, END
from langgraph.prebuilt import ToolNode, tools_condition
from model.factory import chat_model
from utils.prompt_loader import load_system_prompts, load_report_prompts
from agent.tools.agent_tools import (rag_summarize, get_weather, get_user_location, get_user_id,
                                     get_current_month, fetch_external_data, fill_context_for_report)
from utils.logger_handler import logger


class ReactAgent:
    def __init__(self):
        self.tools = [rag_summarize, get_weather, get_user_location, get_user_id,
                      get_current_month, fetch_external_data, fill_context_for_report]
        self.model_with_tools = chat_model.bind_tools(self.tools)
        self.system_prompt = load_system_prompts()
        self.graph = self._build_graph()

    def _call_model(self, state: MessagesState):
        messages = state["messages"]

        # 检测是否已调用 fill_context_for_report，切换到报告提示词
        report_triggered = any(
            isinstance(m, ToolMessage) and hasattr(m, 'name') and m.name == "fill_context_for_report"
            for m in messages
        )
        prompt = load_report_prompts() if report_triggered else self.system_prompt

        if not any(isinstance(m, SystemMessage) for m in messages):
            messages = [SystemMessage(content=prompt)] + messages

        response = self.model_with_tools.invoke(messages)
        logger.info(f"[model]调用模型，返回 {len(response.content)} 字符")
        return {"messages": [response]}

    def _build_graph(self):
        tool_node = ToolNode(self.tools)

        graph = StateGraph(MessagesState)
        graph.add_node("agent", self._call_model)
        graph.add_node("tools", tool_node)
        graph.add_edge(START, "agent")
        graph.add_conditional_edges("agent", tools_condition, {"tools": "tools", END: END})
        graph.add_edge("tools", "agent")

        return graph.compile()

    def execute_stream(self, query: str):
        input_dict = {
            "messages": [
                {"role": "user", "content": query},
            ]
        }

        for chunk in self.graph.stream(input_dict, stream_mode="values"):
            latest_message = chunk["messages"][-1]
            if latest_message.content:
                yield latest_message.content.strip() + "\n"


if __name__ == '__main__':
    agent = ReactAgent()

    for chunk in agent.execute_stream("给我生成我的使用报告"):
        print(chunk, end="", flush=True)

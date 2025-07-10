from typing import List, Sequence, cast

import chainlit as cl
import yaml

from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.teams import SelectorGroupChat
from autogen_agentchat.messages import TextMessage, ModelClientStreamingChunkEvent, BaseAgentEvent, BaseChatMessage
from autogen_core.models import ChatCompletionClient
from autogen_core import CancellationToken


@cl.step(type="tool")
async def search_web(query: str) -> str:
    return f"🌐 检索结果：'{query}' 的最新网页摘要如下……"

@cl.step(type="tool")
async def analyze_data(data: str) -> str:
    return f"📊 针对数据'{data}'的初步分析结果：……"


def selector_func(messages: Sequence[BaseAgentEvent | BaseChatMessage]) -> str | None:
    MAX_TURNS = 6
    print("message_len")
    print(len(messages))
    if len(messages) == 1:
        return "InputRefiner"
    if len(messages) == MAX_TURNS - 1:
        return "OutputSummarizer"
    return None



@cl.on_chat_start  # type: ignore
async def start_chat() -> None:
    with open("model_config.yaml", "r") as f:
        model_cfg = yaml.safe_load(f)
    model_client = ChatCompletionClient.load_component(model_cfg)

    input_refiner = AssistantAgent(
        name="InputRefiner",
        system_message="你善于将用户输入精炼为简明、结构化、信息密度高的任务描述。必须注意：你的发言是高度概括性的，只需要一句话，一般不超过20个字。",
        model_client=model_client,
        model_client_stream=True,
        reflect_on_tool_use=False,
    )

    info_retriever = AssistantAgent(
        name="InfoRetriever",
        system_message="你善于检索与任务相关的知识、实例、数据，必要时可调用search_web工具。",
        tools=[search_web],
        model_client=model_client,
        model_client_stream=True,
        reflect_on_tool_use=True,
    )

    analyst = AssistantAgent(
        name="Analyst",
        system_message="你擅长对给定任务或信息进行条理清晰的分析，可调用analyze_data工具协助判断。",
        tools=[analyze_data],
        model_client=model_client,
        model_client_stream=True,
        reflect_on_tool_use=True,
    )

    output_summarizer = AssistantAgent(
        name="OutputSummarizer",
        system_message="你不直接参与和其他agent的交流，你只需要对目前上下文中的其他团队成员给出的输出做出系统性的总结，需要是有条理的，易于理解的。",
        model_client=model_client,
        model_client_stream=True,
        reflect_on_tool_use=False,
    )

    team = SelectorGroupChat(
        [input_refiner, info_retriever, analyst, output_summarizer],
        model_client=model_client,
        selector_func=selector_func,  # 首尾定序，中间自由
        max_turns=6,
    )

    cl.user_session.set("team", team)  # type: ignore


@cl.set_starters  # type: ignore
async def set_starts() -> List[cl.Starter]:
    return [
        cl.Starter(
            label="法律咨询",
            message="我最近被公司解雇，对方没有提前一个月通知我，只支付了一个月工资补偿，请问我能否要求更多补偿？有哪些相关的法律依据和案例？我需要注意哪些风险？"
        ),
        cl.Starter(
            label="旅游攻略",
            message="我想去云南自由行5天，能帮我设计一份详细路线和注意事项吗？"
        ),
        cl.Starter(
            label="数据分析",
            message="请帮我分析一份销售数据，给出增长瓶颈和改进建议。原始数据如下：......"
        ),
    ]


@cl.on_message
async def chat(message: cl.Message) -> None:
    user_text = message.content
    team = cast(SelectorGroupChat, cl.user_session.get("team"))

    msg = None
    input_refiner_content = ""

    async for evt in team.run_stream(
        task=user_text,
        cancellation_token=CancellationToken(),
    ):
        agent_name = getattr(evt, "source", None) or getattr(getattr(evt, "chat_message", None), "source", None)
        print("agent_name")
        print(agent_name)

        if agent_name == "InputRefiner":
            if hasattr(evt, "content") and isinstance(evt.content, str):
                input_refiner_content += evt.content
            elif hasattr(evt, "content"):
                with open("input_refiner.txt", "a", encoding="utf-8") as f:
                    f.write(input_refiner_content.strip() + "\n")
                input_refiner_content = ""

        elif agent_name == "OutputSummarizer":
            if msg is None:
                msg = cl.Message(author="OutputSummarizer", content="")
            if hasattr(evt, "content") and isinstance(evt.content, str):
                await msg.stream_token(evt.content)
            elif hasattr(evt, "content"):
                await msg.send()

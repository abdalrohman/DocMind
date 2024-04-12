import logging
from operator import itemgetter
from typing import Dict, List, Optional, Sequence

from langchain_core.documents import Document
from langchain_core.language_models import LanguageModelLike
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import (
    ChatPromptTemplate,
    MessagesPlaceholder,
    PromptTemplate,
)
from langchain_core.pydantic_v1 import BaseModel
from langchain_core.retrievers import BaseRetriever
from langchain_core.runnables import (
    ConfigurableField,
    Runnable,
    RunnableBranch,
    RunnableLambda,
    RunnablePassthrough,
)

from libs.prompts import RAG_TEMPLATE, REPHRASE_TEMPLATE
from libs.search_tools import Search

logger = logging.getLogger(__name__)


class ChatRequest(BaseModel):
    question: str
    chat_history: Optional[List[Dict[str, str]]]


def serialize_history(request: ChatRequest):
    chat_history = request["chat_history"] or []
    converted_chat_history = []

    for message in chat_history:
        if isinstance(message, HumanMessage) or isinstance(message, AIMessage):
            converted_chat_history.append(message)
        else:
            if message.get("human") is not None:
                converted_chat_history.append(HumanMessage(content=message["human"]))
            if message.get("ai") is not None:
                converted_chat_history.append(AIMessage(content=message["ai"]))
    return converted_chat_history


def create_retriever_chain(
    llm: LanguageModelLike, retriever: BaseRetriever
) -> Runnable:
    CONDENSE_QUESTION_PROMPT = PromptTemplate.from_template(REPHRASE_TEMPLATE)
    condense_question_chain = (
        CONDENSE_QUESTION_PROMPT | llm | StrOutputParser()
    ).with_config(
        run_name="CondenseQuestion",
    )
    conversation_chain = condense_question_chain | retriever
    return RunnableBranch(
        (
            RunnableLambda(lambda x: bool(x.get("chat_history"))).with_config(
                run_name="HasChatHistoryCheck"
            ),
            conversation_chain.with_config(run_name="RetrievalChainWithHistory"),
        ),
        (
            RunnableLambda(itemgetter("question")).with_config(
                run_name="Itemgetter:question"
            )
            | retriever
        ).with_config(run_name="RetrievalChainWithNoHistory"),
    ).with_config(run_name="RouteDependingOnChatHistory")


def format_docs(docs: Sequence[Document]) -> str:
    formatted_docs = []
    for i, doc in enumerate(docs):
        doc_string = f"<doc id='{i}'>{doc.page_content}</doc>"
        formatted_docs.append(doc_string)
    return "\n".join(formatted_docs)


def create_chain(llm: LanguageModelLike, retriever: BaseRetriever) -> Runnable:
    retriever_chain = create_retriever_chain(
        llm,
        retriever,
    ).with_config(run_name="FindDocs")
    context = (
        RunnablePassthrough.assign(docs=retriever_chain)
        .assign(context=lambda x: format_docs(x["docs"]))
        .with_config(run_name="RetrieveDocs")
    )
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", RAG_TEMPLATE),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{question}"),
        ]
    )
    default_response_synthesizer = prompt | llm

    response_synthesizer = (
        default_response_synthesizer.configurable_alternatives(
            ConfigurableField("llm"),
            default_key="chat_fireworks",
            gemini_pro=default_response_synthesizer,
            groq=default_response_synthesizer,
            openai=default_response_synthesizer,
        )
        | StrOutputParser()
    ).with_config(run_name="GenerateResponse")
    return (
        RunnablePassthrough.assign(chat_history=serialize_history)
        | context
        | response_synthesizer
    )


# -- SEARCH CAIN --
def format_search_result(result: List[Dict]) -> str:
    formatted_result = []

    for res in result:
        result_string = f"<res url='{res['url']}'>{res['content']}</res>"
        formatted_result.append(result_string)
    return "\n".join(formatted_result)


def search_result_chain(
    llm: LanguageModelLike,
    search_engine: Search,
) -> Runnable:
    condense_question_prompt = PromptTemplate.from_template(REPHRASE_TEMPLATE)
    condense_question_chain = (
        condense_question_prompt | llm | StrOutputParser()
    ).with_config(
        run_name="CondenseQuestion",
    )

    chain = condense_question_chain | search_engine

    return RunnableBranch(
        (
            RunnableLambda(lambda x: bool(x.get("chat_history"))).with_config(
                run_name="HasChatHistoryCheck"
            ),
            chain.with_config(run_name="RetrievalChainWithHistory"),
        ),
        (
            RunnableLambda(itemgetter("question")).with_config(
                run_name="Itemgetter:question"
            )
            | search_engine
        ).with_config(run_name="RetrievalChainWithNoHistory"),
    ).with_config(run_name="RouteDependingOnChatHistory")


def create_search_chain(
    llm: LanguageModelLike,
    search_engine: Search = Search(search_engine="google", max_num_results=10),
) -> Runnable:
    search_chain_run = search_result_chain(llm, search_engine).with_config(
        run_name="SearchInternet"
    )
    context = (
        RunnablePassthrough.assign(output=search_chain_run)
        .assign(context=lambda x: format_search_result(x["output"]))
        .with_config(run_name="GetInformationFromInternet")
    )

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", RAG_TEMPLATE),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{question}"),
        ]
    )

    response_synthesizer = (prompt | llm | StrOutputParser()).with_config(
        run_name="GenerateResponse"
    )
    return (
        RunnablePassthrough.assign(chat_history=serialize_history)
        | context
        | response_synthesizer
    )

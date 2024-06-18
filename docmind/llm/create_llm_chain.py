from operator import itemgetter
from typing import Sequence, Optional, List, Dict

from langchain_core.documents import Document
from langchain_core.language_models import LanguageModelLike
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate, ChatPromptTemplate, MessagesPlaceholder
from langchain_core.pydantic_v1 import BaseModel
from langchain_core.retrievers import BaseRetriever
from langchain_core.runnables import Runnable, RunnableBranch, RunnableLambda, RunnablePassthrough

REPHRASE_TEMPLATE = """\
Given the following conversation and a follow up question, rephrase the follow up \
question to be a standalone question.

You should response only with standalone question without any explanation. \
The question should be clear and informative. \

Chat History:
{chat_history}
Follow Up Input: {question}

REMEMBER: You should response only with standalone question without any \
explanation.

Standalone Question:"""

RAG_TEMPLATE = """\
You are a helpful assistant tasked with answering questions based on information from a knowledge base. Your goal is to provide a comprehensive and informative answer that directly incorporates and cites the relevant context.

Given a series of documents formatted in a specific way, adhere to these guidelines:

1. **Comprehensive Answers:** Strive for informative and thorough responses that fully address the user's question.
2. **Direct Citation:** When using information from a document, seamlessly integrate it into your answer and enclose the quoted or paraphrased content in quotation marks.
3. **Unique Numbered Attribution:** Assign a unique numerical citation in square brackets [1] to each distinct source used in your answer.
4. **Source List at the End:** At the end of your answer, provide a numbered list of the unique sources used, where each number corresponds to the citation used in the answer.  Include the document's source (as indicated after `Source:` in the document's formatting) along with the number.
5. **No External Knowledge:** Base your answer exclusively on the information provided in the documents. Do not introduce any external knowledge or assumptions.
6. **Unbiased and Journalistic Tone:** Maintain an objective and neutral tone, refraining from personal opinions or interpretations.
7. **Disambiguation:** If the documents mention multiple entities with the same name, address each one separately in your response.
8. **Concise and Non-Repetitive:** Avoid unnecessary repetition and aim for a concise answer.
9. **No Hallucination:** If the documents lack relevant information to answer the question, simply state, "Hmm, I'm not sure." Do not invent or guess information.


**Document Format:**

Each document is formatted as follows:
\"\"\"Source: source Content: combined_content\"\"\"

**Important:** Use the content within these formatted documents to answer the question. Ensure that the answer to each question is a combination of the relevant information from the context and the corresponding unique numerical citation in square brackets at the end of each sentence where a new source is introduced. At the end of the response, include a numbered list of unique sources as described above.

#Example
"Source: document1.txt Content: This is the content of the first document. It contains some information about a company. The company's CEO is John Smith. The company was founded in 2005."
"Source: document2.txt Content: This is the content of the second document. It contains some additional information about the same company."

"Response:
The company's CEO is John Smith [1]. The company was founded in 2005 [1].

Sources:
`1. document1.txt`"

{context}
"""


class ChatRequest(BaseModel):
    question: str
    chat_history: Optional[List[Dict[str, str]]]


def serialize_history(request: ChatRequest) -> List:
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


def format_docs(docs: Sequence[Document]) -> str:
    """Formats documents into a source-content structure for the prompt."""

    source_to_content = {}
    for doc in docs:
        source = doc.metadata["source"]
        source_to_content.setdefault(source, []).append(doc.page_content)

    formatted_docs = []
    for source, content_list in source_to_content.items():
        formatted_docs.append(
            f'"""Source: {source} Content: {" ".join(content_list)}"""'
        )

    return "\n".join(formatted_docs)


def create_retriever_chain(llm: LanguageModelLike, retriever: BaseRetriever) -> Runnable:
    condense_question_prompt = PromptTemplate.from_template(REPHRASE_TEMPLATE)
    condense_question_chain = (
            condense_question_prompt | llm | StrOutputParser()
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


def create_llm_with_retriever_chain(llm: LanguageModelLike, retriever: BaseRetriever) -> Runnable:
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
            default_response_synthesizer
            | StrOutputParser()
    ).with_config(run_name="GenerateResponse")
    return (
            RunnablePassthrough.assign(chat_history=serialize_history)
            | context
            | response_synthesizer
    )

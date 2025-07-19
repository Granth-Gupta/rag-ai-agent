import chromadb
import streamlit as st
from sentence_transformers import SentenceTransformer
import PyPDF2
from typing import Annotated, List
from typing_extensions import TypedDict, NotRequired
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.messages import HumanMessage, AIMessage
from langchain.chat_models import init_chat_model
from langchain_community.tools import DuckDuckGoSearchResults
import os
import re
import time
import uuid

# Set environment variable for Google API
os.environ["GOOGLE_API_KEY"] = "AIzaSyCGIUvlDC9uHzuLTxD421dJR_vJ3_m3e-I"


# Initialize global components
@st.cache_resource
def load_embedding_model():
    """Load the sentence transformer model"""
    return SentenceTransformer('all-MiniLM-L6-v2')


@st.cache_resource
def setup_vector_database():
    """Setup ChromaDB vector database"""
    client = chromadb.Client()
    try:
        collection = client.get_collection("ai_documents")
    except:
        collection = client.create_collection("ai_documents")
    return collection


# Load models and database
embedding_model = load_embedding_model()
collection = setup_vector_database()

class AgentState(TypedDict):
    messages: Annotated[list, add_messages]
    query: NotRequired[str]
    documents_found: NotRequired[bool]
    relevant_docs: NotRequired[str]
    document_response: NotRequired[str]
    needs_general_llm: NotRequired[bool]
    route: NotRequired[str]



# Document Processing Functions
def extract_text_from_pdf(pdf_file):
    """Extract text from PDF file"""
    try:
        pdf_reader = PyPDF2.PdfReader(pdf_file)
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text()
        return text
    except Exception as e:
        st.error(f"Error extracting text from PDF: {e}")
        return ""


def chunk_text(text, chunk_size=512, overlap=50):
    """Split text into chunks with overlap"""
    words = text.split()
    chunks = []

    for i in range(0, len(words), chunk_size - overlap):
        chunk = " ".join(words[i:i + chunk_size])
        chunks.append(chunk)
        if i + chunk_size >= len(words):
            break

    return chunks


def clean_text(text):
    """Clean and preprocess text"""
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text)
    # Remove special characters but keep basic punctuation
    text = re.sub(r'[^\w\s.,!?-]', '', text)
    return text.strip()


def store_documents_in_vector_db(documents, filenames):
    """Store documents in ChromaDB vector database"""
    try:
        for doc, filename in zip(documents, filenames):
            # Clean the document text
            cleaned_doc = clean_text(doc)

            if len(cleaned_doc) > 50:  # Only store meaningful chunks
                # Generate embedding
                embedding = embedding_model.encode(cleaned_doc)

                # Create unique ID
                doc_id = f"{filename}_{hash(cleaned_doc)}"

                # Store in ChromaDB
                collection.add(
                    embeddings=[embedding.tolist()],
                    documents=[cleaned_doc],
                    ids=[doc_id],
                    metadatas=[{"filename": filename, "length": len(cleaned_doc)}]
                )

        return True
    except Exception as e:
        st.error(f"Error storing documents: {e}")
        return False


# Agent Node - Analyze intent and route
def agent_node(state: AgentState) -> dict:
    """Analyze user query intent and determine processing route"""
    try:
        query = state["messages"][-1].content.lower()

        # Analyze intent categories
        if any(keyword in query for keyword in ["latest", "recent", "current", "news", "update", "2024", "2025"]):
            intent = "current_info"
        elif any(keyword in query for keyword in ["definition", "what is", "explain", "concept"]):
            intent = "knowledge_based"
        elif any(keyword in query for keyword in ["how to", "implement", "code", "example", "tutorial"]):
            intent = "practical"
        elif any(keyword in query for keyword in ["compare", "difference", "vs", "versus"]):
            intent = "comparison"
        else:
            intent = "general"

        return {"intent": intent}

    except Exception as e:
        return {"intent": "general"}

def document_search_node(state: AgentState) -> dict:
    """Always attempt document retrieval for every query"""
    try:
        query = state["messages"][-1].content

        # Generate query embedding
        query_embedding = embedding_model.encode(query)

        # Search for relevant documents
        results = collection.query(
            query_embeddings=[query_embedding.tolist()],
            n_results=5
        )

        documents_found = False
        relevant_docs = ""

        if results and results['documents'] and results['documents'][0]:
            docs = results['documents'][0]
            # Filter meaningful documents
            filtered_docs = [doc for doc in docs if len(doc.strip()) > 100]

            if filtered_docs:
                documents_found = True
                relevant_docs = "\n\n---\n\n".join([
                    f"Document {i + 1}:\n{doc}"
                    for i, doc in enumerate(filtered_docs)
                ])

        return {
            "query": query,
            "documents_found": documents_found,
            "relevant_docs": relevant_docs,
            "route": "document_llm" if documents_found else "general_llm"
        }

    except Exception as e:
        # On failure, route to general LLM
        return {
            "query": state["messages"][-1].content,
            "documents_found": False,
            "relevant_docs": "",
            "route": "general_llm"
        }

# Gen Know Node - Generate Knowledge from Documents
def gen_know_node(state: AgentState) -> dict:
    """Generate structured knowledge from retrieved documents"""
    try:
        llm = init_chat_model("google_genai:gemini-2.0-flash")

        query = state["messages"][-1].content
        relevant_docs = state.get("relevant_docs", "")
        intent = state.get("intent", "general")

        # Intent-specific knowledge generation prompts
        if intent == "practical":
            knowledge_prompt = f"""
            Based on the following documents, extract and organize practical information for this query: {query}

            Documents:
            {relevant_docs}

            Generate structured knowledge focusing on:
            1. Step-by-step processes or implementations
            2. Code examples and technical details  
            3. Best practices and common pitfalls
            4. Prerequisites and requirements

            Provide clear, actionable knowledge:
            """
        elif intent == "comparison":
            knowledge_prompt = f"""
            Based on the following documents, extract comparative information for this query: {query}

            Documents:
            {relevant_docs}

            Generate structured knowledge focusing on:
            1. Key differences and similarities
            2. Pros and cons of each option
            3. Use cases and scenarios
            4. Performance or capability comparisons

            Provide balanced comparative knowledge:
            """
        elif intent == "knowledge_based":
            knowledge_prompt = f"""
            Based on the following documents, extract conceptual information for this query: {query}

            Documents:
            {relevant_docs}

            Generate structured knowledge focusing on:
            1. Core concepts and definitions
            2. Theoretical foundations
            3. Relationships between concepts
            4. Applications and examples

            Provide comprehensive conceptual knowledge:
            """
        else:
            knowledge_prompt = f"""
            Based on the following documents, extract relevant information for this query: {query}

            Documents:
            {relevant_docs}

            Generate well-structured knowledge that directly addresses the user's question.
            Focus on accuracy and relevance:
            """

        response = llm.invoke([HumanMessage(content=knowledge_prompt)])
        generated_knowledge = response.content

        return {"generated_knowledge": generated_knowledge}

    except Exception as e:
        st.error(f"Knowledge generation failed: {e}")
        return {"generated_knowledge": "Failed to generate knowledge from documents."}

def document_llm_node(state: AgentState) -> dict:
    """Process queries using document knowledge"""
    try:
        llm = init_chat_model("google_genai:gemini-2.0-flash")
        query = state["query"]
        relevant_docs = state.get("relevant_docs", "")

        document_prompt = f"""
        Based on the following documents, answer this query: {query}

        Documents:
        {relevant_docs}

        Instructions:
        1. If the documents contain relevant information, provide a comprehensive answer
        2. If the documents don't contain relevant information, respond with "INSUFFICIENT_CONTEXT"
        3. Use only information from the provided documents
        4. Be specific and cite relevant parts of the documents

        Answer:
        """

        response = llm.invoke([HumanMessage(content=document_prompt)])
        doc_response = response.content

        # Check if document response is insufficient
        needs_general = "INSUFFICIENT_CONTEXT" in doc_response or len(doc_response.strip()) < 50

        if needs_general:
            return {
                "document_response": doc_response,
                "needs_general_llm": True,
                "route": "general_llm"
            }
        else:
            return {
                "messages": [AIMessage(content=doc_response)],
                "needs_general_llm": False,
                "route": "end"
            }

    except Exception as e:
        # On failure, route to general LLM
        return {
            "document_response": f"Document processing failed: {str(e)}",
            "needs_general_llm": True,
            "route": "general_llm"
        }

def general_llm_node(state: AgentState) -> dict:
    """Handle queries using general knowledge"""
    try:
        llm = init_chat_model("google_genai:gemini-2.0-flash")
        query = state["query"]

        general_prompt = f"""
        You are Perplexity, an AI assistant specializing in providing accurate, comprehensive answers.

        User Query: {query}

        Instructions:
        1. Provide a comprehensive answer using your general knowledge
        2. Focus on accuracy and helpfulness
        3. Structure your response clearly
        4. If this query was routed here because documents were insufficient, provide the best general knowledge answer possible

        Generate your response:
        """

        response = llm.invoke([HumanMessage(content=general_prompt)])

        return {
            "messages": [AIMessage(content=response.content)],
            "route": "end"
        }

    except Exception as e:
        error_message = f"I apologize, but I encountered an error: {str(e)}"
        return {"messages": [AIMessage(content=error_message)]}

def route_after_document_search(state: AgentState) -> str:
    """Route based on document availability"""
    return state.get("route", "general_llm")

def route_after_document_llm(state: AgentState) -> str:
    """Route based on document LLM response quality"""
    if state.get("needs_general_llm", False):
        return "general_llm"
    else:
        return "end"

# Routing function
def route_based_on_documents(state: AgentState) -> str:
    """Route to appropriate LLM based on document availability"""
    return state.get("route", "general_path")

def build_rag_agent():
    """Build the dual-path RAG agent"""
    memory = MemorySaver()
    workflow = StateGraph(AgentState)

    # Add nodes
    workflow.add_node("document_search", document_search_node)
    workflow.add_node("document_llm", document_llm_node)
    workflow.add_node("general_llm", general_llm_node)

    # Define workflow edges
    workflow.add_edge(START, "document_search")

    # Conditional routing after document search
    workflow.add_conditional_edges(
        "document_search",
        route_after_document_search,
        {
            "document_llm": "document_llm",
            "general_llm": "general_llm"
        }
    )

    # Conditional routing after document LLM
    workflow.add_conditional_edges(
        "document_llm",
        route_after_document_llm,
        {
            "general_llm": "general_llm",
            "end": END
        }
    )

    # General LLM always ends
    workflow.add_edge("general_llm", END)

    return workflow.compile(checkpointer=memory)

import os
import glob


def load_default_documents():
    """Load default documents from the documents folder on startup"""
    documents_folder = "documents"

    if not os.path.exists(documents_folder):
        st.warning("Documents folder not found. Creating empty folder.")
        os.makedirs(documents_folder)
        return False

    pdf_files = glob.glob(os.path.join(documents_folder, "*.pdf"))

    if not pdf_files:
        st.info("No PDF files found in documents folder.")
        return False

    with st.spinner("Loading default documents..."):
        all_chunks = []
        all_filenames = []

        for pdf_path in pdf_files:
            filename = os.path.basename(pdf_path)

            try:
                with open(pdf_path, 'rb') as file:
                    text = extract_text_from_pdf(file)
                    if text:
                        cleaned_text = clean_text(text)
                        chunks = chunk_text(cleaned_text)
                        all_chunks.extend(chunks)
                        all_filenames.extend([filename] * len(chunks))
                        st.success(f"✅ Loaded {filename}")
            except Exception as e:
                st.error(f"❌ Error loading {filename}: {e}")

        if all_chunks:
            success = store_documents_in_vector_db(all_chunks, all_filenames)
            if success:
                st.success(f"🎉 Auto-loaded {len(pdf_files)} documents with {len(all_chunks)} chunks!")
                return True

    return False

def main():
    st.set_page_config(
        page_title="RAG AI Agent - New Architecture",
        page_icon="🤖",
        layout="wide"
    )

    st.title("🤖 RAG AI Agent")
    st.markdown("""
    **Flow**: Agent → INT Doc → Gen Know → Document_LLM → General_LLM → Output

    *Documents are retrieved intelligently based on query intent, then processed through specialized knowledge generation.*
    """)

    # Initialize session state
    if "agent" not in st.session_state:
        st.session_state.agent = build_rag_agent()
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "thread_id" not in st.session_state:
        st.session_state.thread_id = str(uuid.uuid4())

    # Auto-load documents on first run
    if "documents_loaded" not in st.session_state:
        st.session_state.documents_loaded = load_default_documents()

    # Sidebar for document management
    with st.sidebar:
        st.header("📁 Document Management")

        # File upload
        uploaded_files = st.file_uploader(
            "Upload PDF documents",
            type=['pdf'],
            accept_multiple_files=True,
            key="pdf_uploader"
        )

        if uploaded_files:
            if st.button("📚 Process Documents", type="primary"):
                with st.spinner("Processing documents..."):
                    all_chunks = []
                    all_filenames = []

                    for uploaded_file in uploaded_files:
                        # Extract text from PDF
                        text = extract_text_from_pdf(uploaded_file)
                        if text:
                            # Clean and chunk the text
                            cleaned_text = clean_text(text)
                            chunks = chunk_text(cleaned_text)

                            # Add chunks and filenames
                            all_chunks.extend(chunks)
                            all_filenames.extend([uploaded_file.name] * len(chunks))

                    if all_chunks:
                        # Store in vector database
                        success = store_documents_in_vector_db(all_chunks, all_filenames)
                        if success:
                            st.success(f"✅ Processed {len(uploaded_files)} documents with {len(all_chunks)} chunks!")
                        else:
                            st.error("❌ Error processing documents")
                    else:
                        st.error("❌ No text extracted from documents")

        # Database stats
        st.subheader("📊 Database Stats")
        try:
            stats = collection.count()
            st.metric("Documents in Database", stats)
        except:
            st.metric("Documents in Database", 0)

        # Clear database
        if st.button("🗑️ Clear Database", type="secondary"):
            try:
                collection.delete()
                st.success("Database cleared!")
                time.sleep(1)
                st.rerun()
            except Exception as e:
                st.error(f"Error clearing database: {e}")

    # Main chat interface
    st.header("💬 Chat Interface")

    # Display chat messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.write(message["content"])

    # Chat input
    if prompt := st.chat_input("Ask me anything about AI, ML, or your uploaded documents..."):
        # Add user message to chat
        st.session_state.messages.append({"role": "user", "content": prompt})

        # Display user message
        with st.chat_message("user"):
            st.write(prompt)

        # Generate assistant response
        with st.chat_message("assistant"):
            with st.spinner("Processing..."):
                try:
                    # Create state for the agent
                    config = {"configurable": {"thread_id": st.session_state.thread_id}}

                    # Run the agent
                    result = st.session_state.agent.invoke(
                        {"messages": [HumanMessage(content=prompt)]},
                        config=config
                    )

                    # Get the response
                    response = result["messages"][-1].content

                    # Display response
                    st.write(response)

                    # Add to session state
                    st.session_state.messages.append({"role": "assistant", "content": response})

                except Exception as e:
                    error_msg = f"I apologize, but I encountered an error: {str(e)}"
                    st.error(error_msg)
                    st.session_state.messages.append({"role": "assistant", "content": error_msg})

    # Clear chat history
    if st.button("🗑️ Clear Chat", type="secondary"):
        st.session_state.messages = []
        st.session_state.thread_id = str(uuid.uuid4())
        st.rerun()

    # Architecture visualization
    with st.expander("🔍 View Architecture Flow"):
        st.markdown("""
        ### New Architecture Flow:

        1. **Agent Node**: Analyzes query intent (practical, comparison, knowledge-based, etc.)
        2. **INT Doc Node**: Intelligently retrieves documents based on intent with optimized parameters
        3. **Conditional Routing**: 
           - If documents found → Go to Gen Know Node
           - If no documents → Go directly to General LLM Node
        4. **Gen Know Node**: Generates structured knowledge from retrieved documents
        5. **Document LLM Node**: Creates final response using generated knowledge
        6. **General LLM Node**: Handles queries when no relevant documents exist
        7. **Output**: Final response to user

        ### Key Features:
        - **Intent-based processing**: Different strategies for different query types
        - **Intelligent document retrieval**: Adaptive search parameters
        - **Knowledge synthesis**: Structured information extraction from documents
        - **Fallback mechanism**: General knowledge when documents aren't available
        - **Clean routing**: Simple decision tree with clear paths
        """)


if __name__ == "__main__":
    main()

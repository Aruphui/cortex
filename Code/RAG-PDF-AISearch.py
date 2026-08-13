
# ============================================
# COMPLETE RAG PIPELINE
# Azure OpenAI + Azure AI Search
# ============================================

import os
from openai import AzureOpenAI
from langchain_text_splitters import RecursiveCharacterTextSplitter
from dotenv import load_dotenv
from azure.search.documents import SearchClient
from azure.search.documents.models import VectorizedQuery
from azure.core.credentials import AzureKeyCredential
from azure.search.documents.indexes import SearchIndexClient
from azure.search.documents.models import QueryType, SemanticErrorMode
from azure.search.documents.indexes.models import (
    SearchIndex,
    SearchField,
    SearchFieldDataType,
    SimpleField,
    SearchableField,
    VectorSearch,
    HnswAlgorithmConfiguration,
    VectorSearchProfile,
    SemanticConfiguration,
    SemanticSearch,
    SemanticPrioritizedFields,
    SemanticField,
)

# Load environment variables
load_dotenv()
# ============================================
# CLIENTS & CONFIG
# ============================================

EMBEDDING_DIMENSIONS = 1536  # Standard for Azure OpenAI embeddings
AZURE_SEARCH_ENDPOINT = os.environ["AZURE_SEARCH_ENDPOINT"]
AZURE_SEARCH_API_KEY = os.environ["AZURE_SEARCH_API_KEY"]
AZURE_SEARCH_INDEX_NAME = os.environ["AZURE_SEARCH_INDEX_NAME"]

client = AzureOpenAI(
    azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
    api_key=os.environ["AZURE_OPENAI_API_KEY"],
    api_version=os.environ["AZURE_OPENAI_API_VERSION"],
)

EMBED_DEPLOY = os.environ["AZURE_OPENAI_EMBEDDING_DEPLOYMENT"]
CHAT_DEPLOY  = os.environ["AZURE_OPENAI_CHAT_DEPLOYMENT"]

search_client = SearchClient(
    endpoint=AZURE_SEARCH_ENDPOINT,
    index_name=AZURE_SEARCH_INDEX_NAME,
    credential=AzureKeyCredential(AZURE_SEARCH_API_KEY)
)

from pypdf import PdfReader
import re

def extract_pdf_text(pdf_paths: str | list[str]) -> list[dict]:
    """
    Extract and clean text from one or multiple PDFs.
    Accepts a single path string or a list of path strings.
    Returns: list of {"text", "page_number", "source"}
    """
    # Normalize to list so single path and multiple paths are handled the same
    if isinstance(pdf_paths, str):
        pdf_paths = [pdf_paths]

    all_pages = []
    for pdf_path in pdf_paths:
        reader = PdfReader(pdf_path)
        for page_num, page in enumerate(reader.pages, start=1):
            raw   = page.extract_text() or ""
            clean = clean_text(raw)
            if clean.strip():
                all_pages.append({
                    "text":        clean,
                    "page_number": page_num,
                    "source":      pdf_path   # ← tracks which PDF each page came from
                })
        print(f"Extracted pages from {pdf_path}")

    print(f"✅ Total pages extracted: {len(all_pages)}")
    return all_pages
def clean_text(text: str) -> str:
    """Remove noise commonly introduced by PDF extraction."""
    text = re.sub(r'\s+', ' ', text)      # collapse multiple spaces/newlines
    text = re.sub(r'[•●▪▸]', '-', text)   # normalise bullet symbols
    return text.strip()

def create_chunk(pages: list[dict]) -> list[str]:
    """Split each page's text into overlapping chunks."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50,
        separators=["\n\n", "\n", ". ", " ", ""]
    )
    all_chunks = []
    for page in pages:                          # ← loop over pages
        page_chunks = splitter.split_text(page["text"])
        all_chunks.extend(page_chunks)
    print(f"✅ Split into {len(all_chunks)} chunks")
    return all_chunks                           # ← return the chunks


def create_embedding(chunks: list[str]) -> list[list[float]]:
    """Generate embeddings for every chunk."""
    resp = client.embeddings.create(model=EMBED_DEPLOY, input=chunks)
    vecs = [r.embedding for r in sorted(resp.data, key=lambda x: x.index)]
    print(f"✅ Generated {len(vecs)} embeddings")
    return vecs  

def create_index():
    index_client = SearchIndexClient(
        endpoint=AZURE_SEARCH_ENDPOINT,
        credential=AzureKeyCredential(AZURE_SEARCH_API_KEY),
    )

    fields = [
        SimpleField(
            name="id",
            type=SearchFieldDataType.String,
            key=True,
            filterable=True,
        ),
        SearchableField(
            name="title",
            type=SearchFieldDataType.String,
            filterable=True,
            sortable=True,
        ),
        SearchableField(
            name="content",
            type=SearchFieldDataType.String,
        ),
        SearchField(
            name="content_vector",
            type=SearchFieldDataType.Collection(SearchFieldDataType.Single),
            searchable=True,
            vector_search_dimensions=EMBEDDING_DIMENSIONS,
            vector_search_profile_name="acme-vector-profile",
        ),
    ]

    vector_search = VectorSearch(
        algorithms=[HnswAlgorithmConfiguration(name="acme-hnsw")],
        profiles=[
            VectorSearchProfile(
                name="acme-vector-profile",
                algorithm_configuration_name="acme-hnsw",
            )
        ],
    )

    semantic_config = SemanticConfiguration(
        name="acme-semantic-config",
        prioritized_fields=SemanticPrioritizedFields(
            title_field=SemanticField(field_name="title"),
            content_fields=[SemanticField(field_name="content")],
        ),
    )

    index = SearchIndex(
        name=AZURE_SEARCH_INDEX_NAME,
        fields=fields,
        vector_search=vector_search,
        semantic_search=SemanticSearch(configurations=[semantic_config]),
    )

    result = index_client.create_or_update_index(index)
    print(f"✅ Index '{result.name}' created/updated successfully.")


# ── Step 4: Index documents ───────────────────────────────────────────────────
def index_documents(chunks, vectors):
    """Upload chunks and their embeddings to Azure AI Search"""
    documents = []
    for i, (chunk, vector) in enumerate(zip(chunks, vectors)):
        documents.append({
            "id": str(i),
            "title": f"Acme Corp Policy {i+1}",
            "content": chunk,
            "content_vector": vector,
        })
    
    try:
        result = search_client.upload_documents(documents)
        print(f"✅ Uploaded {len(documents)} documents to index")
    except Exception as e:
        print(f"❌ Error uploading documents: {e}")
        raise


# ── Step 5: RAG Query Function ────────────────────────────────────────────────
def rag_query(user_query: str, top_k: int = 3) -> str:
    """Execute RAG pipeline: embed query → search → generate response"""
    
    # Embed the user query
    query_resp = client.embeddings.create(model=EMBED_DEPLOY, input=user_query)
    query_vector = query_resp.data[0].embedding
    
    # Search Azure AI Search for relevant documents
    search_results = search_client.search(
        search_text=user_query,
        vector_queries=[VectorizedQuery(vector=query_vector, k_nearest_neighbors=top_k, fields="content_vector")],
        select=["id", "content"],

    # ── these 3 lines activate semantic ranking ──
        query_type=QueryType.SEMANTIC,                    # switches on semantic reranker
        semantic_configuration_name="acme-semantic-config",  # must match name in create_index()
        semantic_error_mode=SemanticErrorMode.PARTIAL,    # fallback if semantic fails

        top=top_k, 
    )
    
    # Extract relevant context from search results
    context = ""
    for result in search_results:
        context += f"\n{result['content']}"
    
    if not context.strip():
        context = "No relevant information found in knowledge base."
    
    # Generate response using Azure OpenAI
    system_prompt = """You are an Azure training assistant for freshers/interns.

STRICT RULES — follow these without exception:
1. Answer ONLY using the context provided below. 
2. Do NOT use your own training knowledge, even if you know the answer.
3. If the context does not contain enough information to answer, respond exactly with:
   "I don't have information about this in the provided documents."
4. Do not make assumptions or infer beyond what is explicitly stated in the context.
5. Always be concise and factual."""

    
    response = client.chat.completions.create(
        model=CHAT_DEPLOY,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {user_query}"}
        ],
        temperature=0.7,
        max_completion_tokens=500,
    )
    
    return response.choices[0].message.content


# ── Main ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    try:
        print("\n🚀 Starting RAG Pipeline Setup...\n")
      
        pages  = extract_pdf_text(["Az_900_1689824836.pdf", "Az700 Short Note.pdf", "AZ-104 Guide.pdf", "Azure Storage Account.pdf", "Azure Backup.pdf", "Azure Compute.pdf"])
        chunks = create_chunk(pages)            # ← capture return value
        vecs   = create_embedding(chunks)       # ← capture return value

        print("Step 1: Creating Azure AI Search index...")
        create_index()

        print("\nStep 2: Indexing documents...")
        index_documents(chunks, vecs)

        print("\nStep 3: Testing RAG with sample query...")
        query = "What is PV, PVC and daemonsets in AKS?"
        print(f"\nQuery: {query}")
        response = rag_query(query)
        print(f"\nResponse:\n{response}")

        print("\n✅ RAG Pipeline Setup Complete!")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        raise

if __name__ == "__main__":
    try:
        query = "Storage account tier in azure?"
        print(f"\nQuery: {query}")
        response = rag_query(query)
        print(f"\nResponse:\n{response}")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        raise
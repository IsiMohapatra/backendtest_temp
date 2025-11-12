"""
RAG tools for vehicle assistant - Complete integration from version3_refactor.py
This module contains the @tool-decorated helper functions used by the diagnostic workflow.
"""

import os
import re
import json
import logging
import requests
from dotenv import load_dotenv
from typing import Optional, List, Dict, Any
from pathlib import Path
import sys
import numpy as np

from langchain_core.tools import tool
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

# Load env
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# -------------------------
# Paths and Setup
# -------------------------
current_dir = Path(__file__).resolve().parent
parent_dir = current_dir.parent
sys.path.insert(0, str(parent_dir))

# Import the consistent paths from config
from configs.RAG_config import FAISS_DB_DIR, FAISS_INDEX_PATH, FAISS_METADATA_PATH

vector_store_type = "faiss"

# -------------------------
# Initialize LLM + embeddings + retriever
# -------------------------
if not OPENAI_API_KEY:
    raise RuntimeError("OPENAI_API_KEY missing - set it in .env")

llm = ChatOpenAI(model="gpt-5-nano",temperature=0,openai_api_key=OPENAI_API_KEY)
embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)

# FAISS imports
try:
    import faiss
    import pickle
    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False
    print("⚠️ FAISS not installed. Install with: pip install faiss-cpu (or faiss-gpu)")

# Initialize retriever variable
retriever = None

class FAISSRetriever:
    """FAISS-based retriever that mimics LangChain retriever interface"""
    
    def __init__(self):
        self.index = None
        self.metadata_store = []
        self.id_to_index = {}
        self.embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
        self._init_faiss()
    
    def _init_faiss(self):
        """Initialize FAISS with enhanced metadata format"""
        print(f"🔧 Using FAISS vector store")
        
        if not FAISS_AVAILABLE:
            logger.error("❌ FAISS not available. Install with: pip install faiss-cpu")
            raise SystemExit(1)
        
        # Check if FAISS files exist
        if not Path(FAISS_INDEX_PATH).exists() or not Path(FAISS_METADATA_PATH).exists():
            logger.error(f"❌ FAISS index not found at {FAISS_INDEX_PATH} or {FAISS_METADATA_PATH}. Run rag_setup.py first.")
            raise SystemExit(1)
        
        try:
            # Load FAISS index
            self.index = faiss.read_index(FAISS_INDEX_PATH)
            
            # Load enhanced metadata format
            with open(FAISS_METADATA_PATH, 'rb') as f:
                data = pickle.load(f)
                self.metadata_store = data['metadata_store']
                self.id_to_index = data['id_to_index']
            
            print(f"✅ FAISS vector store loaded successfully")
            print(f"📊 Index contains {self.index.ntotal} vectors")
            print(f"📝 Metadata store contains {len(self.metadata_store)} documents")
            
        except Exception as e:
            logger.error(f"❌ Error loading FAISS vector store: {str(e)}")
            import traceback
            traceback.print_exc()
            raise SystemExit(1)
    
    def invoke(self, query: str, k: int = 5):
        """Query FAISS and return LangChain-compatible Document objects"""
        if self.index is None:
            return []
        
        try:
            # Generate query embedding
            query_embedding = self.embeddings.embed_query(query)
            query_vector = np.array([query_embedding], dtype=np.float32)
            faiss.normalize_L2(query_vector)
            
            # Search
            scores, indices = self.index.search(query_vector, k)
            
            # Convert to LangChain Document format
            from langchain.docstore.document import Document
            
            documents = []
            for score, idx in zip(scores[0], indices[0]):
                if idx >= 0 and idx < len(self.metadata_store):
                    metadata = self.metadata_store[idx]
                    
                    # Create LangChain-compatible metadata
                    doc_metadata = {
                        'pages': metadata.get('pages', [1]),
                        'page': metadata.get('pages', [1])[0] if metadata.get('pages') else 1,
                        'heading': metadata.get('heading', ''),
                        'source_file': metadata.get('source_file', ''),
                        'chunk_index': metadata.get('chunk_index', 0),
                        'has_tables': metadata.get('has_tables', False),
                        'has_figures': metadata.get('has_figures', False),
                        'content_type': metadata.get('content_type', 'text'),
                        'similarity_score': float(score),
                    }
                    
                    # Create Document object
                    doc = Document(
                        page_content=metadata['text'],
                        metadata=doc_metadata
                    )
                    documents.append(doc)
            
            return documents
            
        except Exception as e:
            logger.error(f"❌ FAISS query error: {str(e)}")
            return []

# Initialize the retriever
try:
    if FAISS_AVAILABLE:
        retriever = FAISSRetriever()
        print("✅ FAISS retriever initialized successfully")
    else:
        logger.warning("FAISS not available, retriever will be None")
        retriever = None
except Exception as e:
    logger.error(f"Failed to initialize FAISS retriever: {e}")
    retriever = None


# Retrieval grader setup
grader_llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.0, openai_api_key=OPENAI_API_KEY)
grader_prompt = PromptTemplate(
    template="""You are a teacher grading a quiz. You will be given: 1/ a QUESTION 2/ A FACT provided by the student
You are grading RELEVANCE RECALL: A score of 1 means that ANY of the statements in the FACT are relevant to the QUESTION. A score of 0 means that NONE of the statements in the FACT are relevant to the QUESTION.
Provide the binary score as a JSON with a single key 'score' and no preamble or explanation.
Question: {question} Fact: {documents}""",
    input_variables=["question", "documents"],
)
retrieval_grader = grader_prompt | grader_llm | JsonOutputParser()

# Global NLP model (lazy-loaded)
_nlp_model = None

def _get_nlp():
    global _nlp_model
    if _nlp_model is None:
        import spacy
        try:
            _nlp_model = spacy.load("en_core_web_trf")
        except Exception:
            try:
                _nlp_model = spacy.load("en_core_web_sm")
            except Exception as e:
                logger.warning(f"Could not load spacy model: {e}")
                _nlp_model = None
    return _nlp_model


@tool
def is_vehicle_related(question: str) -> dict:
    """Check if the question is vehicle-related before processing."""
    classifier_prompt = f"""
You are a classifier. Decide if the user question is about vehicle diagnostics, repair, or automotive problems.
Answer YES if it's about vehicle issues, maintenance, repairs, faults, or checks.
Answer NO if it's unrelated to vehicles.
Examples:
- "How do I replace my brake pads?" → YES
- "What is the capital of France?" → NO
- "Can I change the engine oil myself?" → YES
- "Tell me a joke." → NO
QUESTION: {question}
Answer only with "YES" or "NO".
"""
    try:
        response = llm.invoke(classifier_prompt)
        is_related = response.content.strip().upper() == "YES"
        logger.info(f"Vehicle relation check: {question[:50]}... → {is_related}")
        return {
            "is_vehicle_related": is_related,
            "message": "Vehicle-related question detected" if is_related else "Not vehicle-related",
        }
    except Exception as e:
        logger.error(f"Error in vehicle relation check: {e}")
        return {"is_vehicle_related": True, "message": "Error in classification, defaulting to vehicle-related"}

@tool
def extract_vehicle_model(question: str) -> dict:
    """Extract vehicle make and model from the question using NLP."""
    nlp_model = _get_nlp()
    if not nlp_model:
        logger.warning("NLP model not available, falling back to simple extraction")
        # Simple fallback extraction
        words = question.split()
        potential_vehicle = []
        for i, word in enumerate(words):
            if word.lower() in ['toyota', 'honda', 'ford', 'bmw', 'mercedes', 'audi', 'volkswagen', 'nissan', 'hyundai', 'kia']:
                potential_vehicle.append(word.title())
                if i + 1 < len(words):
                    potential_vehicle.append(words[i + 1].title())
                break
        
        if potential_vehicle:
            vehicle_info = " ".join(potential_vehicle[:2])
            return {"vehicle_info": vehicle_info, "found": True}
        else:
            return {"vehicle_info": None, "found": False}
    
    doc = nlp_model(question)
    entities = []
    
    for ent in doc.ents:
        if ent.label_ in ["ORG", "PRODUCT"]:
            model_tokens = [ent.text]
            next_token = ent.end
            while next_token < len(doc) and (
                doc[next_token].is_title or doc[next_token].like_num or doc[next_token].is_lower
            ):
                model_tokens.append(doc[next_token].text)
                next_token += 1
            entities.append(" ".join(model_tokens))

    if entities:
        vehicle_info = entities[0].title()
    else:
        tokens = [t for t in doc if not t.is_stop and t.pos_ in ["PROPN", "NUM", "NOUN"]]
        if len(tokens) >= 2:
            model_tokens = []
            for t in tokens:
                if t.pos_ in ["PROPN", "NUM"]:
                    model_tokens.append(t.text)
                else:
                    break
            vehicle_info = " ".join(model_tokens).title() if model_tokens else None
        else:
            vehicle_info = None

    logger.info(f"Vehicle extraction: {question[:50]}... → {vehicle_info}")
    return {"vehicle_info": vehicle_info, "found": vehicle_info is not None}

def normalize_dtc_codes(text: str) -> str:
    """
    Normalize spoken DTC codes to standard format - ULTRA FLEXIBLE VERSION.
    
    Examples:
    - "P zero three zero one" → "P0301"
    - "tell me about p zero three zero one" → "tell me about P0301"
    """
    
    # Dictionary to convert word numbers to digits
    word_to_digit = {
        'zero': '0', 'one': '1', 'two': '2', 'three': '3', 'four': '4',
        'five': '5', 'six': '6', 'seven': '7', 'eight': '8', 'nine': '9'
    }
    
    # ULTRA FLEXIBLE pattern - allows any non-word chars between parts
    pattern = r'\b([PBCUpbcu])\W*(zero|one|two|three|four|five|six|seven|eight|nine)\W*(zero|one|two|three|four|five|six|seven|eight|nine)\W*(zero|one|two|three|four|five|six|seven|eight|nine)\W*(zero|one|two|three|four|five|six|seven|eight|nine)\b'
    
    def replace_dtc(match):
        prefix = match.group(1).upper()  # P, B, C, or U
        digit1 = word_to_digit[match.group(2).lower()]
        digit2 = word_to_digit[match.group(3).lower()]
        digit3 = word_to_digit[match.group(4).lower()]
        digit4 = word_to_digit[match.group(5).lower()]
        
        result = f"{prefix}{digit1}{digit2}{digit3}{digit4}"
        logger.info(f"🔧 DTC Match found: '{match.group(0)}' → '{result}'")
        return result
    
    # Apply the replacement
    normalized_text = re.sub(pattern, replace_dtc, text, flags=re.IGNORECASE)
    return normalized_text

@tool
def search_vehicle_documents(question: str, dtc_code: str = None, vehicle_info: str = None) -> dict:
    """Search vehicle diagnostic documents for relevant information."""
    question = normalize_dtc_codes(question)
    logger.info(f"🔍 Searching documents for: {question}") 
    

    if not retriever:
        logger.error("Retriever not available")
        return {
            "answer": "Document search not available - database not initialized.",
            "source_documents": [],
            "has_rag_info": False,
            "dtc_code": dtc_code,
            "vehicle_info": vehicle_info,
            "selected_chunk_label": "ERROR",
            "selected_chunk_content": ""
        }
    
    # Check for DTC code in question
    dtc_match = re.search(r"\b([PBUC]\d{4})\b", question.upper())
    if dtc_match:
        dtc_code = dtc_match.group(1)
    
    try:
        # Retrieve relevant documents
        docs = retriever.invoke(question)
        
        if not docs:
            return {
                "answer": "No relevant information found in the PDF.",
                "source_documents": [],
                "has_rag_info": False,
                "dtc_code": dtc_code,
                "vehicle_info": vehicle_info,
                "selected_chunk_label": "NONE",
                "selected_chunk_content": ""
            }
        
        print(f"📊 Retrieved {len(docs)} documents from vector store")

        # Display retrieved chunks with media information (like rag_test_basic)
        print("\n🔎 Retrieved Chunks:")
        for i, d in enumerate(docs, 1):
            pages = d.metadata.get("pages") or d.metadata.get("page") or "?"
            snippet = d.page_content[:300].replace('\n', ' ') + ('...' if len(d.page_content) > 300 else '')
            
            # Check for media using the same function as rag_test_basic
            media_info = extract_media_references_enhanced(d.page_content)
            media_indicators = []
            if media_info['images']:
                media_indicators.append(f"📷{len(media_info['images'])}")
            if media_info['tables']:
                media_indicators.append(f"📊{len(media_info['tables'])}")
            
            media_str = f" [{', '.join(media_indicators)}]" if media_indicators else ""
            
            print(f"  {i}. pages={pages} chars={len(d.page_content)}{media_str}")
            print(f"     snippet={snippet}")

        # Build context using the SAME format as rag_test_basic.py
        blocks = []
        for i, d in enumerate(docs, 1):
            pages = d.metadata.get("pages") or d.metadata.get("page") or "?"
            
            # Format content with media information (like rag_test_basic)
            formatted_content = format_content_with_media_enhanced(d.page_content, i)
            
            blocks.append(f"[DOC {i} | pages: {pages}]\n{formatted_content}")
        
        context = "\n\n".join(blocks)
        
        # Use the SAME prompt structure as rag_test_basic.py
        prompt = (
            "You are a helpful automotive assistant. Answer the user's question using the provided context.\n\n"
            "CRITICAL PRESERVATION RULES:\n"
            "1. When the context contains image markdown (![...](...)): Copy the EXACT line without any changes\n"
            "2. When the context contains tables (lines with | characters): Copy the ENTIRE table block exactly as shown\n"
            "3. When the context contains numbered steps: Copy the exact numbering and text\n"
            "4. DO NOT summarize, rephrase, or reformat any images, tables, or step procedures\n"
            "5. DO NOT change file paths in image links (even if they look like C:\\\\... paths)\n\n"
            "OUTPUT FORMAT:\n"
            "- Provide a direct answer to the question (2-3 sentences)\n"
            "- If relevant images/tables/steps exist in context, include a section:\n"
            "  '### References from document'\n"
            "- Under References, paste the relevant markdown blocks EXACTLY as they appear\n"
            "- Include ALL relevant images/tables/steps that help answer the question\n\n"
            "EXAMPLE OF WHAT TO PRESERVE:\n"
            "- Images: ![Image](C:\\\\path\\\\image.jpg) ← Copy this EXACTLY\n"
            "- Tables: | Header | Value | ← Copy entire table including all rows\n"
            "- Steps: 1. Check the sensor... ← Copy exact numbering and text\n\n"
            "Your task: Answer the question and preserve all relevant visual/structured content VERBATIM.\n\n"
            f"User Question: {question}\n\n"
            f"Context from Documents:\n{context}\n\n"
            "Provide your answer following the format above:"
        )
        
        response = llm.invoke(prompt)
        answer_text = response.content.strip()
        print(f"🤖 Direct LLM Answer: {answer_text}")
        
        if answer_text == "I don't know." or "I don't know" in answer_text:
            print(f"❌ LLM couldn't find relevant information")
            return {
                "answer": "No relevant information found in the PDF.",
                "source_documents": [],
                "has_rag_info": False,
                "dtc_code": dtc_code,
                "vehicle_info": vehicle_info,
                "selected_chunk_label": "NONE",
                "selected_chunk_content": ""
            }
        else:
            # Return the enhanced content with media information
            has_rag_info = True
            source = [{
                "page_number": docs[0].metadata.get("pages", "N/A"),
                "content": answer_text  # Return the LLM's processed answer
            }]
            
            result = {
                "answer": answer_text,
                "source_documents": source,
                "has_rag_info": has_rag_info,
                "dtc_code": dtc_code,
                "vehicle_info": vehicle_info,
                "selected_chunk_label": "PROCESSED",
                "selected_chunk_content": answer_text
            }
            return result

    except Exception as e:
        logger.error(f"Error in document search: {e}")
        return {
            "answer": "Error occurred during document search.",
            "source_documents": [],
            "has_rag_info": False,
            "dtc_code": dtc_code,
            "vehicle_info": vehicle_info,
            "selected_chunk_label": "ERROR",
            "selected_chunk_content": ""
        }

def extract_media_references_enhanced(content: str):
    """Extract image links and table references from content (same as rag_test_basic)"""
    media_info = {
        'images': [],
        'tables': [],
        'has_media': False
    }
    
    # Look for image references (common patterns)
    image_patterns = [
        r'!\[.*?\]\((.*?)\)',  # Markdown images
        r'<img.*?src=["\']([^"\']+)["\']',  # HTML images
        r'Image:\s*([^\s\n]+)',  # Custom image format
        r'Figure\s+\d+[:\-]?\s*([^\n]+)',  # Figure references
        r'\[Image:\s*([^\]]+)\]',  # Bracketed image refs
    ]
    
    for pattern in image_patterns:
        matches = re.findall(pattern, content, re.IGNORECASE)
        media_info['images'].extend(matches)
    
    # Look for table references
    table_patterns = [
        r'Table\s+\d+[:\-]?\s*([^\n]+)',  # Table references
        r'\|.*\|.*\|',  # Markdown table rows
        r'<table.*?</table>',  # HTML tables
    ]
    
    for pattern in table_patterns:
        matches = re.findall(pattern, content, re.IGNORECASE | re.DOTALL)
        media_info['tables'].extend(matches)
    
    media_info['has_media'] = bool(media_info['images'] or media_info['tables'])
    return media_info

def format_content_with_media_enhanced(content: str, doc_num: int):
    """Format content and extract media references (same as rag_test_basic)"""
    media_info = extract_media_references_enhanced(content)
    formatted_content = content
    
    # Add media information if found
    if media_info['has_media']:
        media_section = f"\n[MEDIA IN DOC {doc_num}]"
        
        if media_info['images']:
            media_section += f"\n📷 Images/Figures: {len(media_info['images'])} found"
            for i, img in enumerate(media_info['images'][:3], 1):  # Show first 3
                media_section += f"\n  - Image {i}: {img[:100]}{'...' if len(img) > 100 else ''}"
        
        if media_info['tables']:
            media_section += f"\n📊 Tables: {len(media_info['tables'])} found"
            for i, table in enumerate(media_info['tables'][:2], 1):  # Show first 2
                table_preview = table[:150].replace('\n', ' ')
                media_section += f"\n  - Table {i}: {table_preview}{'...' if len(table) > 150 else ''}"
        
        formatted_content += media_section
    
    return formatted_content

@tool
def grade_document_relevance(question: str, document_content: str, chunk_label: str = "UNKNOWN") -> dict:
    """Grade the relevance of retrieved documents (EXACT from version3_refactor)."""
    logger.info(f"📊 Grading relevance for {chunk_label}")
    
    print(f"🔍 GRADING DEBUG: chunk_label={chunk_label}, content_length={len(document_content if document_content else '')}")

    if not document_content or document_content == "No relevant information found in the PDF.":
        logger.info(f"📉 {chunk_label} has no content, score=0")
        result = {"relevance_score": 0, "graded": True, "chunk": chunk_label}
        print(f"🔍 GRADING RESULT: {result}")
        return result
    
    # Truncate for speed
    if len(document_content) > 300:
        truncated_content = document_content[:1000] + "..."
        logger.info(f"✂️ Truncated {chunk_label} from {len(document_content)} to 1000 chars")
    else:
        truncated_content = document_content
    
    try:
        score = retrieval_grader.invoke({"question": question, "documents": truncated_content})
        relevance_score = score.get('score', 0)
        logger.info(f"✅ {chunk_label} relevance score = {relevance_score}")
        result = {
            "relevance_score": relevance_score,
            "graded": True,
            "chunk": chunk_label
        }
        print(f"🔍 GRADING RESULT: {result}")
        return result
    except Exception as e:
        logger.error(f"⚠️ Error grading {chunk_label}: {e}")
        result = {"relevance_score": 0, "graded": True, "chunk": chunk_label}
        print(f"🔍 GRADING RESULT (error fallback): {result}")
        return result

@tool
def search_web_for_vehicle_info(query: str, dtc_code: str = None, vehicle_info: str = None) -> dict:
    """Search web for additional vehicle diagnostic information using Tavily (EXACT from version3_refactor)."""
    logger.info("🌐 Performing Tavily web search")
    
    tavily_api_key = os.getenv("TAVILY_API_KEY")
    if not tavily_api_key:
        logger.warning("No Tavily API key found, skipping web search")
        return {"results": [], "success": False, "error": "Missing TAVILY_API_KEY"}
    
    try:
        from tavily import TavilyClient
        tavily_client = TavilyClient(api_key=tavily_api_key)
    except ImportError:
        logger.warning("Tavily package not installed, skipping web search")
        return {"results": [], "success": False, "error": "Tavily package not available"}
    
    search_term = dtc_code or vehicle_info or query
    
    if dtc_code:
        search_query = f"{search_term} vehicle diagnostic trouble code causes solutions"
        logger.info(f"Searching for DTC code: {search_term}")
    elif vehicle_info:
        search_query = f"{search_term} common problems solutions"
        logger.info(f"Searching for vehicle info: {search_term}")
    else:
        search_query = search_term
        logger.info(f"Searching for general query: {search_term}")
    
    try:
        response = tavily_client.search(query=search_query, max_results=3, search_depth="advanced")
        results = response.get('results', [])
        logger.info(f"Found {len(results)} results from web search")
        
        return {
            "results": results,
            "success": True,
            "query_used": search_query
        }
    except Exception as e:
        logger.error(f"Tavily search error: {e}")
        return {"results": [], "success": False, "error": str(e)}

@tool
def search_youtube_videos(query: str, dtc_code: str = None, vehicle_info: str = None) -> dict:
    """Search YouTube for diagnostic videos (EXACT from version3_refactor)."""
    logger.info("📺 Performing YouTube search")
    
    YOUTUBE_API_KEY = os.getenv("YOUTUBE_API_KEY")
    if not YOUTUBE_API_KEY:
        logger.warning("YouTube API key not found in environment variables")
        return {"youtube_results": [], "success": False,"error": "Missing YOUTUBE_API_KEY"}
    
    search_term = dtc_code or vehicle_info or query
    
    if dtc_code:
        search_query = f"{search_term} diagnostic trouble code repair"
        logger.info(f"Searching YouTube for DTC code: {search_term}")
    elif vehicle_info:
        search_query = f"{search_term} repair maintenance"
        logger.info(f"Searching YouTube for vehicle: {search_term}")
    else:
        search_query = f"car {search_term}"
        logger.info(f"Searching YouTube for general query: {search_term}")
    
    url = "https://www.googleapis.com/youtube/v3/search"
    params = {
        "part": "snippet",
        "q": search_query,
        "key": YOUTUBE_API_KEY,
        "maxResults": 4,
        "type": "video"
    }
    
    try:
        response = requests.get(url, params=params)
        if response.status_code != 200:
            logger.error(f"YouTube API error: {response.status_code} {response.text}")
            return {"youtube_results": [], "success": False}
        
        data = response.json()
        videos = []
        
        for item in data.get("items", []):
            video_id = item["id"]["videoId"]
            
            # Get the best available thumbnail
            thumbnails = item["snippet"]["thumbnails"]
            thumbnail_url = ""
            
            # Priority order: maxres > high > medium > default
            if "maxresdefault" in thumbnails:
                thumbnail_url = thumbnails["maxresdefault"]["url"]
            elif "high" in thumbnails:
                thumbnail_url = thumbnails["high"]["url"]
            elif "medium" in thumbnails:
                thumbnail_url = thumbnails["medium"]["url"]
            elif "default" in thumbnails:
                thumbnail_url = thumbnails["default"]["url"]
            else:
                # Fallback to manual URL construction
                thumbnail_url = f"https://img.youtube.com/vi/{video_id}/mqdefault.jpg"
            
            videos.append({
                "url": f"https://www.youtube.com/watch?v={video_id}",
                "video_id": video_id,
                "title": item["snippet"]["title"],
                "thumbnail": thumbnail_url
            })
        
        logger.info(f"Found {len(videos)} YouTube videos")
        return {
            "youtube_results": videos,
            "success": True,
            "query_used": search_query
        }
    except Exception as e:
        logger.error(f"YouTube search error: {e}")
        return {"youtube_results": [], "success": False, "error": str(e)}

@tool
def format_diagnostic_results(
    question: str,
    rag_answer: str,
    web_results: Optional[List[dict]] = None,
    youtube_results: Optional[List[dict]] = None,
    dtc_code: Optional[str] = None,
    relevance_score: int = 0,
) -> dict:
    """Format the final diagnostic results with proper structure for frontend."""
    logger.info("📝 Formatting final results")
    
    print(f"🔍 DEBUG: Received relevance_score = {relevance_score}")
    
    # Handle the rag_answer properly - it might be JSON string from tool result
    if isinstance(rag_answer, str):
        try:
            # Try to parse as JSON first
            rag_data = json.loads(rag_answer)
            rag_content = rag_data.get('answer', rag_answer)
            print(f"✅ Parsed JSON, extracted answer field")
        except json.JSONDecodeError:
            # If not JSON, use as-is
            rag_content = rag_answer
            print(f"✅ Using raw string content")
    else:
        rag_content = str(rag_answer)

    # Check if RAG found relevant information AND has good relevance score
    has_rag_info = (rag_content and 
                   rag_content != "No relevant information found in the PDF." and
                   "No relevant information found" not in rag_content)
    
    # CRITICAL: Use relevance score to decide formatting logic
    use_rag = has_rag_info and relevance_score == 1  # ← BOTH conditions needed
    
    print(f"🔍 DEBUG: has_rag_info={has_rag_info}, relevance_score={relevance_score}, use_rag={use_rag}")
    
    # Prepare structured web sources and YouTube videos
    structured_web_sources = []
    structured_youtube_videos = []
    
    if web_results:
        for result in web_results[:3]:  # Limit to 3 web sources
            if "url" in result and "title" in result:
                structured_web_sources.append({
                    "url": result["url"],
                    "title": result.get("title", "Web Source")
                })
    
    if youtube_results:
        for video in youtube_results[:4]:  # Limit to 4 YouTube videos
            if "url" in video:
                # Normalize URL: strip whitespace and newlines to prevent JSON contamination
                clean_url = video["url"].strip().replace('\n', '').replace('\r', '')
                video_id = extract_youtube_video_id(clean_url)
                
                # Use the thumbnail from the video data, fallback to constructed URL
                thumbnail_url = video.get("thumbnail", f"https://img.youtube.com/vi/{video_id}/mqdefault.jpg")
                if thumbnail_url:
                    thumbnail_url = thumbnail_url.strip().replace('\n', '').replace('\r', '')
                
                # Ensure we don't use the default fallback image
                if "default/default.jpg" in thumbnail_url:
                    thumbnail_url = f"https://img.youtube.com/vi/{video_id}/mqdefault.jpg"
                
                # Normalize title to prevent JSON issues
                clean_title = video.get("title", "Diagnostic Video").strip()
                
                structured_youtube_videos.append({
                    "url": clean_url,
                    "title": clean_title,
                    "thumbnail": thumbnail_url,
                    "video_id": video_id
                })
    
    # If RAG has good info AND good relevance score, process it for steps/images/tables
    if use_rag:
        logger.info("Using RAG answer with step/image/table formatting")
        processed_rag_content = process_content_with_inline_images(rag_content)
        
        # Remove YouTube URLs from content since we have them in structured format
        if structured_youtube_videos:
            for video in structured_youtube_videos:
                video_url = video["url"]
                # Remove the URL from the main content (handle various line formats)
                processed_rag_content = processed_rag_content.replace(video_url, "")
                # Also remove common patterns like "- {url}" or "• {url}"
                processed_rag_content = re.sub(rf'^[•\-\*]\s*{re.escape(video_url)}\s*$', '', processed_rag_content, flags=re.MULTILINE)
                # Clean up empty lines
                processed_rag_content = re.sub(r'\n\s*\n\s*\n', '\n\n', processed_rag_content)
            processed_rag_content = processed_rag_content.strip()
        
        # Clean any HTML artifacts and decode Unicode escape sequences
        processed_rag_content = clean_html_artifacts(processed_rag_content)
        voice_summary = create_voice_summary(processed_rag_content, question)
        
        return {
            "formatted_response": {
                "voice_output": voice_summary,
                "text_output": {
                    "content": processed_rag_content,
                    "web_sources": structured_web_sources,
                    "youtube_videos": structured_youtube_videos,
                    "has_external_sources": len(structured_web_sources) > 0 or len(structured_youtube_videos) > 0
                }
            }
        }
    
    # If RAG is not relevant (score 0) OR has no content, use web search
    if not use_rag and (web_results or youtube_results):
        logger.info("No RAG info found OR low relevance score, using web search formatting logic")
        print(f"🌐 Using web search logic - relevance_score={relevance_score}")
        
        # Combine web search content
        web_content = "\n\n".join([r.get("content", "") for r in web_results or [] if "content" in r])
        
        # Choose appropriate prompt based on presence of DTC code
        if dtc_code:
            prompt_template = f"""
You are an expert automotive diagnostic technician analyzing the Diagnostic Trouble Code (DTC): {dtc_code}

Based on the following information:
RAG ANSWER: {rag_answer}
WEB SEARCH RESULTS: {web_content}

Create a comprehensive diagnostic report that STRICTLY follows this EXACT format:

**Category:** [one-line description of what this DTC code represents]

**Potential Causes:**

• [cause 1]
• [cause 2]
• [continue until you have up to 5 causes, be specific and technical]

**Diagnostic Steps:**

• [step 1]
• [step 2]
• [continue until you have up to 5 clear diagnostic steps]

**Possible Solutions:**
• [solution 1]
• [solution 2]
• [solution 3]
• [continue until you have up to 5 solutions, be specific and technical]

Your response MUST follow this format exactly, with these exact section headings.
Be concise and technical in your bullet points. Do not add any other sections or explanations.
"""
        else:
            prompt_template = f"""
You are an automotive expert assistant. Create a comprehensive response based on the following information:

QUESTION: {question}

RAG ANSWER: {rag_answer}

WEB SEARCH RESULTS: 
{web_content}

Provide a detailed, helpful response that synthesizes all this information.
If the RAG answer indicates "No relevant information found in the PDF", prioritize the web search results.
Format your response with clear sections and bullet points for better readability.
"""
        
        try:
            response = llm.invoke(prompt_template)
            diagnostic_content = response.content.strip()
            
            # Remove YouTube URLs from content since we have them in structured format
            if structured_youtube_videos:
                for video in structured_youtube_videos:
                    video_url = video["url"]
                    # Remove the URL from the main content (handle various line formats)
                    diagnostic_content = diagnostic_content.replace(video_url, "")
                    # Also remove common patterns like "- {url}" or "• {url}"
                    diagnostic_content = re.sub(rf'^[•\-\*]\s*{re.escape(video_url)}\s*$', '', diagnostic_content, flags=re.MULTILINE)
                    # Clean up empty lines
                    diagnostic_content = re.sub(r'\n\s*\n\s*\n', '\n\n', diagnostic_content)
                diagnostic_content = diagnostic_content.strip()
            
            # Clean any HTML artifacts and decode Unicode escape sequences
            diagnostic_content = clean_html_artifacts(diagnostic_content)
            voice_summary = create_voice_summary(diagnostic_content, question)
            
            logger.info("Generated structured diagnostic report from web sources")
            return {
                "formatted_response": {
                    "voice_output": voice_summary,
                    "text_output": {
                        "content": diagnostic_content,
                        "web_sources": structured_web_sources,
                        "youtube_videos": structured_youtube_videos,
                        "has_external_sources": len(structured_web_sources) > 0 or len(structured_youtube_videos) > 0
                    }
                }
            }
        
        except Exception as e:
            logger.error(f"Error formatting results: {e}")
            voice_summary = create_voice_summary(rag_content, question)
            return {
                "formatted_response": {
                    "voice_output": voice_summary,
                    "text_output": {
                        "content": rag_content,
                        "web_sources": structured_web_sources,
                        "youtube_videos": structured_youtube_videos,
                        "has_external_sources": len(structured_web_sources) > 0 or len(structured_youtube_videos) > 0
                    }
                }
            }
    
    # Fallback: Return RAG content even if relevance is low (no web results available)
    logger.info("Fallback: Using RAG content despite low relevance (no web results)")
    print("⚠️ Fallback: No web results, using RAG content despite low relevance")
    processed_rag_content = process_content_with_inline_images(rag_content)
    voice_summary = create_voice_summary(processed_rag_content, question)
    
    return {
        "formatted_response": {
            "voice_output": voice_summary,
            "text_output": {
                "content": processed_rag_content,
                "web_sources": structured_web_sources,
                "youtube_videos": structured_youtube_videos,
                "has_external_sources": len(structured_web_sources) > 0 or len(structured_youtube_videos) > 0
            }
        }
    }

def process_content_with_inline_images(content: str) -> str:
    """Process content to display images and tables inline with steps (EXACT from version3_refactor)."""
    
    if not content or len(content) < 10:
        return content
    
    try:
        lines = content.split('\n')
        processed_lines = []
        
        for line in lines:
            line = line.strip()
            
            # Check for numbered steps
            if re.match(r'^\d+\.\s+', line):
                processed_lines.append(f"**{line}**")
            
            # Check for YES/NO decision points and bullet points
            elif line.startswith(('YES -', 'NO -', '- YES', '- NO', '-')):
                processed_lines.append(f"   **{line}**")
            
            # Check for image references - show markdown as-is
            elif '![Image](' in line:
                processed_lines.append(f"   {line}")
                processed_lines.append("")  # Space after image
            
            # Check for table rows (contains | characters)
            elif '|' in line and len([c for c in line if c == '|']) >= 2:
                processed_lines.append(line)  # Keep table formatting as-is
            
            # Check for table separators
            elif '---' in line and '|' in line:
                processed_lines.append(line)
            
            # Check for questions (ending with ?)
            elif line.endswith('?'):
                processed_lines.append(f"\n**{line}**")
            
            else:
                if line:
                    processed_lines.append(line)
        
        return '\n'.join(processed_lines)
        
    except Exception as e:
        logger.error(f"Error in process_content_with_inline_images: {e}")
        return content


def clean_html_artifacts(content: str) -> str:
    """Remove HTML artifacts and properly decode Unicode escape sequences."""
    # Remove HTML tags
    content = re.sub(r'<[^>]+>', '', content)
    # Remove escaped quotes and other HTML entities
    content = content.replace('&quot;', '"').replace('&amp;', '&').replace('&lt;', '<').replace('&gt;', '>')
    
    # Decode common Unicode escape sequences
    content = content.replace('\\u2022', '•')  # bullet point
    content = content.replace('\\u2013', '–')  # en dash
    content = content.replace('\\u2014', '—')  # em dash
    content = content.replace('\\u201c', '"')  # left double quote
    content = content.replace('\\u201d', '"')  # right double quote
    content = content.replace('\\u2019', "'")  # right single quote
    
    # Replace escaped newlines with actual newlines
    content = content.replace('\\n', '\n')
    
    # Clean up multiple spaces and newlines but preserve paragraph structure
    content = re.sub(r' +', ' ', content)  # Multiple spaces to single space
    content = re.sub(r'\n\s*\n\s*\n+', '\n\n', content)  # Multiple newlines to double newline
    
    return content.strip()


def extract_youtube_video_id(url: str) -> str:
    """Extract YouTube video ID from URL."""
    import re
    
    # Match various YouTube URL formats
    patterns = [
        r'(?:https?://)?(?:www\.)?youtube\.com/watch\?v=([a-zA-Z0-9_-]+)',
        r'(?:https?://)?(?:www\.)?youtu\.be/([a-zA-Z0-9_-]+)',
        r'(?:https?://)?(?:www\.)?youtube\.com/embed/([a-zA-Z0-9_-]+)',
    ]
    
    for pattern in patterns:
        match = re.search(pattern, url)
        if match:
            return match.group(1)
    
    # If no match found, return a default ID
    return "dQw4w9WgXcQ"  # Rick Roll as fallback :)


def create_voice_summary(diagnostic_content: str, question: str) -> str:
    """Create a concise voice summary from detailed diagnostic content."""
    try:
        # Create a concise summary for voice output (3-4 sentences max)
        summary_prompt = f"""
Based on this detailed diagnostic information:

{diagnostic_content[:800]}...

Create a concise voice summary that answers the user's question: "{question}"

Requirements:
- Maximum 3-4 sentences
- Focus on the most critical findings and immediate actions
- Use conversational tone suitable for speech
- Mention key diagnostic points but avoid lengthy explanations
- Under 80 words total

Example format: "The [code] indicates [main issue]. The primary causes are [brief list]. I recommend [key action]. Check the diagnostic panel for detailed steps and resources."
"""
        
        response = llm.invoke(summary_prompt)
        voice_summary = response.content.strip()
        
        # Ensure it's not too long for TTS
        if len(voice_summary) > 300:
            # Truncate to first 3 sentences if too long
            sentences = voice_summary.split('.')
            voice_summary = '. '.join(sentences[:3]) + '.'
        
        # Fallback to first paragraph if LLM fails
        if not voice_summary or len(voice_summary) < 20:
            lines = diagnostic_content.split('\n')
            for line in lines:
                if line.strip() and len(line.strip()) > 30:
                    # Create a simple summary from the content
                    return f"I found diagnostic information for your question. {line.strip()[:150]}... Check the diagnostic panel for complete details."
        
        return voice_summary
        
    except Exception as e:
        logger.error(f"Error creating voice summary: {e}")
        # Simple fallback
        return f"I found diagnostic information for your {question}. Please check the detailed report in the diagnostic panel for complete analysis, steps, and resources."

# Export all tools
__all__ = [
    "is_vehicle_related",
    "extract_vehicle_model", 
    "search_vehicle_documents",
    "grade_document_relevance",
    "search_web_for_vehicle_info",
    "search_youtube_videos",
    "format_diagnostic_results",
    "llm",
    "retriever",
]

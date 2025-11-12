# setup.py
import os
import re
import json
import pickle
import numpy as np
from pathlib import Path
from dotenv import load_dotenv
from typing import List, Dict, Any
import sys
import csv

current_dir = Path(__file__).resolve().parent
parent_dir = current_dir.parent
sys.path.insert(0, str(parent_dir))

from configs.RAG_config import PDF_FOLDER, MARKDOWN_DIR, CHUNKS_DIR, FAISS_DB_DIR, FAISS_METADATA_PATH,FAISS_INDEX_PATH,VECTOR_STORE_TYPE
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import (
    PdfPipelineOptions,
    PictureDescriptionApiOptions,
)
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling_core.types.doc.document import ImageRefMode

# FAISS imports
try:
    import faiss
    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False
    print("⚠️ FAISS not installed. Install with: pip install faiss-cpu (or faiss-gpu)")

# Updated deprecated import
try:
    from langchain_openai import OpenAIEmbeddings  # modern location
except ImportError:  # fallback for older envs
    from langchain.embeddings.openai import OpenAIEmbeddings  # type: ignore

load_dotenv()
api_key = os.environ.get("OPENAI_API_KEY")

# -----------------------------
# Helpers
# -----------------------------
def ensure_dirs():
    """Make sure all output directories exist before running pipeline."""
    for d in [MARKDOWN_DIR, CHUNKS_DIR, FAISS_DB_DIR]:
        Path(d).mkdir(parents=True, exist_ok=True)
    print(f"✅ Ensured directories: {MARKDOWN_DIR}, {CHUNKS_DIR}, {FAISS_DB_DIR}")

# -----------------------------
# 1️⃣ PDF Parsing with Docling
# -----------------------------
def convert_with_image_annotation(input_doc_path):
    pipeline_options = PdfPipelineOptions(
        do_ocr=True,
        do_table_structure=True,
        preserve_font_styles=True,
        preserve_font_colors=True,
        preserve_layout=True,
        generate_page_images=True,
        enable_remote_services=False,
        do_picture_description=False,
    )

    converter = DocumentConverter(
        format_options={InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)}
    )
    conv_res = converter.convert(source=input_doc_path)
    return conv_res

def export_single_md_with_images_and_serials(conv_res, output_path: Path):
    output_path = Path(output_path)
    output_path.mkdir(parents=True, exist_ok=True)
    doc_filename = conv_res.input.file.stem.replace(" ", "_")
    md_filename = output_path / f"{doc_filename}-full-with-serials.md"

    conv_res.document.save_as_markdown(
        md_filename,
        image_mode=ImageRefMode.REFERENCED,
        include_annotations=True,
        page_break_placeholder="<!-- PAGE_BREAK -->",
    )

    # Add page-end markers
    with open(md_filename, "r", encoding="utf-8") as f:
        md_text = f.read()

    pages = md_text.split("<!-- PAGE_BREAK -->")
    final_md = ""
    for idx, page_text in enumerate(pages, start=1):
        final_md += page_text.strip() + f"\n\n<!-- PAGE {idx} END -->\n\n"

    Path(md_filename).write_text(final_md, encoding="utf-8")
    print(f"✅ Markdown saved with serials → {md_filename}")
    return md_filename

def parse_pdfs():
    print("📄 Parsing PDFs with Docling pipeline...")
    markdown_files = []
    pdf_dir = Path(PDF_FOLDER)  # ✅ convert str → Path
    for pdf_path in pdf_dir.glob("*.pdf"):
        print(f"📄 Processing: {pdf_path.name}")
        conv_res = convert_with_image_annotation(pdf_path)
        md_file = export_single_md_with_images_and_serials(conv_res, MARKDOWN_DIR)
        markdown_files.append(md_file)
    return markdown_files

# -----------------------------
# 2️⃣ Hybrid Chunking
# -----------------------------

# ...existing code...

def save_debug_log(log_entries, file_name="chunk_debug_log.txt"):
    """Save a detailed debug log showing which chunks were merged or kept."""
    debug_file = Path(CHUNKS_DIR) / file_name
    try:
        with open(debug_file, "w", encoding="utf-8") as f:
            f.write("### 🧩 CHUNKING DEBUG LOG ###\n\n")
            for entry in log_entries:
                f.write(f"{entry}\n")
        print(f"🐞 Debug log saved to: {debug_file}")
    except Exception as e:
        print(f"⚠️ Could not save debug log: {str(e)}")

def simple_heading_based_chunking(markdown_files, min_chars=1500, max_chars=4000):
    """
    Simple heading-based chunking:
    1. Each heading = one chunk initially
    2. Merge small adjacent chunks if total < min_chars
    3. Keep large chunks (>max_chars) as single chunks
    """
    
    def extract_heading_sections(text):
        """Extract sections based on ## headings."""
        # Split by ## headings but keep the headings
        sections = []
        lines = text.split('\n')
        current_section = {'heading': '', 'content': '', 'start_line': 0}
        
        for i, line in enumerate(lines):
            # Check if this is a main heading (## but not ###)
            if re.match(r'^##\s+(?!#)', line.strip()):
                # Save previous section if it has content
                if current_section['content'].strip():
                    sections.append(current_section.copy())
                
                # Start new section
                current_section = {
                    'heading': line.strip(),
                    'content': line + '\n',
                    'start_line': i
                }
            else:
                # Add to current section
                current_section['content'] += line + '\n'
        
        # Don't forget the last section
        if current_section['content'].strip():
            sections.append(current_section)
        
        return sections
    
    def get_section_metadata(content, heading):
        """Extract metadata from section content."""
        # Extract page numbers
        page_matches = re.findall(r'<!-- PAGE (\d+) END -->', content)
        pages = sorted(list(set(map(int, page_matches)))) if page_matches else [1]
        
        # Determine content type based on heading and content
        heading_upper = heading.upper()
        
        return {
            'pages': pages,
            'char_count': len(content),
            'has_tables': '|' in content and '---' in content,
            'has_figures': 'Image](' in content or '![' in content,
            'has_steps': bool(re.search(r'^\d+\.\s+', content, re.MULTILINE))
        }
    
    def create_chunk_from_sections(sections_to_merge, source_file):
        """Create a single chunk from one or more sections."""
        # Combine content
        combined_content = ""
        combined_heading_parts = []
        all_pages = set()
        
        for section in sections_to_merge:
            combined_content += section['content'] + "\n"
            # Clean heading (remove ## symbols)
            clean_heading = re.sub(r'^#{1,4}\s*', '', section['heading']).strip()
            if clean_heading:
                combined_heading_parts.append(clean_heading)
            
            # Get metadata for this section
            metadata = get_section_metadata(section['content'], section['heading'])
            all_pages.update(metadata['pages'])
        
        # Create final heading
        final_heading = ' | '.join(combined_heading_parts) if combined_heading_parts else "Content"
        
        # Get overall metadata
        final_metadata = get_section_metadata(combined_content, final_heading)
        final_metadata['pages'] = sorted(list(all_pages))
        
        return {
            'heading': final_heading,
            'chunk_text': combined_content.strip(),
            'source_file': source_file,
            'original_text': combined_content.strip(),
            **final_metadata
        }
    
    def merge_small_chunks(sections, source_file, *, debug=True, greedy_fill=True):
        """
        Merge adjacent small sections into larger chunks (min_chars / max_chars rules).
        - greedy_fill=True → keeps merging until next section would exceed max_chars.
        - Relies on outer-scope: min_chars, max_chars
        - Uses outer function: create_chunk_from_sections
        """
        chunks = []
        i = 0

        def clean_heading(h):
            """Remove leading ## and extra spaces from heading text."""
            return re.sub(r'^#{1,6}\s*', '', (h or '')).strip()

        while i < len(sections):
            current_sections = [sections[i]]
            current_size = len(sections[i]['content'])
            start_idx = i

            # If section already large enough, keep it as is
            if current_size >= min_chars and not greedy_fill:
                chunk = create_chunk_from_sections(current_sections, source_file)
                chunks.append(chunk)
                if debug:
                    print(f"📦 Large section kept as single chunk: '{clean_heading(sections[i]['heading'])}' ({current_size} chars)")
                i += 1
                continue

            j = i + 1

            # ✅ Greedy merging logic
            if greedy_fill:
                while j < len(sections):
                    next_size = len(sections[j]['content'])
                    if current_size + next_size > max_chars:
                        break
                    current_sections.append(sections[j])
                    current_size += next_size
                    j += 1
            else:
                # Old logic: stop as soon as min_chars is reached
                while j < len(sections) and current_size < min_chars:
                    next_size = len(sections[j]['content'])
                    if current_size + next_size > max_chars:
                        break
                    current_sections.append(sections[j])
                    current_size += next_size
                    j += 1
                    if current_size >= min_chars:
                        break

            # Create final chunk
            chunk = create_chunk_from_sections(current_sections, source_file)
            chunks.append(chunk)

            merged_count = len(current_sections)
            headings = [clean_heading(s['heading']) for s in current_sections]

            # Print debug info
            if merged_count > 1:
                print(f"🔗 Merged {merged_count} sections: {headings} ({current_size} chars)")
            else:
                print(f"📦 Section kept as chunk: '{headings[0]}' ({current_size} chars)")

            i = j

        print(f"\n✅ Created {len(chunks)} final chunks from {len(sections)} sections\n")
        return chunks
    
    # Main processing
    all_chunks = []
    
    # Handle both single file and list of files
    if isinstance(markdown_files, (str, Path)):
        markdown_files = [markdown_files]
    
    for md_file in markdown_files:
        try:
            md_path = Path(md_file)
            if not md_path.exists():
                print(f"⚠️ Markdown file not found: {md_path}")
                continue
                
            print(f"📖 Processing: {md_path.name}")
            
            with open(md_path, 'r', encoding='utf-8') as f:
                text = f.read()
            
            # Extract sections by headings
            sections = extract_heading_sections(text)
            print(f"   🔍 Found {len(sections)} heading sections")
            
            # Merge small sections into optimal chunks
            file_chunks = merge_small_chunks(sections, md_path.stem, debug=True)

            
            all_chunks.extend(file_chunks)
            print(f"   ✅ Created {len(file_chunks)} final chunks")
            
        except Exception as e:
            print(f"❌ Error processing {md_file}: {str(e)}")
            continue
    
    # Save chunks preview
    save_chunks_preview(all_chunks)
    
    print(f"\n📊 Chunking Summary:")
    print(f"   • Total chunks: {len(all_chunks)}")
    print(f"   • Average chunk size: {sum(len(c['chunk_text']) for c in all_chunks) // len(all_chunks) if all_chunks else 0} chars")
    print(f"   • Size range: {min(len(c['chunk_text']) for c in all_chunks) if all_chunks else 0} - {max(len(c['chunk_text']) for c in all_chunks) if all_chunks else 0} chars")
    
    return all_chunks

def chunk_markdowns(markdown_files, min_chars=1500):
    """Updated function to use simple heading-based chunking."""
    return simple_heading_based_chunking(markdown_files, min_chars)

def save_chunks_preview(chunks, max_preview_chunks=15):
    """Save a preview of chunks to file for inspection."""
    try:
        preview_file = Path(CHUNKS_DIR) / "chunks_preview.md"
        
        with open(preview_file, 'w', encoding='utf-8') as f:
            f.write("# 📋 Chunks Preview\n\n")
            f.write(f"**Total Chunks:** {len(chunks)}\n\n")
            f.write("---\n\n")
            
            for i, chunk in enumerate(chunks[:max_preview_chunks]):
                f.write(f"### 🧩 Chunk {i+1}: {chunk.get('heading', 'No Heading')}\n")
                f.write(f"**Pages:** {', '.join(map(str, chunk.get('pages', [1])))}\n")
                f.write(f"**Character Count:** {chunk.get('char_count', len(chunk['chunk_text']))}\n")
                f.write("---\n\n")
                f.write(f"🔸 **{chunk['chunk_text']}**\n\n")
            
            if len(chunks) > max_preview_chunks:
                f.write(f"... and {len(chunks) - max_preview_chunks} more chunks\n")
        
        print(f"📄 Chunks preview saved to: {preview_file}")
        
    except Exception as e:
        print(f"⚠️ Could not save chunks preview: {str(e)}")

# -----------------------------
# 3️⃣ Enhanced Vector Store Implementation
# -----------------------------

class BaseVectorStore:
    """Base class for vector stores."""
    
    def __init__(self):
        self.embeddings = OpenAIEmbeddings(
            api_key=api_key,
            model="text-embedding-3-small"
        )
    
    def add_documents(self, chunks: List[Dict[str, Any]]) -> bool:
        """Add documents to the vector store."""
        raise NotImplementedError
    
    def query(self, query: str, n_results: int = 5) -> Dict[str, Any]:
        """Query the vector store."""
        raise NotImplementedError
    
    def get_stats(self) -> Dict[str, Any]:
        """Get statistics about the vector store."""
        raise NotImplementedError


class FAISSVectorStore(BaseVectorStore):
    """FAISS implementation with metadata support."""
    
    def __init__(self):
        super().__init__()
        self.index = None
        self.metadata_store = []
        self.id_to_index = {}
        self.batch_size = 50
    
    def add_documents(self, chunks: List[Dict[str, Any]]) -> bool:
        """Add documents to FAISS."""
        print(f"🔎 Building FAISS embeddings with metadata...")
        
        if not chunks:
            print("⚠️ No chunks provided.")
            return False
        
        # Prepare texts for embedding
        texts = [chunk['chunk_text'] for chunk in chunks]
        print(f"   🔄 Generating embeddings for {len(texts)} chunks...")
        
        try:
            # Generate embeddings in batches
            all_embeddings = []
            for i in range(0, len(texts), self.batch_size):
                batch_texts = texts[i:i+self.batch_size]
                batch_embeddings = self.embeddings.embed_documents(batch_texts)
                all_embeddings.extend(batch_embeddings)
                print(f"   📦 Generated embeddings for batch {i//self.batch_size + 1}: {len(batch_texts)} chunks")
            
            # Convert to numpy array
            embeddings_array = np.array(all_embeddings, dtype=np.float32)
            print(f"   📊 Embeddings shape: {embeddings_array.shape}")
            
            # Create FAISS index (using flat index for simplicity)
            dimension = embeddings_array.shape[1]
            self.index = faiss.IndexFlatIP(dimension)  # Inner product for cosine similarity
            
            # Normalize embeddings for cosine similarity
            faiss.normalize_L2(embeddings_array)
            
            # Add embeddings to index
            self.index.add(embeddings_array)
            
            # Store metadata separately
            self.metadata_store = []
            self.id_to_index = {}
            
            for i, chunk in enumerate(chunks):
                chunk_id = f"{chunk.get('source_file', 'unknown')}_{i}"
                self.id_to_index[chunk_id] = i
                
                metadata = {
                    'id': chunk_id,
                    'text': chunk['chunk_text'],
                    'original_text': chunk.get('original_text', chunk['chunk_text']),
                    'pages': chunk.get('pages', [1]),
                    'page_count': len(chunk.get('pages', [1])),
                    'heading': chunk.get('heading', ''),
                    'source_file': chunk.get('source_file', ''),
                    'chunk_index': i,
                    'has_tables': chunk.get('has_tables', False),
                    'has_figures': chunk.get('has_figures', False),
                    'content_type': 'mixed' if (chunk.get('has_tables', False) or chunk.get('has_figures', False)) else 'text',
                    'text_length': len(chunk.get('original_text', chunk['chunk_text'])),
                    'is_introduction': bool(re.search(r'\b(introduction|overview|summary)\b', chunk.get('heading', ''), re.IGNORECASE)),
                    'is_conclusion': bool(re.search(r'\b(conclusion|summary|results?|findings?)\b', chunk.get('heading', ''), re.IGNORECASE)),
                    'is_technical': bool(re.search(r'\b(algorithm|method|approach|implementation|technical|specification)\b', chunk['chunk_text'], re.IGNORECASE)),
                }
                self.metadata_store.append(metadata)
            
            # Save index and metadata to disk
            self._save_to_disk()
            
            print(f"✅ Stored {len(chunks)} chunks in FAISS index")
            return True
            
        except Exception as e:
            print(f"❌ Error building FAISS index: {str(e)}")
            import traceback
            traceback.print_exc()
            return False

    
    def get_stats(self) -> Dict[str, Any]:
        """Get FAISS statistics."""
        if self.index is None:
            self._load_from_disk()
        
        if self.index is None:
            return {"total_documents": 0}
        
        return {
            "total_documents": self.index.ntotal,
            "index_type": "flat",
            "dimension": self.index.d if hasattr(self.index, 'd') else 1536
        }
    
    def _save_to_disk(self):
        """Save FAISS index and metadata to disk."""
        try:
            # Ensure directory exists
            Path(FAISS_INDEX_PATH).parent.mkdir(parents=True, exist_ok=True)
            
            # Save FAISS index
            faiss.write_index(self.index, FAISS_INDEX_PATH)
            
            # Save metadata
            with open(FAISS_METADATA_PATH, 'wb') as f:
                pickle.dump({
                    'metadata_store': self.metadata_store,
                    'id_to_index': self.id_to_index
                }, f)
            
            print(f"💾 FAISS index saved to: {FAISS_INDEX_PATH}")
            print(f"💾 Metadata saved to: {FAISS_METADATA_PATH}")
            
        except Exception as e:
            print(f"❌ Error saving FAISS to disk: {str(e)}")
            import traceback
            traceback.print_exc()
    
    def _load_from_disk(self) -> bool:
        """Load FAISS index and metadata from disk."""
        try:
            # Check if files exist
            if not Path(FAISS_INDEX_PATH).exists() or not Path(FAISS_METADATA_PATH).exists():
                return False
            
            # Load FAISS index
            self.index = faiss.read_index(FAISS_INDEX_PATH)
            
            # Load metadata
            with open(FAISS_METADATA_PATH, 'rb') as f:
                data = pickle.load(f)
                self.metadata_store = data['metadata_store']
                self.id_to_index = data['id_to_index']
            
            print(f"📂 FAISS index loaded from: {FAISS_INDEX_PATH}")
            print(f"📂 Metadata loaded from: {FAISS_METADATA_PATH}")
            return True
            
        except Exception as e:
            print(f"❌ Error loading FAISS from disk: {str(e)}")
            return False

def create_vector_store() -> BaseVectorStore:
    """Factory function to create the appropriate vector store."""
    if VECTOR_STORE_TYPE == 'faiss':
        if not FAISS_AVAILABLE:
            print("❌ FAISS not available, falling back to ChromaDB")
        print(f"🏗️ Using FAISS vector store")
        return FAISSVectorStore()
    else:
        raise ValueError(f"Unsupported vector store: {VECTOR_STORE_TYPE}")

def build_embeddings(chunks):
    """Enhanced build embeddings function using the vector store factory."""
    vector_store = create_vector_store()
    success = vector_store.add_documents(chunks)
    
    if success:
        # Print statistics
        stats = vector_store.get_stats()
        print(f"📊 Vector Store Statistics:")
        for key, value in stats.items():
            print(f"   • {key}: {value}")
        
        print(f"✅ Vector store setup complete!")
        return vector_store
    else:
        print(f"❌ Vector store setup failed!")
        return None

# -----------------------------
# Main
# -----------------------------
def main():
    load_dotenv()
    ensure_dirs()
    markdowns = parse_pdfs()
    chunks = chunk_markdowns(markdowns)
    vector_store = build_embeddings(chunks)
    
    if vector_store:
        print("\n🎉 Setup complete! Multimodal RAG is ready.")
        

if __name__ == "__main__":
    main()
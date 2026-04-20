from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_experimental.text_splitter import SemanticChunker
from langchain_openai import OpenAIEmbeddings
from loaders.extract_tables import TableExtractor
from loaders.img_extraction import ImageExtractor
from loaders.img_description import get_table_documents, get_image_descriptions
from config import Config

config = Config()


def load_pdf(file_path: str, table_dir="./data/extracted_tables", img_dir="./data/extracted_images"):
    TableExtractor().extract_tables(file_path, output_folder=table_dir)
    ImageExtractor().extract_images(file_path, output_folder=img_dir)

    # 표: 별도 Document 생성 + 페이지 본문에 삽입할 참조 텍스트
    table_references, table_docs = get_table_documents(table_dir)

    # 이미지: 페이지 본문에 삽입할 설명
    image_descriptions = get_image_descriptions(img_dir)

    loader = PyPDFLoader(file_path)
    documents = loader.load()

    for doc in documents:
        page_num = doc.metadata.get("page", 0) + 1
        extras = []
        if page_num in table_references:
            extras.extend(table_references[page_num])
        if page_num in image_descriptions:
            extras.extend(image_descriptions[page_num])
        if extras:
            doc.page_content += "\n\n" + "\n".join(extras)

    # 표 번호 → Document 직접 조회용 레지스트리 (벡터 검색 우회)
    table_registry = {d.metadata["table_number"]: d for d in table_docs}

    # 페이지 문서 + 표 문서를 합쳐 반환
    return documents + table_docs, table_registry


def split_documents(documents):
    embeddings = OpenAIEmbeddings(model=config.EMBEDDING_MODEL)

    # 1차: 의미 기반 분할 - 임베딩 유사도가 급변하는 지점에서 분리
    semantic_splitter = SemanticChunker(
        embeddings=embeddings,
        breakpoint_threshold_type="percentile",
        breakpoint_threshold_amount=80,
    )
    semantic_chunks = semantic_splitter.split_documents(documents)

    # 2차: 너무 큰 청크는 추가 분할
    size_splitter = RecursiveCharacterTextSplitter(
        chunk_size=config.CHUNK_SIZE,
        chunk_overlap=config.CHUNK_OVERLAP,
    )
    final_chunks = size_splitter.split_documents(semantic_chunks)

    return final_chunks
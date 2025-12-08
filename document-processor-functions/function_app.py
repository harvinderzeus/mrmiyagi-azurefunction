import fitz  # PyMuPDF
import azure.functions as func
import json
import logging
import requests
from azure.ai.formrecognizer import DocumentAnalysisClient
from azure.core.credentials import AzureKeyCredential
from openai import AzureOpenAI
from azure.storage.blob import BlobServiceClient
import os
from typing import Dict, List, Any
from datetime import datetime
import uuid
from azure.eventhub import EventHubProducerClient, EventData


# ============= CONFIGURATION =============
FORM_RECOGNIZER_ENDPOINT = os.environ.get("FORM_RECOGNIZER_ENDPOINT", "")
FORM_RECOGNIZER_KEY = os.environ.get("FORM_RECOGNIZER_KEY", "")
OPENAI_ENDPOINT = os.environ.get("OPENAI_ENDPOINT", "")
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")
OPENAI_DEPLOYMENT = os.environ.get("OPENAI_DEPLOYMENT", "gpt-4o")

# Azure Blob Storage Configuration
STORAGE_ACCOUNT_NAME = os.environ.get("STORAGE_ACCOUNT_NAME", "sadevmrmiyagi")
STORAGE_ACCOUNT_KEY = os.environ.get("STORAGE_ACCOUNT_KEY", "")
STORAGE_CONTAINER_NAME = "dev-mr-miyagi"

# Event Hub Configuration for Logging
EVENT_HUB_CONNECTION_STRING = os.environ.get("EVENT_HUB_CONNECTION_STRING", "")
EVENT_HUB_NAME = os.environ.get("EVENT_HUB_NAME", "dev-mrmiyagi-logs")

app = func.FunctionApp()

# Generate unique run ID for each function execution
RUN_ID = None
JOB_ID = "azure_function"


# ============= LOGGING FUNCTION =============

def log_event(topic, sub_topic, message, status="info", file_name=None, error_details=None,
              total_pages=None, input_tokens=None, output_tokens=None, embedding_tokens=None, api_call=None):
    """Log an event to Event Hub"""

    global RUN_ID
    if RUN_ID is None:
        RUN_ID = f"run_{str(uuid.uuid4())}"

    # Create log entry
    log_entry = {
        "timestamp": datetime.utcnow().isoformat(),
        "run_id": RUN_ID,
        "job_id": JOB_ID,
        "status": status,
        "topic": topic,
        "sub_topic": sub_topic,
        "file_name": file_name,
        "message": message,
        "error_details": error_details,
        "total_pages": total_pages,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "embedding_tokens": embedding_tokens,
        "api_calls": api_call
    }

    # Send to Event Hub
    if EVENT_HUB_CONNECTION_STRING:
        try:
            producer = EventHubProducerClient.from_connection_string(
                conn_str=EVENT_HUB_CONNECTION_STRING,
                eventhub_name=EVENT_HUB_NAME
            )
            with producer:
                event_data_batch = producer.create_batch()
                event_data_batch.add(EventData(json.dumps(log_entry)))
                producer.send_batch(event_data_batch)
            logging.info(f"[LOG SENT] {topic} - {sub_topic}")
        except Exception as e:
            logging.error(f"[LOG ERROR] Failed to send to Event Hub: {e}")
            logging.info(f"[LOG FALLBACK] {json.dumps(log_entry)}")
    else:
        logging.info(f"[LOG] {json.dumps(log_entry)}")


# ============= HELPER FUNCTIONS =============

def get_blob_client(blob_name: str):
    """Get blob client for a specific blob"""
    connection_string = f"DefaultEndpointsProtocol=https;AccountName={STORAGE_ACCOUNT_NAME};AccountKey={STORAGE_ACCOUNT_KEY};EndpointSuffix=core.windows.net"
    blob_service_client = BlobServiceClient.from_connection_string(
        connection_string)
    return blob_service_client.get_blob_client(container=STORAGE_CONTAINER_NAME, blob=blob_name)


def read_file_from_blob(file_path: str) -> bytes:
    """
    Read file from blob storage (PDF or Markdown)
    Expected format: dev-mr-miyagi/files/file1.pdf or dev-mr-miyagi/files/file1.md
    """
    try:
        # Remove container name prefix if present
        if file_path.startswith(f"{STORAGE_CONTAINER_NAME}/"):
            blob_name = file_path[len(STORAGE_CONTAINER_NAME)+1:]
        else:
            blob_name = file_path

        blob_client = get_blob_client(blob_name)
        file_bytes = blob_client.download_blob().readall()

        log_event(
            topic="file_operations",
            sub_topic="read_blob",
            message=f"Successfully read file from blob: {blob_name}",
            status="success",
            file_name=blob_name
        )

        return file_bytes
    except Exception as e:
        log_event(
            topic="file_operations",
            sub_topic="read_blob",
            message=f"Error reading file from blob storage",
            status="error",
            file_name=file_path,
            error_details=str(e)
        )
        raise


def extract_text_with_document_intelligence(pdf_bytes: bytes) -> Dict[int, Any]:
    """Extract text from PDF using Azure Document Intelligence"""
    try:
        log_event(
            topic="document_intelligence",
            sub_topic="extraction_start",
            message="Starting text extraction with Document Intelligence",
            status="info"
        )

        client = DocumentAnalysisClient(
            endpoint=FORM_RECOGNIZER_ENDPOINT,
            credential=AzureKeyCredential(FORM_RECOGNIZER_KEY)
        )

        poller = client.begin_analyze_document("prebuilt-layout", pdf_bytes)
        result = poller.result()

        # Store full page information including layout details
        pages_data = {}

        for page in result.pages:
            page_num = page.page_number
            page_info = {
                "page_number": page_num,
                "width": page.width,
                "height": page.height,
                "lines": [],
                "text": []
            }

            if hasattr(page, 'lines'):
                for line in page.lines:
                    line_data = {
                        "content": line.content,
                        "bounding_box": line.polygon if hasattr(line, 'polygon') else None
                    }
                    page_info["lines"].append(line_data)
                    page_info["text"].append(line.content)

            # Join text for easy access
            page_info["full_text"] = "\n".join(page_info["text"])
            pages_data[page_num] = page_info

        log_event(
            topic="document_intelligence",
            sub_topic="extraction_complete",
            message=f"Successfully extracted text from {len(pages_data)} pages",
            status="success",
            total_pages=len(pages_data),
            api_call="begin_analyze_document"
        )

        return pages_data

    except Exception as e:
        log_event(
            topic="document_intelligence",
            sub_topic="extraction_error",
            message="Error in document intelligence extraction",
            status="error",
            error_details=str(e),
            api_call="begin_analyze_document"
        )
        raise


def extract_markdown_with_document_intelligence(pdf_bytes: bytes) -> str:
    """Extract markdown from PDF using Azure Document Intelligence with markdown output"""
    try:
        log_event(
            topic="document_intelligence",
            sub_topic="markdown_extraction_start",
            message="Starting markdown extraction with Document Intelligence",
            status="info"
        )

        client = DocumentAnalysisClient(
            endpoint=FORM_RECOGNIZER_ENDPOINT,
            credential=AzureKeyCredential(FORM_RECOGNIZER_KEY)
        )

        # Use markdown output format
        poller = client.begin_analyze_document(
            model_id="prebuilt-layout",
            document=pdf_bytes,
            output_content_format="markdown"
        )
        result = poller.result()

        # Get markdown content
        markdown_content = result.content if hasattr(result, 'content') else ""

        log_event(
            topic="document_intelligence",
            sub_topic="markdown_extraction_complete",
            message=f"Successfully extracted markdown content ({len(markdown_content)} chars)",
            status="success",
            api_call="begin_analyze_document"
        )

        return markdown_content

    except Exception as e:
        log_event(
            topic="document_intelligence",
            sub_topic="markdown_extraction_error",
            message="Error in markdown extraction",
            status="error",
            error_details=str(e),
            api_call="begin_analyze_document"
        )
        raise


def save_extracted_json(file_path: str, extraction_data: Dict, output_folder: str = None) -> str:
    """
    Save extraction data to blob storage
    Input: dev-mr-miyagi/files/file1.pdf
    Output: dev-mr-miyagi/outcome/file1/file1-extracted.json (or custom output_folder)
    """
    try:
        # Extract filename without extension
        if file_path.startswith(f"{STORAGE_CONTAINER_NAME}/"):
            file_path = file_path[len(STORAGE_CONTAINER_NAME)+1:]

        filename = os.path.basename(file_path).rsplit('.', 1)[0]

        # Create output path
        if output_folder:
            output_blob_name = f"{output_folder}/{filename}-extracted.json"
        else:
            output_blob_name = f"outcome/{filename}/{filename}-extracted.json"

        # Upload JSON
        blob_client = get_blob_client(output_blob_name)
        json_data = json.dumps(extraction_data, indent=2)
        blob_client.upload_blob(json_data, overwrite=True)

        output_path = f"{STORAGE_CONTAINER_NAME}/{output_blob_name}"

        log_event(
            topic="file_operations",
            sub_topic="save_json",
            message=f"Saved extracted JSON to: {output_path}",
            status="success",
            file_name=output_blob_name
        )

        return output_path

    except Exception as e:
        log_event(
            topic="file_operations",
            sub_topic="save_json_error",
            message="Error saving extracted JSON",
            status="error",
            file_name=file_path,
            error_details=str(e)
        )
        raise


def format_classification_results(classification_result: Dict) -> Dict:
    """Format classification results"""
    formatted = {}

    if "documents" in classification_result:
        for doc in classification_result["documents"]:
            doc_type = doc.get("document_type", "Other")

            if doc_type not in formatted:
                formatted[doc_type] = []

            formatted[doc_type].append({
                "start_page": doc.get("start_page"),
                "end_page": doc.get("end_page"),
                "confidence": doc.get("confidence", "unknown"),
                "description": doc.get("description", "")
            })

    return formatted


def classify_documents_with_ai(content: str, document_types: List[str] = None, is_markdown: bool = False) -> Dict:
    """
    Use OpenAI to classify document types
    Accepts either pages_data dict or markdown string
    """

    log_event(
        topic="ai_classification",
        sub_topic="classification_start",
        message="Starting document classification",
        status="info"
    )

    base_prompt = """You are a document classification expert. Analyze the following document and identify different document types within it.

Your task:
1. Read through the content and identify distinct document types
2. For multi-page PDFs: Group consecutive pages that belong to the same document and provide page ranges
3. For markdown: Identify sections based on content structure and headings
4. Provide clear boundaries for each document type
5. If you cannot identify a document type, classify it as "Other"

"""

    if document_types:
        types_str = ", ".join(document_types)
        base_prompt += f"\nLook specifically for these document types: {types_str}\n"
    else:
        base_prompt += "\nCommon document types to look for include: closed deal sheet, MESA (Master Equipment Sale Agreement), emails, contracts, invoices, agreements, correspondence, forms, reports, etc.\n"

    if is_markdown:
        base_prompt += """
For markdown content, return your response as a JSON object with this structure:
{
    "documents": [
        {
            "document_type": "name of document type",
            "section_start": "heading or marker where section starts",
            "section_end": "heading or marker where section ends",
            "confidence": "high/medium/low",
            "description": "brief description of what was found"
        }
    ]
}
"""
    else:
        base_prompt += """
For PDF content, return your response as a JSON object with this structure:
{
    "documents": [
        {
            "document_type": "name of document type",
            "start_page": 1,
            "end_page": 3,
            "confidence": "high/medium/low",
            "description": "brief description of what was found"
        }
    ]
}
"""

    base_prompt += f"\n\nHere is the content to analyze:\n\n{content[:15000]}\n"

    try:
        client = AzureOpenAI(
            azure_endpoint=OPENAI_ENDPOINT,
            api_key=OPENAI_API_KEY,
            api_version="2025-01-01-preview"
        )

        response = client.chat.completions.create(
            model=OPENAI_DEPLOYMENT,
            messages=[
                {"role": "system", "content": "You are a document classification expert. Always respond with valid JSON."},
                {"role": "user", "content": base_prompt}
            ],
            temperature=0.3,
            response_format={"type": "json_object"}
        )

        result_text = response.choices[0].message.content
        result_json = json.loads(result_text)

        # Track token usage
        usage = response.usage
        log_event(
            topic="ai_classification",
            sub_topic="classification_complete",
            message="Document classification completed successfully",
            status="success",
            input_tokens=usage.prompt_tokens if usage else None,
            output_tokens=usage.completion_tokens if usage else None,
            api_call="chat.completions.create"
        )

        if "documents" not in result_json:
            result_json = {"documents": []}

        return result_json

    except Exception as e:
        log_event(
            topic="ai_classification",
            sub_topic="classification_error",
            message="Error in AI classification",
            status="error",
            error_details=str(e),
            api_call="chat.completions.create"
        )
        raise


def extract_key_value_pairs_with_ai(content: str, file_name: str = None) -> Dict:
    """
    Extract all key-value pairs from document including tables using LLM
    Content-aware extraction that understands document context
    """

    log_event(
        topic="key_value_extraction",
        sub_topic="extraction_start",
        message="Starting key-value pair extraction",
        status="info",
        file_name=file_name
    )

    system_prompt = """You are an expert document data extraction specialist. Extract ALL key-value pairs from the document in a structured, intelligent manner.

EXTRACTION RULES:
1. **Form Fields**: Extract all labeled fields (e.g., "Name: John Doe" → {"Name": "John Doe"})
2. **Tables**: Convert tables to nested structures with row/column relationships
3. **Sections**: Group related data under section headers as nested objects
4. **Lists**: Extract bulleted/numbered lists as arrays
5. **Metadata**: Extract dates, reference numbers, document IDs, signatures
6. **Smart Context**: Use document context to infer field meanings and relationships
7. **Data Types**: Preserve appropriate types (strings, numbers, dates, booleans, arrays, objects)

TABLE HANDLING:
- For simple tables: Create array of objects with column headers as keys
- For complex tables: Create nested structure preserving hierarchy
- For financial tables: Preserve numerical precision and currency

OUTPUT FORMAT:
Return a JSON object with this structure:
{
  "document_metadata": {
    "document_type": "string",
    "document_id": "string or null",
    "date": "string or null",
    "total_sections": number
  },
  "extracted_data": {
    "section_name": {
      "field_name": "value",
      "nested_field": {...},
      "table_name": [
        {"column1": "value", "column2": "value"},
        ...
      ]
    },
    ...
  },
  "tables": [
    {
      "table_name": "string",
      "headers": ["col1", "col2", ...],
      "rows": [
        {"col1": "val1", "col2": "val2"},
        ...
      ]
    }
  ],
  "key_entities": {
    "parties": ["entity1", "entity2"],
    "amounts": [{"description": "string", "value": number, "currency": "string"}],
    "dates": [{"label": "string", "date": "string"}]
  }
}

IMPORTANT:
- Extract EVERYTHING - no data should be left behind
- Be smart about grouping related information
- Preserve document structure and hierarchy
- Use clear, descriptive key names
- Handle missing values as null
- For ambiguous data, use best judgment based on context"""

    user_prompt = f"""Extract all key-value pairs from the following document. Be thorough and context-aware.

DOCUMENT CONTENT:
{content}

Return comprehensive extraction in the specified JSON format."""

    try:
        client = AzureOpenAI(
            azure_endpoint=OPENAI_ENDPOINT,
            api_key=OPENAI_API_KEY,
            api_version="2025-01-01-preview"
        )

        response = client.chat.completions.create(
            model=OPENAI_DEPLOYMENT,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.1,
            response_format={"type": "json_object"}
        )

        result_text = response.choices[0].message.content
        result_json = json.loads(result_text)

        usage = response.usage
        log_event(
            topic="key_value_extraction",
            sub_topic="extraction_complete",
            message="Successfully extracted key-value pairs",
            status="success",
            file_name=file_name,
            input_tokens=usage.prompt_tokens if usage else None,
            output_tokens=usage.completion_tokens if usage else None,
            api_call="chat.completions.create"
        )

        return result_json

    except Exception as e:
        log_event(
            topic="key_value_extraction",
            sub_topic="extraction_error",
            message="Error extracting key-value pairs",
            status="error",
            file_name=file_name,
            error_details=str(e),
            api_call="chat.completions.create"
        )
        return {"error": str(e), "extracted_data": {}}

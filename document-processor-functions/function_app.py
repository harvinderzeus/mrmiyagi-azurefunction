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


# ============= CONFIGURATION =============
FORM_RECOGNIZER_ENDPOINT = os.environ.get("FORM_RECOGNIZER_ENDPOINT", "")
FORM_RECOGNIZER_KEY = os.environ.get("FORM_RECOGNIZER_KEY", "")
OPENAI_ENDPOINT = os.environ.get("OPENAI_ENDPOINT", "")
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")
OPENAI_DEPLOYMENT = os.environ.get("OPENAI_DEPLOYMENT", "gpt-4o")

# Azure Blob Storage Configuration
STORAGE_ACCOUNT_NAME = os.environ.get("STORAGE_ACCOUNT_NAME", "sadevmrmiyagi")
STORAGE_ACCOUNT_KEY = os.environ.get("STORAGE_ACCOUNT_KEY", "")
STORAGE_CONTAINER_NAME = os.environ.get(
    "STORAGE_CONTAINER_NAME", "dev-mr-miyagi")

app = func.FunctionApp()


# ============= HELPER FUNCTIONS =============

def read_pdf_from_url(pdf_url):
    """Download PDF from URL (e.g., Azure Blob Storage)"""
    try:
        response = requests.get(pdf_url)
        response.raise_for_status()
        logging.info(f"Successfully downloaded PDF from {pdf_url}")
        return response.content
    except Exception as e:
        logging.error(f"Error downloading PDF: {str(e)}")
        raise


def read_pdf_from_blob_storage(blob_name):
    """Read PDF from Azure Blob Storage"""
    try:
        connection_string = f"DefaultEndpointsProtocol=https;AccountName={STORAGE_ACCOUNT_NAME};AccountKey={STORAGE_ACCOUNT_KEY};EndpointSuffix=core.windows.net"

        blob_service_client = BlobServiceClient.from_connection_string(
            connection_string)
        blob_client = blob_service_client.get_blob_client(
            container=STORAGE_CONTAINER_NAME,
            blob=blob_name
        )

        pdf_bytes = blob_client.download_blob().readall()
        logging.info(f"Successfully read PDF from blob storage: {blob_name}")
        return pdf_bytes
    except Exception as e:
        logging.error(f"Error reading PDF from blob storage: {str(e)}")
        raise


def read_pdf_from_local_storage(storage_path):
    """Read PDF from local storage path (fallback)"""
    try:
        with open(storage_path, 'rb') as f:
            pdf_bytes = f.read()
        logging.info(f"Successfully read PDF from {storage_path}")
        return pdf_bytes
    except Exception as e:
        logging.error(f"Error reading PDF: {str(e)}")
        raise


def extract_text_with_document_intelligence(pdf_bytes):
    """Extract text from PDF using Azure Document Intelligence"""
    try:
        client = DocumentAnalysisClient(
            endpoint=FORM_RECOGNIZER_ENDPOINT,
            credential=AzureKeyCredential(FORM_RECOGNIZER_KEY)
        )

        poller = client.begin_analyze_document("prebuilt-layout", pdf_bytes)
        result = poller.result()

        pages_text = {}

        for page in result.pages:
            page_num = page.page_number
            page_text = []

            if hasattr(page, 'lines'):
                for line in page.lines:
                    page_text.append(line.content)

            pages_text[page_num] = "\n".join(page_text)

        logging.info(f"Extracted text from {len(pages_text)} pages")
        return pages_text

    except Exception as e:
        logging.error(f"Error in document intelligence extraction: {str(e)}")
        raise


def classify_documents_with_ai(pages_text, document_types=None):
    """Use OpenAI to classify document types"""

    base_prompt = """You are a document classification expert. Analyze the following multi-page document and identify different document types within it.

Your task:
1. Read through all pages and identify distinct document types
2. Group consecutive pages that belong to the same document
3. Provide page ranges for each document type
4. If you cannot identify a document type, classify it as "Other"

"""

    if document_types:
        types_str = ", ".join(document_types)
        base_prompt += f"\nLook specifically for these document types: {types_str}\n"
    else:
        base_prompt += "\nCommon document types to look for include: closed deal sheet, MESA (Master Equipment Sale Agreement), emails, contracts, invoices, agreements, correspondence, forms, reports, etc.\n"

    base_prompt += """
Return your response as a JSON object with this exact structure:
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

Here are the pages to analyze:

"""

    for page_num in sorted(pages_text.keys()):
        text = pages_text[page_num]
        base_prompt += f"\n--- PAGE {page_num} ---\n{text[:2000]}\n"

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

        logging.info("Classification completed successfully")
        return result_json

    except Exception as e:
        logging.error(f"Error in AI classification: {str(e)}")
        raise


def format_classification_results(classification_result):
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


# ============= HTTP TRIGGER FUNCTIONS =============

@app.route(route="classify", auth_level=func.AuthLevel.ANONYMOUS, methods=["POST"])
def classify_document(req: func.HttpRequest) -> func.HttpResponse:
    """Classify documents in a PDF"""

    logging.info("ClassifyDocument HTTP trigger called")

    try:
        req_body = req.get_json()

        pdf_source = req_body.get('pdf_path') or req_body.get('pdf_url')
        document_types = req_body.get('document_types')

        if not pdf_source:
            return func.HttpResponse(
                json.dumps(
                    {"error": "Missing 'pdf_path' or 'pdf_url' in request body"}),
                status_code=400,
                mimetype="application/json"
            )

        # Read PDF from appropriate source
        if pdf_source.startswith('http://') or pdf_source.startswith('https://'):
            # URL (direct link)
            pdf_bytes = read_pdf_from_url(pdf_source)
        elif pdf_source.startswith('blob://'):
            # Azure Blob Storage reference (e.g., blob://folder/file.pdf)
            blob_name = pdf_source.replace('blob://', '')
            pdf_bytes = read_pdf_from_blob_storage(blob_name)
        else:
            # Local file path (fallback)
            pdf_bytes = read_pdf_from_local_storage(pdf_source)

        # Extract text
        pages_text = extract_text_with_document_intelligence(pdf_bytes)

        # Classify
        classification_result = classify_documents_with_ai(
            pages_text, document_types)

        # Format results
        formatted_results = format_classification_results(
            classification_result)

        return func.HttpResponse(
            json.dumps({
                "status": "success",
                "total_pages": len(pages_text),
                "results": formatted_results
            }),
            status_code=200,
            mimetype="application/json"
        )

    except ValueError as e:
        logging.error(f"Invalid JSON: {str(e)}")
        return func.HttpResponse(
            json.dumps({"error": "Invalid JSON in request body"}),
            status_code=400,
            mimetype="application/json"
        )

    except Exception as e:
        logging.error(f"Error: {str(e)}")
        return func.HttpResponse(
            json.dumps({"error": f"Processing error: {str(e)}"}),
            status_code=500,
            mimetype="application/json"
        )


@app.route(route="extract", auth_level=func.AuthLevel.ANONYMOUS, methods=["POST"])
def extract_text(req: func.HttpRequest) -> func.HttpResponse:
    """Extract text from PDF"""

    logging.info("ExtractText HTTP trigger called")

    try:
        req_body = req.get_json()

        pdf_source = req_body.get('pdf_path') or req_body.get('pdf_url')

        if not pdf_source:
            return func.HttpResponse(
                json.dumps(
                    {"error": "Missing 'pdf_path' or 'pdf_url' in request body"}),
                status_code=400,
                mimetype="application/json"
            )

        # Read PDF from appropriate source
        if pdf_source.startswith('http://') or pdf_source.startswith('https://'):
            # URL (direct link)
            pdf_bytes = read_pdf_from_url(pdf_source)
        elif pdf_source.startswith('blob://'):
            # Azure Blob Storage reference (e.g., blob://folder/file.pdf)
            blob_name = pdf_source.replace('blob://', '')
            pdf_bytes = read_pdf_from_blob_storage(blob_name)
        else:
            # Local file path (fallback)
            pdf_bytes = read_pdf_from_local_storage(pdf_source)

        # Extract text
        pages_text = extract_text_with_document_intelligence(pdf_bytes)

        return func.HttpResponse(
            json.dumps({
                "status": "success",
                "total_pages": len(pages_text),
                "pages": pages_text
            }),
            status_code=200,
            mimetype="application/json"
        )

    except Exception as e:
        logging.error(f"Error: {str(e)}")
        return func.HttpResponse(
            json.dumps({"error": f"Processing error: {str(e)}"}),
            status_code=500,
            mimetype="application/json"
        )


@app.route(route="batch", auth_level=func.AuthLevel.ANONYMOUS, methods=["POST"])
def batch_process(req: func.HttpRequest) -> func.HttpResponse:
    """Process multiple PDFs"""

    logging.info("BatchProcess HTTP trigger called")

    try:
        req_body = req.get_json()

        pdf_sources = req_body.get('pdf_sources', [])
        document_types = req_body.get('document_types')

        if not pdf_sources:
            return func.HttpResponse(
                json.dumps(
                    {"error": "Missing 'pdf_sources' array in request body"}),
                status_code=400,
                mimetype="application/json"
            )

        results = []

        for idx, pdf_source in enumerate(pdf_sources):
            try:
                logging.info(f"Processing PDF {idx+1}/{len(pdf_sources)}")

                # Read PDF
                if pdf_source.startswith('http://') or pdf_source.startswith('https://'):
                    pdf_bytes = read_pdf_from_url(pdf_source)
                else:
                    pdf_bytes = read_pdf_from_storage(pdf_source)

                # Extract text
                pages_text = extract_text_with_document_intelligence(pdf_bytes)

                # Classify
                classification_result = classify_documents_with_ai(
                    pages_text, document_types)

                # Format results
                formatted_results = format_classification_results(
                    classification_result)

                results.append({
                    "pdf_source": pdf_source,
                    "status": "success",
                    "total_pages": len(pages_text),
                    "classification": formatted_results
                })

            except Exception as e:
                logging.error(f"Error processing {pdf_source}: {str(e)}")
                results.append({
                    "pdf_source": pdf_source,
                    "status": "failed",
                    "error": str(e)
                })

        return func.HttpResponse(
            json.dumps({
                "status": "completed",
                "total_processed": len(pdf_sources),
                "results": results
            }),
            status_code=200,
            mimetype="application/json"
        )

    except Exception as e:
        logging.error(f"Error: {str(e)}")
        return func.HttpResponse(
            json.dumps({"error": f"Processing error: {str(e)}"}),
            status_code=500,
            mimetype="application/json"
        )


@app.route(route="split", auth_level=func.AuthLevel.ANONYMOUS, methods=["POST"])
def split_document(req: func.HttpRequest) -> func.HttpResponse:
    """Split a PDF into multiple smaller PDFs based on page ranges and document types"""

    logging.info("SplitDocument HTTP trigger called")

    try:
        req_body = req.get_json()

        # Accept both pdf_path and pdf_url like other functions
        pdf_source = req_body.get('pdf_path') or req_body.get('pdf_url')
        split_info = req_body.get('split_info')

        if not pdf_source or not split_info:
            return func.HttpResponse(
                json.dumps(
                    {"error": "Missing 'pdf_path' or 'pdf_url' and 'split_info' in request body"}),
                status_code=400,
                mimetype="application/json"
            )

        # --- Read PDF bytes from blob, URL, or local storage ---
        if pdf_source.startswith('http://') or pdf_source.startswith('https://'):
            pdf_bytes = read_pdf_from_url(pdf_source)
        elif pdf_source.startswith('blob://'):
            blob_name = pdf_source.replace('blob://', '')
            pdf_bytes = read_pdf_from_blob_storage(blob_name)
        else:
            pdf_bytes = read_pdf_from_local_storage(pdf_source)

        # --- Load original PDF ---
        src_doc = fitz.open(stream=pdf_bytes, filetype="pdf")

        connection_string = f"DefaultEndpointsProtocol=https;AccountName={STORAGE_ACCOUNT_NAME};AccountKey={STORAGE_ACCOUNT_KEY};EndpointSuffix=core.windows.net"
        blob_service_client = BlobServiceClient.from_connection_string(
            connection_string)
        container_client = blob_service_client.get_container_client(
            STORAGE_CONTAINER_NAME)

        new_files = []

        # --- For each document type ---
        for doc_type, sections in split_info.items():
            for i, section in enumerate(sections, start=1):
                start_page = section.get("start_page")
                end_page = section.get("end_page")

                if start_page is None or end_page is None:
                    continue

                # Create new PDF
                new_pdf = fitz.open()
                for pno in range(start_page - 1, end_page):  # PyMuPDF is 0-indexed
                    new_pdf.insert_pdf(src_doc, from_page=pno, to_page=pno)

                # Generate new filename
                # Extract base name from different source types
                if pdf_source.startswith('blob://'):
                    base_name = os.path.basename(
                        pdf_source.replace('blob://', '')).replace(".pdf", "")
                elif pdf_source.startswith('http://') or pdf_source.startswith('https://'):
                    base_name = os.path.basename(
                        pdf_source.split('?')[0]).replace(".pdf", "")
                else:
                    base_name = os.path.basename(
                        pdf_source).replace(".pdf", "")

                safe_type = doc_type.replace(" ", "_").replace("/", "_")
                new_filename = f"{base_name}-{safe_type}-{i}.pdf"

                # Save to memory
                pdf_bytes_out = new_pdf.tobytes()
                blob_client = container_client.get_blob_client(new_filename)
                blob_client.upload_blob(pdf_bytes_out, overwrite=True)

                blob_url = f"https://{STORAGE_ACCOUNT_NAME}.blob.core.windows.net/{STORAGE_CONTAINER_NAME}/{new_filename}"
                new_files.append({
                    "document_type": doc_type,
                    "pages": f"{start_page}-{end_page}",
                    "blob_name": new_filename,
                    "blob_url": blob_url
                })

                new_pdf.close()

        src_doc.close()

        return func.HttpResponse(
            json.dumps({
                "status": "success",
                "message": f"Split into {len(new_files)} files successfully.",
                "files": new_files
            }),
            status_code=200,
            mimetype="application/json"
        )

    except Exception as e:
        logging.error(f"Error in split_document: {str(e)}")
        return func.HttpResponse(
            json.dumps({"error": f"Processing error: {str(e)}"}),
            status_code=500,
            mimetype="application/json"
        )

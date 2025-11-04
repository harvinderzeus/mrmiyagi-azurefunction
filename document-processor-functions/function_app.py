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

app = func.FunctionApp()


# ============= HELPER FUNCTIONS =============

def get_blob_client(blob_name: str):
    """Get blob client for a specific blob"""
    connection_string = f"DefaultEndpointsProtocol=https;AccountName={STORAGE_ACCOUNT_NAME};AccountKey={STORAGE_ACCOUNT_KEY};EndpointSuffix=core.windows.net"
    blob_service_client = BlobServiceClient.from_connection_string(
        connection_string)
    return blob_service_client.get_blob_client(container=STORAGE_CONTAINER_NAME, blob=blob_name)


def read_pdf_from_blob(file_path: str) -> bytes:
    """
    Read PDF from blob storage
    Expected format: dev-mr-miyagi/files/file1.pdf
    """
    try:
        # Remove container name prefix if present
        if file_path.startswith(f"{STORAGE_CONTAINER_NAME}/"):
            blob_name = file_path[len(STORAGE_CONTAINER_NAME)+1:]
        else:
            blob_name = file_path

        blob_client = get_blob_client(blob_name)
        pdf_bytes = blob_client.download_blob().readall()
        logging.info(f"Successfully read PDF from blob: {blob_name}")
        return pdf_bytes
    except Exception as e:
        logging.error(f"Error reading PDF from blob storage: {str(e)}")
        raise


def extract_text_with_document_intelligence(pdf_bytes: bytes) -> Dict[int, Any]:
    """Extract text from PDF using Azure Document Intelligence"""
    try:
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

        logging.info(f"Extracted text from {len(pages_data)} pages")
        return pages_data

    except Exception as e:
        logging.error(f"Error in document intelligence extraction: {str(e)}")
        raise


def save_extracted_json(file_path: str, extraction_data: Dict) -> str:
    """
    Save extraction data to blob storage
    Input: dev-mr-miyagi/files/file1.pdf
    Output: dev-mr-miyagi/outcome/file1/file1-extracted.json
    """
    try:
        # Extract filename without extension
        if file_path.startswith(f"{STORAGE_CONTAINER_NAME}/"):
            file_path = file_path[len(STORAGE_CONTAINER_NAME)+1:]

        filename = os.path.basename(file_path).replace('.pdf', '')

        # Create output path
        output_blob_name = f"outcome/{filename}/{filename}-extracted.json"

        # Upload JSON
        blob_client = get_blob_client(output_blob_name)
        json_data = json.dumps(extraction_data, indent=2)
        blob_client.upload_blob(json_data, overwrite=True)

        output_path = f"{STORAGE_CONTAINER_NAME}/{output_blob_name}"
        logging.info(f"Saved extracted JSON to: {output_path}")
        return output_path

    except Exception as e:
        logging.error(f"Error saving extracted JSON: {str(e)}")
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


def extract_items_from_pages(pages_data: Dict[int, Any], page_range: tuple, items_config: List[Dict]) -> Dict:
    """
    Extract specific items from pages using AI
    
    Args:
        pages_data: Dictionary of page data
        page_range: Tuple of (start_page, end_page)
        items_config: List of dictionaries with format:
            [
                {"search_key": "broker name", "variable_name": "broker", "type": "single"},
                {"search_key": "customer", "variable_name": "customer", "type": "single"},
                {"search_key": "price", "variable_name": "prices", "type": "list"}
            ]
    
    Returns:
        Dictionary with extracted items using variable_name as keys
    """
    start_page, end_page = page_range

    if not items_config:
        return {}

    # Build the system prompt
    system_prompt = """You are a precise data extraction specialist.
Extract the requested items from the document pages.

IMPORTANT RULES:
- For 'single' type items: Extract only ONE value (the most prominent/relevant one across all pages)
- For 'list' type items: Extract ALL occurrences found across all pages as an array
- Return null for items not found
- Extract exact values as they appear in the document
- Do not invent or infer values

ONLY return a valid JSON object with this structure:
{
  "extracted_items": {
    "variable_name": "value for single items" OR ["value1", "value2"] for list items,
    ...
  }
}"""

    # Build items description for the prompt
    items_desc = "\n".join([
        f"- Search for '{item['search_key']}' and store as '{item['variable_name']}' (type: {item['type']})"
        for item in items_config
    ])

    # Collect text from all pages in range
    pages_text = ""
    for page_num in range(start_page, end_page + 1):
        if page_num in pages_data:
            pages_text += f"\n--- PAGE {page_num} ---\n{pages_data[page_num]['full_text']}\n"

    user_prompt = f"""Extract the following items from the document pages:

{items_desc}

Document text:
{pages_text}

Return the extracted items in JSON format using the variable names as keys."""

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

        extracted_items = result_json.get("extracted_items", {})
        logging.info(f"Successfully extracted {len(extracted_items)} items")

        return extracted_items

    except Exception as e:
        logging.error(f"Error extracting items: {str(e)}")
        # Return empty dict with null values for all requested items
        return {item['variable_name']: None for item in items_config}


def identify_headings_and_subheadings(pages_data: Dict[int, Any], document_type: str, page_range: tuple, items_config: List[Dict] = None) -> List[Dict]:
    """
    Use AI to identify headings and subheadings for pages in a document section
    Optionally extract specific items and add as metadata to each chunk
    """
    start_page, end_page = page_range

    system_prompt = """You are a precise document-structure analyst.
ONLY return a single JSON object with this exact schema:
{
  "page": <int>,
  "headings": ["<string>", "..."],
  "subheadings": ["<string>", "..."]
}
Guidance:
- Consider that true headings are top-level section titles on THIS page (e.g., document title, ARTICLE/SECTION headers, bold/all-caps prominent lines, or numbered section starts like '1.' / 'Section 1').
- Consider subheadings as secondary titles under those headings on THIS page (e.g., 1.1, Exhibit labels, clause titles under a section).
- Ignore body paragraphs, boilerplate, and long sentences.
- Do not infer content from other pages.
- If none found, return empty arrays.
Return VALID JSON ONLY."""

    results = []

    # Extract items once for the entire section if items_config is provided
    metadata = {}
    if items_config:
        extracted_items = extract_items_from_pages(
            pages_data, page_range, items_config)
        metadata = extracted_items
        logging.info(
            f"Extracted metadata for section: {list(metadata.keys())}")

    try:
        client = AzureOpenAI(
            azure_endpoint=OPENAI_ENDPOINT,
            api_key=OPENAI_API_KEY,
            api_version="2025-01-01-preview"
        )

        # Process each page individually
        for page_num in range(start_page, end_page + 1):
            if page_num not in pages_data:
                results.append({
                    "page_number": page_num,
                    "heading": document_type,
                    "subheading": None,
                    "metadata": metadata  # Add metadata to every chunk
                })
                continue

            page_text = pages_data[page_num]["full_text"]

            user_prompt = f"""PAGE {page_num} TEXT (verbatim from OCR):
{page_text}

Extract headings and subheadings for THIS page only."""

            try:
                response = client.chat.completions.create(
                    model=OPENAI_DEPLOYMENT,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt}
                    ],
                    temperature=0.2,
                    response_format={"type": "json_object"}
                )

                result_text = response.choices[0].message.content
                result_json = json.loads(result_text)

                # Convert the response format to match your existing structure
                headings = result_json.get("headings", [])
                subheadings = result_json.get("subheadings", [])

                results.append({
                    "page_number": page_num,
                    "heading": headings[0] if headings else document_type,
                    "subheading": subheadings[0] if subheadings else None,
                    "all_headings": headings,
                    "all_subheadings": subheadings,
                    "metadata": metadata  # Add metadata to every chunk
                })

            except Exception as e:
                logging.error(f"Error processing page {page_num}: {str(e)}")
                results.append({
                    "page_number": page_num,
                    "heading": document_type,
                    "subheading": None,
                    "metadata": metadata  # Add metadata even on error
                })

        return results

    except Exception as e:
        logging.error(f"Error identifying headings: {str(e)}")
        # Fallback: use document type as heading
        return [
            {
                "page_number": page_num,
                "heading": document_type,
                "subheading": None,
                "metadata": metadata
            }
            for page_num in range(start_page, end_page + 1)
        ]


def classify_documents_with_ai(pages_data: Dict[int, Any], document_types: List[str] = None) -> Dict:
    """Use OpenAI to classify document types with enhanced error handling"""

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

    for page_num in sorted(pages_data.keys()):
        text = pages_data[page_num]["full_text"]
        base_prompt += f"\n--- PAGE {page_num} ---\n{text[:2000]}\n"

    try:
        client = AzureOpenAI(
            azure_endpoint=OPENAI_ENDPOINT,
            api_key=OPENAI_API_KEY,
            api_version="2025-01-01-preview"
        )

        logging.info("Sending classification request to OpenAI")
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
        logging.info(f"Received OpenAI response: {result_text[:200]}...")

        try:
            result_json = json.loads(result_text)
        except json.JSONDecodeError as e:
            logging.error(f"Failed to parse OpenAI response as JSON: {str(e)}")
            logging.error(f"Response text: {result_text}")
            # Return a default structure
            return {
                "documents": [{
                    "document_type": "Other",
                    "start_page": 1,
                    "end_page": len(pages_data),
                    "confidence": "low",
                    "description": "Classification failed, defaulting to single document"
                }]
            }

        # Validate structure
        if "documents" not in result_json:
            logging.warning("OpenAI response missing 'documents' key")
            result_json = {"documents": []}

        logging.info("Classification completed successfully")
        return result_json

    except Exception as e:
        logging.error(f"Error in AI classification: {str(e)}", exc_info=True)
        raise


# ============= HTTP TRIGGER FUNCTIONS =============

@app.route(route="classify", auth_level=func.AuthLevel.ANONYMOUS, methods=["POST"])
def classify_document(req: func.HttpRequest) -> func.HttpResponse:
    """
    API 1 - Classify
    Input: File path in format dev-mr-miyagi/files/file1.pdf
    Output: Classification result + path to extracted JSON
    """
    logging.info("API 1: Classify - HTTP trigger called")

    try:
        # Validate environment variables first
        required_vars = {
            "FORM_RECOGNIZER_ENDPOINT": FORM_RECOGNIZER_ENDPOINT,
            "FORM_RECOGNIZER_KEY": FORM_RECOGNIZER_KEY,
            "OPENAI_ENDPOINT": OPENAI_ENDPOINT,
            "OPENAI_API_KEY": OPENAI_API_KEY,
            "STORAGE_ACCOUNT_KEY": STORAGE_ACCOUNT_KEY
        }

        missing_vars = [name for name,
                        value in required_vars.items() if not value]
        if missing_vars:
            error_msg = f"Missing required environment variables: {', '.join(missing_vars)}"
            logging.error(error_msg)
            return func.HttpResponse(
                json.dumps({"error": error_msg}),
                status_code=500,
                mimetype="application/json"
            )

        # Parse request body with error handling
        try:
            req_body = req.get_json()
        except ValueError as e:
            logging.error(f"Invalid JSON in request: {str(e)}")
            return func.HttpResponse(
                json.dumps({"error": "Invalid JSON in request body"}),
                status_code=400,
                mimetype="application/json"
            )

        file_path = req_body.get('file_path')
        document_types = req_body.get('document_types')

        if not file_path:
            return func.HttpResponse(
                json.dumps({"error": "Missing 'file_path' in request body"}),
                status_code=400,
                mimetype="application/json"
            )

        logging.info(f"Processing file: {file_path}")

        # Read PDF from blob storage
        try:
            pdf_bytes = read_pdf_from_blob(file_path)
            logging.info(
                f"PDF read successfully, size: {len(pdf_bytes)} bytes")
        except Exception as e:
            logging.error(f"Failed to read PDF from blob: {str(e)}")
            return func.HttpResponse(
                json.dumps({"error": f"Failed to read PDF: {str(e)}"}),
                status_code=404,
                mimetype="application/json"
            )

        # Extract text with Document Intelligence
        try:
            pages_data = extract_text_with_document_intelligence(pdf_bytes)
            logging.info(f"Extracted {len(pages_data)} pages")
        except Exception as e:
            logging.error(f"Document Intelligence failed: {str(e)}")
            return func.HttpResponse(
                json.dumps({"error": f"Text extraction failed: {str(e)}"}),
                status_code=500,
                mimetype="application/json"
            )

        # Save extracted JSON
        try:
            extracted_json_path = save_extracted_json(file_path, pages_data)
            logging.info(f"Saved extraction to: {extracted_json_path}")
        except Exception as e:
            logging.error(f"Failed to save extracted JSON: {str(e)}")
            return func.HttpResponse(
                json.dumps({"error": f"Failed to save extraction: {str(e)}"}),
                status_code=500,
                mimetype="application/json"
            )

        # Classify documents
        try:
            classification_result = classify_documents_with_ai(
                pages_data, document_types)
            logging.info("Classification completed")
        except Exception as e:
            logging.error(f"Classification failed: {str(e)}")
            return func.HttpResponse(
                json.dumps({"error": f"Classification failed: {str(e)}"}),
                status_code=500,
                mimetype="application/json"
            )

        # Format results
        formatted_results = format_classification_results(
            classification_result)

        response_data = {
            "status": "success",
            "file_path": file_path,
            "total_pages": len(pages_data),
            "classification": formatted_results,
            "extracted_json_path": extracted_json_path
        }

        logging.info("Classification successful")
        return func.HttpResponse(
            json.dumps(response_data, indent=2),
            status_code=200,
            mimetype="application/json"
        )

    except Exception as e:
        # Catch-all for unexpected errors
        logging.error(
            f"Unexpected error in classify_document: {str(e)}", exc_info=True)
        return func.HttpResponse(
            json.dumps({"error": f"Internal server error: {str(e)}"}),
            status_code=500,
            mimetype="application/json"
        )


@app.route(route="split", auth_level=func.AuthLevel.ANONYMOUS, methods=["POST"])
def split_document(req: func.HttpRequest) -> func.HttpResponse:
    """
    API 2 - Split
    Input: File path + classification response
    Output: Split file paths + metadata
    """
    logging.info("API 2: Split - HTTP trigger called")

    try:
        req_body = req.get_json()
        file_path = req_body.get('file_path')
        classification = req_body.get('classification')

        if not file_path or not classification:
            return func.HttpResponse(
                json.dumps(
                    {"error": "Missing 'file_path' or 'classification' in request body"}),
                status_code=400,
                mimetype="application/json"
            )

        # Read PDF from blob storage
        pdf_bytes = read_pdf_from_blob(file_path)
        src_doc = fitz.open(stream=pdf_bytes, filetype="pdf")

        # Extract base filename
        if file_path.startswith(f"{STORAGE_CONTAINER_NAME}/"):
            file_path_clean = file_path[len(STORAGE_CONTAINER_NAME)+1:]
        else:
            file_path_clean = file_path

        base_filename = os.path.basename(file_path_clean).replace('.pdf', '')

        split_results = []

        # Process each document type
        for doc_type, sections in classification.items():
            for seq, section in enumerate(sections, start=1):
                start_page = section.get("start_page")
                end_page = section.get("end_page")

                if start_page is None or end_page is None:
                    continue

                # Create new PDF
                new_pdf = fitz.open()
                for page_num in range(start_page - 1, end_page):  # PyMuPDF is 0-indexed
                    new_pdf.insert_pdf(
                        src_doc, from_page=page_num, to_page=page_num)

                # Generate filename: file1-{document_type}-{seq}.pdf
                safe_doc_type = doc_type.replace(" ", "_").replace("/", "_")
                new_filename = f"{base_filename}-{safe_doc_type}-{seq}.pdf"

                # Save to outcome folder: dev-mr-miyagi/outcome/file1/file1-{document_type}-{seq}.pdf
                output_blob_name = f"outcome/{base_filename}/{new_filename}"

                # Upload to blob storage
                pdf_bytes_out = new_pdf.tobytes()
                blob_client = get_blob_client(output_blob_name)
                blob_client.upload_blob(pdf_bytes_out, overwrite=True)

                blob_url = f"https://{STORAGE_ACCOUNT_NAME}.blob.core.windows.net/{STORAGE_CONTAINER_NAME}/{output_blob_name}"

                split_results.append({
                    "document_type": doc_type,
                    "sequence": seq,
                    "start_page": start_page,
                    "end_page": end_page,
                    "page_count": end_page - start_page + 1,
                    "file_path": f"{STORAGE_CONTAINER_NAME}/{output_blob_name}",
                    "blob_url": blob_url,
                    "confidence": section.get("confidence"),
                    "description": section.get("description")
                })

                new_pdf.close()

        src_doc.close()

        return func.HttpResponse(
            json.dumps({
                "status": "success",
                "message": f"Split into {len(split_results)} files",
                "source_file": file_path,
                "files": split_results
            }, indent=2),
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


@app.route(route="structure", auth_level=func.AuthLevel.ANONYMOUS, methods=["POST"])
def analyze_structure(req: func.HttpRequest) -> func.HttpResponse:
    """
    API 3 - Structure (Headings & Subheadings with Metadata)
    Input: 
        - classification: Classification response
        - extracted_json_path: Path to extracted JSON
        - items_to_extract: List of dicts with format:
            [
                {"search_key": "broker name", "variable_name": "broker", "type": "single"},
                {"search_key": "customer", "variable_name": "customer", "type": "single"},
                {"search_key": "price", "variable_name": "prices", "type": "list"}
            ]
    Output: Structured response with document type, headings, subheadings, page numbers, and metadata per chunk
    """
    logging.info("API 3: Structure - HTTP trigger called")

    try:
        req_body = req.get_json()
        classification = req_body.get('classification')
        extracted_json_path = req_body.get('extracted_json_path')
        items_to_extract = req_body.get(
            'items_to_extract')  # New format: list of dicts

        if not classification or not extracted_json_path:
            return func.HttpResponse(
                json.dumps({
                    "error": "Missing 'classification' or 'extracted_json_path' in request body"
                }),
                status_code=400,
                mimetype="application/json"
            )

        # Read extracted JSON from blob
        if extracted_json_path.startswith(f"{STORAGE_CONTAINER_NAME}/"):
            blob_name = extracted_json_path[len(STORAGE_CONTAINER_NAME)+1:]
        else:
            blob_name = extracted_json_path

        blob_client = get_blob_client(blob_name)
        extracted_data_str = blob_client.download_blob().readall().decode('utf-8')
        pages_data = json.loads(extracted_data_str)

        # Convert string keys to int for page numbers
        pages_data = {int(k): v for k, v in pages_data.items()}

        structured_results = []

        # Process each document section
        for doc_type, sections in classification.items():
            for section in sections:
                start_page = section.get("start_page")
                end_page = section.get("end_page")

                if start_page is None or end_page is None:
                    continue

                # Identify headings and subheadings with metadata for this section
                page_structures = identify_headings_and_subheadings(
                    pages_data,
                    doc_type,
                    (start_page, end_page),
                    items_to_extract
                )

                # Build structured output
                for page_struct in page_structures:
                    result_entry = {
                        "document_type": doc_type,
                        "page_number": page_struct.get("page_number"),
                        "heading": page_struct.get("heading", doc_type),
                        "subheading": page_struct.get("subheading"),
                        # Metadata for chunk
                        "metadata": page_struct.get("metadata", {}),
                        "confidence": section.get("confidence"),
                        "description": section.get("description")
                    }

                    structured_results.append(result_entry)

        return func.HttpResponse(
            json.dumps({
                "status": "success",
                "total_pages_analyzed": len(structured_results),
                "structure": structured_results
            }, indent=2),
            status_code=200,
            mimetype="application/json"
        )

    except Exception as e:
        logging.error(f"Error in analyze_structure: {str(e)}")
        return func.HttpResponse(
            json.dumps({"error": f"Processing error: {str(e)}"}),
            status_code=500,
            mimetype="application/json"
        )

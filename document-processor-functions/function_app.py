        import tiktoken
        from typing import List, Dict, Any
        import fitz  # PyMuPDF
        import azure.functions as func
        import json
        import logging
        import os
        from typing import Dict, List, Any, Optional, Tuple
        from datetime import datetime
        from azure.eventhub import EventHubProducerClient, EventData
        from azure.ai.formrecognizer import DocumentAnalysisClient
        from azure.core.credentials import AzureKeyCredential
        from openai import AzureOpenAI
        from azure.storage.blob import BlobServiceClient

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
        EVENT_HUB_CONNECTION_STRING = os.environ.get("EVENT_HUB_CONNECTION_STRING",
                                                    "Endpoint")
        EVENT_HUB_NAME = "dev-mrmiyagi-logs"

        app = func.FunctionApp()

        # ============= LOGGING FUNCTIONS =============


        def log_event(topic: str, sub_topic: str, message: str, status: str = "info",
                    file_name: Optional[str] = None, error_details: Optional[str] = None,
                    total_pages: Optional[int] = None, input_tokens: Optional[int] = None,
                    output_tokens: Optional[int] = None, embedding_tokens: Optional[int] = None,
                    api_call: Optional[str] = None):
            """Log an event to Event Hub with structured data"""

            log_entry = {
                "timestamp": datetime.utcnow().isoformat(),
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
                except Exception as e:
                    logging.error(f"Failed to send to Event Hub: {e}")
                    logging.info(f"Log fallback: {json.dumps(log_entry)}")
            else:
                logging.info(json.dumps(log_entry))

        # ============= HELPER FUNCTIONS =============


        def get_blob_client(blob_name: str):
            """Get blob client for a specific blob"""
            connection_string = f"DefaultEndpointsProtocol=https;AccountName={STORAGE_ACCOUNT_NAME};AccountKey={STORAGE_ACCOUNT_KEY};EndpointSuffix=core.windows.net"
            blob_service_client = BlobServiceClient.from_connection_string(
                connection_string)
            return blob_service_client.get_blob_client(container=STORAGE_CONTAINER_NAME, blob=blob_name)


        def read_file_from_blob(file_path: str) -> bytes:
            """Read file from blob storage (PDF or Markdown)"""
            try:
                if file_path.startswith(f"{STORAGE_CONTAINER_NAME}/"):
                    blob_name = file_path[len(STORAGE_CONTAINER_NAME)+1:]
                else:
                    blob_name = file_path

                blob_client = get_blob_client(blob_name)
                file_bytes = blob_client.download_blob().readall()

                log_event("blob_storage", "read_file", f"Successfully read file from blob: {blob_name}",
                        status="success", file_name=blob_name)
                return file_bytes
            except Exception as e:
                log_event("blob_storage", "read_file", f"Error reading file from blob storage",
                        status="error", file_name=file_path, error_details=str(e))
                raise


        def save_to_blob(content: str, output_path: str, content_type: str = "application/json") -> str:
            """Save content to blob storage"""
            try:
                if output_path.startswith(f"{STORAGE_CONTAINER_NAME}/"):
                    blob_name = output_path[len(STORAGE_CONTAINER_NAME)+1:]
                else:
                    blob_name = output_path

                blob_client = get_blob_client(blob_name)
                blob_client.upload_blob(content, overwrite=True, content_settings={
                                        'content_type': content_type})

                full_path = f"{STORAGE_CONTAINER_NAME}/{blob_name}"
                log_event("blob_storage", "save_file", f"Successfully saved file to blob: {blob_name}",
                        status="success", file_name=blob_name)
                return full_path
            except Exception as e:
                log_event("blob_storage", "save_file", f"Error saving file to blob storage",
                        status="error", file_name=output_path, error_details=str(e))
                raise


        def extract_text_with_document_intelligence(pdf_bytes: bytes, output_format: str = "json") -> Dict[int, Any]:
            """Extract text from PDF using Azure Document Intelligence"""
            try:
                client = DocumentAnalysisClient(
                    endpoint=FORM_RECOGNIZER_ENDPOINT,
                    credential=AzureKeyCredential(FORM_RECOGNIZER_KEY)
                )

                log_event("document_intelligence", "extract_text", "Starting text extraction",
                        status="info", api_call="begin_analyze_document")

                if output_format == "markdown":
                    poller = client.begin_analyze_document(
                        model_id="prebuilt-layout",
                        document=pdf_bytes,
                        # content_type="application/pdf",
                        output_content_format="markdown"
                    )
                else:
                    poller = client.begin_analyze_document(
                        "prebuilt-layout", document=pdf_bytes)

                result = poller.result()

                if output_format == "markdown":
                    # Extract markdown content
                    markdown_content = result.content if hasattr(
                        result, 'content') else ""
                    log_event("document_intelligence", "extract_text",
                            f"Extracted markdown content ({len(markdown_content)} chars)",
                            status="success", api_call="begin_analyze_document")
                    return {"markdown": markdown_content}

                # Store full page information including layout details (JSON format)
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

                    page_info["full_text"] = "\n".join(page_info["text"])
                    pages_data[page_num] = page_info

                log_event("document_intelligence", "extract_text",
                        f"Successfully extracted {len(pages_data)} pages",
                        status="success", total_pages=len(pages_data), api_call="begin_analyze_document")
                return pages_data

            except Exception as e:
                log_event("document_intelligence", "extract_text", "Text extraction failed",
                        status="error", error_details=str(e), api_call="begin_analyze_document")
                raise


        def classify_documents_with_ai(content: str, content_type: str, document_types: List[str] = None) -> Dict:
            """Classify documents using OpenAI (supports both JSON pages data and markdown)"""

            base_prompt = """You are a document classification expert. Analyze the following document and identify different document types within it.

        Your task:
        1. Read through the content and identify distinct document types
        2. Group consecutive pages/sections that belong to the same document
        3. Provide page ranges for each document type
        4. If you cannot identify a document type, classify it as "Other"

        """

            if document_types:
                types_str = ", ".join(document_types)
                base_prompt += f"\nLook specifically for these document types: {types_str}\n"
            else:
                base_prompt += "\nCommon document types: closed deal sheet, MESA, emails, contracts, invoices, agreements, correspondence, forms, reports, etc.\n"

            base_prompt += """
        Return your response as a JSON object with this exact structure:
        {
            "documents": [
                {
                    "document_type": "name of document type",
                    "start_page": 1,
                    "end_page": 3,
                    "confidence": "high/medium/low",
                    "description": "brief description"
                }
            ]
        }

        Content to analyze:
        """

            if content_type == "json":
                pages_data = json.loads(content) if isinstance(
                    content, str) else content
                for page_num in sorted(pages_data.keys()):
                    text = pages_data[page_num]["full_text"]
                    base_prompt += f"\n--- PAGE {page_num} ---\n{text[:2000]}\n"
            else:  # markdown
                base_prompt += f"\n{content}\n"

            try:
                client = AzureOpenAI(
                    azure_endpoint=OPENAI_ENDPOINT,
                    api_key=OPENAI_API_KEY,
                    api_version="2024-08-01-preview"
                )

                log_event("openai", "classify_documents", "Sending classification request",
                        status="info", api_call="chat.completions.create")

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

                # Log token usage
                usage = response.usage
                log_event("openai", "classify_documents", "Classification completed successfully",
                        status="success", input_tokens=usage.prompt_tokens,
                        output_tokens=usage.completion_tokens, api_call="chat.completions.create")

                if "documents" not in result_json:
                    result_json = {"documents": []}

                return result_json

            except Exception as e:
                log_event("openai", "classify_documents", "Classification failed",
                        status="error", error_details=str(e), api_call="chat.completions.create")
                raise


        def extract_items_with_ai(content: str, content_type: str, page_range: Tuple[int, int],
                                items_config: List[Dict]) -> Dict:
            """Extract specific items using AI (supports both JSON and markdown)"""

            if not items_config:
                return {}

            system_prompt = """You are a precise data extraction specialist.
        Extract the requested items from the document content.

        IMPORTANT RULES:
        - For 'single' type items: Extract only ONE value (the most prominent/relevant one)
        - For 'list' type items: Extract ALL occurrences found as an array
        - Return null for items not found
        - Extract exact values as they appear
        - Do not invent or infer values

        ONLY return valid JSON:
        {
        "extracted_items": {
            "variable_name": "value for single items" OR ["value1", "value2"] for list items
        }
        }"""

            items_desc = "\n".join([
                f"- Search for '{item['search_key']}' and store as '{item['variable_name']}' (type: {item['type']})"
                for item in items_config
            ])

            # Prepare content based on type
            if content_type == "json":
                pages_data = json.loads(content) if isinstance(
                    content, str) else content
                content_text = ""
                start_page, end_page = page_range
                for page_num in range(start_page, end_page + 1):
                    if page_num in pages_data:
                        content_text += f"\n--- PAGE {page_num} ---\n{pages_data[page_num]['full_text']}\n"
            else:  # markdown
                content_text = content

            user_prompt = f"""Extract the following items:

        {items_desc}

        Document content:
        {content_text}

        Return the extracted items in JSON format."""

            try:
                client = AzureOpenAI(
                    azure_endpoint=OPENAI_ENDPOINT,
                    api_key=OPENAI_API_KEY,
                    api_version="2025-01-01-preview"
                )

                log_event("openai", "extract_items", f"Extracting {len(items_config)} items",
                        status="info", api_call="chat.completions.create")

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

                usage = response.usage
                log_event("openai", "extract_items", f"Successfully extracted {len(extracted_items)} items",
                        status="success", input_tokens=usage.prompt_tokens,
                        output_tokens=usage.completion_tokens, api_call="chat.completions.create")

                return extracted_items

            except Exception as e:
                log_event("openai", "extract_items", "Item extraction failed",
                        status="error", error_details=str(e), api_call="chat.completions.create")
                return {item['variable_name']: None for item in items_config}

        # ============= HTTP TRIGGER FUNCTIONS =============


        @app.route(route="classify", auth_level=func.AuthLevel.ANONYMOUS, methods=["POST"])
        def classify_document(req: func.HttpRequest) -> func.HttpResponse:
            """
            API 1 - Classify with flexible input/output
            
            Request body:
            {
                "file_path": "dev-mr-miyagi/files/file1.pdf",
                "input_type": "pdf|markdown",  # Optional, default: "pdf"
                "output_format": "json|markdown",  # Optional, default: "json"
                "output_destination": "dev-mr-miyagi/outcome/custom/",  # Optional
                "document_types": ["MESA", "Contract"]  # Optional
            }
            """
            log_event("api", "classify",
                    "API 1: Classify - HTTP trigger called", status="info")

            try:
                req_body = req.get_json()
                file_path = req_body.get('file_path')
                input_type = req_body.get('input_type', 'pdf')
                output_format = req_body.get('output_format', 'json')
                output_destination = req_body.get('output_destination')
                document_types = req_body.get('document_types')

                if not file_path:
                    return func.HttpResponse(
                        json.dumps({"error": "Missing 'file_path' in request body"}),
                        status_code=400,
                        mimetype="application/json"
                    )

                log_event("api", "classify", f"Processing file: {file_path}, input_type: {input_type}",
                        status="info", file_name=file_path)

                # Read file from blob
                file_bytes = read_file_from_blob(file_path)

                # Extract text
                if input_type == "pdf":
                    extracted_data = extract_text_with_document_intelligence(
                        file_bytes, output_format)
                else:  # markdown
                    extracted_data = {"markdown": file_bytes.decode('utf-8')}

                # Save extracted data
                base_filename = os.path.basename(file_path).replace(
                    '.pdf', '').replace('.md', '')

                if output_destination:
                    output_path = f"{output_destination}/{base_filename}-extracted.{output_format}"
                else:
                    output_path = f"outcome/{base_filename}/{base_filename}-extracted.{output_format}"

                if output_format == "markdown":
                    content_to_save = extracted_data.get("markdown", "")
                else:
                    content_to_save = json.dumps(extracted_data, indent=2)

                saved_path = save_to_blob(content_to_save, output_path)

                # Classify documents
                classification_result = classify_documents_with_ai(
                    content_to_save if output_format == "json" else extracted_data.get(
                        "markdown", ""),
                    output_format,
                    document_types
                )

                response_data = {
                    "status": "success",
                    "file_path": file_path,
                    "input_type": input_type,
                    "output_format": output_format,
                    "total_pages": len(extracted_data) if output_format == "json" else None,
                    "classification": classification_result,
                    "extracted_file_path": saved_path
                }

                log_event("api", "classify", "Classification completed successfully",
                        status="success", file_name=file_path)

                return func.HttpResponse(
                    json.dumps(response_data, indent=2),
                    status_code=200,
                    mimetype="application/json"
                )

            except Exception as e:
                log_event("api", "classify", "Classification failed",
                        status="error", file_name=req_body.get('file_path'), error_details=str(e))
                return func.HttpResponse(
                    json.dumps({"error": f"Internal server error: {str(e)}"}),
                    status_code=500,
                    mimetype="application/json"
                )


        @app.route(route="split", auth_level=func.AuthLevel.ANONYMOUS, methods=["POST"])
        def split_document(req: func.HttpRequest) -> func.HttpResponse:
            """
            API 2 - Split with flexible input
            
            Request body:
            {
                "file_path": "dev-mr-miyagi/files/file1.pdf",
                "classification": {...},
                "input_type": "pdf|json|markdown",  # Optional, default: "pdf"
                "output_destination": "dev-mr-miyagi/outcome/custom/"  # Optional
            }
            """
            log_event("api", "split", "API 2: Split - HTTP trigger called", status="info")

            try:
                req_body = req.get_json()
                file_path = req_body.get('file_path')
                classification = req_body.get('classification')
                input_type = req_body.get('input_type', 'pdf')
                output_destination = req_body.get('output_destination')

                if not file_path or not classification:
                    return func.HttpResponse(
                        json.dumps(
                            {"error": "Missing 'file_path' or 'classification'"}),
                        status_code=400,
                        mimetype="application/json"
                    )

                log_event("api", "split", f"Splitting file: {file_path}, input_type: {input_type}",
                        status="info", file_name=file_path)

                # Read file
                file_bytes = read_file_from_blob(file_path)

                if input_type != "pdf":
                    return func.HttpResponse(
                        json.dumps(
                            {"error": "Split operation currently only supports PDF input"}),
                        status_code=400,
                        mimetype="application/json"
                    )

                src_doc = fitz.open(stream=file_bytes, filetype="pdf")
                base_filename = os.path.basename(file_path).replace('.pdf', '')

                split_results = []

                for doc_type, sections in classification.items():
                    for seq, section in enumerate(sections, start=1):
                        start_page = section.get("start_page")
                        end_page = section.get("end_page")

                        if start_page is None or end_page is None:
                            continue

                        new_pdf = fitz.open()
                        for page_num in range(start_page - 1, end_page):
                            new_pdf.insert_pdf(
                                src_doc, from_page=page_num, to_page=page_num)

                        safe_doc_type = doc_type.replace(" ", "_").replace("/", "_")
                        new_filename = f"{base_filename}-{safe_doc_type}-{seq}.pdf"

                        if output_destination:
                            output_blob_name = f"{output_destination}/{new_filename}"
                        else:
                            output_blob_name = f"outcome/{base_filename}/{new_filename}"

                        pdf_bytes_out = new_pdf.tobytes()
                        blob_client = get_blob_client(output_blob_name)
                        blob_client.upload_blob(pdf_bytes_out, overwrite=True)

                        split_results.append({
                            "document_type": doc_type,
                            "sequence": seq,
                            "start_page": start_page,
                            "end_page": end_page,
                            "page_count": end_page - start_page + 1,
                            "file_path": f"{STORAGE_CONTAINER_NAME}/{output_blob_name}"
                        })

                        new_pdf.close()

                src_doc.close()

                log_event("api", "split", f"Split into {len(split_results)} files",
                        status="success", file_name=file_path)

                return func.HttpResponse(
                    json.dumps({
                        "status": "success",
                        "message": f"Split into {len(split_results)} files",
                        "files": split_results
                    }, indent=2),
                    status_code=200,
                    mimetype="application/json"
                )

            except Exception as e:
                log_event("api", "split", "Split operation failed",
                        status="error", file_name=req_body.get('file_path'), error_details=str(e))
                return func.HttpResponse(
                    json.dumps({"error": f"Processing error: {str(e)}"}),
                    status_code=500,
                    mimetype="application/json"
                )


        @app.route(route="extract", auth_level=func.AuthLevel.ANONYMOUS, methods=["POST"])
        def extract_metadata(req: func.HttpRequest) -> func.HttpResponse:
            """
            API 3 - Extract metadata with flexible input
            
            Request body:
            {
                "file_path": "dev-mr-miyagi/outcome/file1/file1-extracted.json",
                "classification": {...},
                "input_type": "json|markdown",  # Optional, default: "json"
                "items_to_extract": [
                    {"search_key": "broker name", "variable_name": "broker", "type": "single"},
                    {"search_key": "prices", "variable_name": "prices", "type": "list"}
                ],
                "output_destination": "dev-mr-miyagi/outcome/custom/"  # Optional
            }
            """
            log_event("api", "extract",
                    "API 3: Extract metadata - HTTP trigger called", status="info")

            try:
                req_body = req.get_json()
                file_path = req_body.get('file_path')
                classification = req_body.get('classification')
                input_type = req_body.get('input_type', 'json')
                items_to_extract = req_body.get('items_to_extract', [])
                output_destination = req_body.get('output_destination')

                if not file_path or not classification:
                    return func.HttpResponse(
                        json.dumps(
                            {"error": "Missing 'file_path' or 'classification'"}),
                        status_code=400,
                        mimetype="application/json"
                    )

                log_event("api", "extract", f"Extracting from: {file_path}, input_type: {input_type}",
                        status="info", file_name=file_path)

                # Read file
                file_content = read_file_from_blob(file_path).decode('utf-8')

                structured_results = []

                for doc_type, sections in classification.items():
                    for section in sections:
                        start_page = section.get("start_page")
                        end_page = section.get("end_page")

                        if start_page is None or end_page is None:
                            continue

                        metadata = {}
                        if items_to_extract:
                            metadata = extract_items_with_ai(
                                file_content,
                                input_type,
                                (start_page, end_page),
                                items_to_extract
                            )

                        structured_results.append({
                            "document_type": doc_type,
                            "start_page": start_page,
                            "end_page": end_page,
                            "page_count": end_page - start_page + 1,
                            "metadata": metadata
                        })

                # Save results if output destination specified
                if output_destination:
                    base_filename = os.path.basename(file_path).replace(
                        '-extracted.json', '').replace('-extracted.md', '')
                    output_path = f"{output_destination}/{base_filename}-metadata.json"
                    save_to_blob(json.dumps(structured_results, indent=2), output_path)

                log_event("api", "extract", f"Extracted metadata from {len(structured_results)} sections",
                        status="success", file_name=file_path)

                return func.HttpResponse(
                    json.dumps({
                        "status": "success",
                        "total_sections_analyzed": len(structured_results),
                        "structure": structured_results
                    }, indent=2),
                    status_code=200,
                    mimetype="application/json"
                )

            except Exception as e:
                log_event("api", "extract", "Metadata extraction failed",
                        status="error", file_name=req_body.get('file_path'), error_details=str(e))
                return func.HttpResponse(
                    json.dumps({"error": f"Processing error: {str(e)}"}),
                    status_code=500,
                    mimetype="application/json"
                )


        # Add this helper function to count tokens


        def count_tokens(text: str, model: str = "gpt-4") -> int:
            """Count tokens in text using tiktoken"""
            try:
                encoding = tiktoken.encoding_for_model(model)
                return len(encoding.encode(text))
            except:
                # Fallback estimation: ~4 chars per token
                return len(text) // 4


        def chunk_markdown_content(content: str, max_tokens: int = 8000) -> List[str]:
            """
            Intelligently chunk markdown content by sections/headers
            while staying under token limit
            """
            chunks = []
            current_chunk = []
            current_tokens = 0

            # Split by double newlines (paragraphs) and headers
            sections = content.split('\n\n')

            for section in sections:
                section_tokens = count_tokens(section)

                # If single section exceeds limit, split by sentences
                if section_tokens > max_tokens:
                    sentences = section.split('. ')
                    for sentence in sentences:
                        sentence_tokens = count_tokens(sentence)
                        if current_tokens + sentence_tokens > max_tokens:
                            if current_chunk:
                                chunks.append('\n\n'.join(current_chunk))
                            current_chunk = [sentence]
                            current_tokens = sentence_tokens
                        else:
                            current_chunk.append(sentence)
                            current_tokens += sentence_tokens
                else:
                    if current_tokens + section_tokens > max_tokens:
                        if current_chunk:
                            chunks.append('\n\n'.join(current_chunk))
                        current_chunk = [section]
                        current_tokens = section_tokens
                    else:
                        current_chunk.append(section)
                        current_tokens += section_tokens

            if current_chunk:
                chunks.append('\n\n'.join(current_chunk))

            return chunks


        def merge_extraction_results(results: List[Dict]) -> Dict:
            """Merge results from multiple chunks"""
            merged = {
                "key_value_pairs": {},
                "tables": []
            }

            for result in results:
                # Merge key-value pairs (avoid duplicates)
                if "key_value_pairs" in result:
                    merged["key_value_pairs"].update(result["key_value_pairs"])

                # Append tables
                if "tables" in result:
                    merged["tables"].extend(result["tables"])

            return merged


        def extract_kv_and_tables_from_chunk(content: str) -> Dict:
            """Extract key-value pairs and tables from a single chunk"""
            system_prompt = """You are a data extraction expert. Analyze the markdown content and extract:

        1. KEY-VALUE PAIRS: Any meaningful data in key-value format (labels, fields, metadata, properties, etc.)
        2. TABLES: All tables with their structure and data

        IMPORTANT:
        - Extract ALL key-value pairs you find (dates, names, IDs, amounts, statuses, etc.)
        - For tables: capture headers and all rows
        - Return exact values as they appear
        - Use consistent key naming (lowercase, underscores)
        - Skip empty or meaningless data

        Return JSON in this EXACT format:
        {
        "key_value_pairs": {
            "key_name": "value",
            "another_key": "another_value"
        },
        "tables": [
            {
            "table_name": "descriptive name or null",
            "headers": ["col1", "col2", "col3"],
            "rows": [
                ["val1", "val2", "val3"],
                ["val4", "val5", "val6"]
            ]
            }
        ]
        }"""

            user_prompt = f"""Extract all key-value pairs and tables from this content:

        {content}

        Return the extraction in JSON format."""

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

                # Log token usage
                usage = response.usage
                log_event("openai", "extract_kv_tables",
                        f"Extracted data from chunk ({usage.prompt_tokens} input tokens)",
                        status="success",
                        input_tokens=usage.prompt_tokens,
                        output_tokens=usage.completion_tokens,
                        api_call="chat.completions.create")

                return result_json

            except Exception as e:
                log_event("openai", "extract_kv_tables", "Extraction failed for chunk",
                        status="error", error_details=str(e), api_call="chat.completions.create")
                return {"key_value_pairs": {}, "tables": []}

        # ============= NEW HTTP TRIGGER FUNCTION =============


        @app.route(route="extract_kv_tables", auth_level=func.AuthLevel.ANONYMOUS, methods=["POST"])
        def extract_kv_tables(req: func.HttpRequest) -> func.HttpResponse:
            """
            API 4 - Extract key-value pairs and tables from markdown
            
            Request body:
            {
                "file_path": "dev-mr-miyagi/files/document.md",
                "output_destination": "dev-mr-miyagi/outcome/custom/"  # Optional
            }
            
            Response:
            {
                "status": "success",
                "file_path": "...",
                "chunks_processed": 3,
                "total_key_value_pairs": 15,
                "total_tables": 2,
                "result": {
                    "key_value_pairs": {...},
                    "tables": [...]
                },
                "output_file_path": "..."  # If output_destination provided
            }
            """
            log_event("api", "extract_kv_tables",
                    "API 4: Extract KV and Tables - HTTP trigger called", status="info")

            try:
                req_body = req.get_json()
                file_path = req_body.get('file_path')
                output_destination = req_body.get('output_destination')

                if not file_path:
                    return func.HttpResponse(
                        json.dumps({"error": "Missing 'file_path' in request body"}),
                        status_code=400,
                        mimetype="application/json"
                    )

                log_event("api", "extract_kv_tables", f"Processing markdown file: {file_path}",
                        status="info", file_name=file_path)

                # Read markdown file from blob
                markdown_content = read_file_from_blob(file_path).decode('utf-8')

                # Check total tokens
                total_tokens = count_tokens(markdown_content)
                log_event("api", "extract_kv_tables",
                        f"Markdown file has ~{total_tokens} tokens",
                        status="info", file_name=file_path)

                # Chunk content if needed (keeping buffer for system prompt and response)
                max_chunk_tokens = 8000  # Conservative limit for GPT-4o

                if total_tokens > max_chunk_tokens:
                    log_event("api", "extract_kv_tables",
                            f"Content exceeds limit, chunking into smaller pieces",
                            status="info", file_name=file_path)
                    chunks = chunk_markdown_content(markdown_content, max_chunk_tokens)
                else:
                    chunks = [markdown_content]

                log_event("api", "extract_kv_tables",
                        f"Processing {len(chunks)} chunk(s)",
                        status="info", file_name=file_path)

                # Process each chunk
                chunk_results = []
                for i, chunk in enumerate(chunks, 1):
                    log_event("api", "extract_kv_tables",
                            f"Processing chunk {i}/{len(chunks)}",
                            status="info", file_name=file_path)
                    result = extract_kv_and_tables_from_chunk(chunk)
                    chunk_results.append(result)

                # Merge all results
                final_result = merge_extraction_results(chunk_results)

                # Save to blob if output destination specified
                output_file_path = None
                if output_destination:
                    base_filename = os.path.basename(file_path).replace('.md', '')
                    output_path = f"{output_destination}/{base_filename}-kv-tables.json"
                    output_file_path = save_to_blob(
                        json.dumps(final_result, indent=2),
                        output_path
                    )

                response_data = {
                    "status": "success",
                    "file_path": file_path,
                    "chunks_processed": len(chunks),
                    "total_key_value_pairs": len(final_result.get("key_value_pairs", {})),
                    "total_tables": len(final_result.get("tables", [])),
                    "result": final_result
                }

                if output_file_path:
                    response_data["output_file_path"] = output_file_path

                log_event("api", "extract_kv_tables",
                        f"Successfully extracted {response_data['total_key_value_pairs']} key-value pairs and {response_data['total_tables']} tables",
                        status="success", file_name=file_path)

                return func.HttpResponse(
                    json.dumps(response_data, indent=2),
                    status_code=200,
                    mimetype="application/json"
                )

            except Exception as e:
                log_event("api", "extract_kv_tables", "Extraction failed",
                        status="error", file_name=req_body.get('file_path'), error_details=str(e))
                return func.HttpResponse(
                    json.dumps({"error": f"Processing error: {str(e)}"}),
                    status_code=500,
                    mimetype="application/json"
                )

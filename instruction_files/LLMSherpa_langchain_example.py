
# First, ensure you have the necessary packages installed
# You can install them using pip if you haven't already
# !pip install llmsherpa langchain

from langchain_community.document_loaders.llmsherpa import LLMSherpaFileLoader

# Specify the path to your PDF file
file_path = "path_to_your_document.pdf"

# Initialize the LLMSherpaFileLoader with your desired settings
loader = LLMSherpaFileLoader(
    file_path=file_path,
    new_indent_parser=True,  # Use the new indent parser for better section alignment
    apply_ocr=True,          # Apply OCR if your document requires it
    strategy="chunks",       # Choose your strategy: 'sections', 'chunks', 'html', or 'text'
    llmsherpa_api_url="https://readers.llmsherpa.com/api/document/developer/parseDocument?renderFormat=all"
)

# Load the document
docs = loader.load()

# Now, 'docs' contains the parsed content of your PDF
# You can process it further as needed

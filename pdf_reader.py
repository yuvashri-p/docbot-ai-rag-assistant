# pdf_reader.py

from pypdf import PdfReader  
# pypdf is the library — PdfReader is the specific tool inside it
# Think of it like: import toolbox → pick the wrench you need

def extract_text_from_pdf(file):
    """
    Takes a PDF file, returns all its text as one string.
    
    Why a function? So app.py can call this one line:
    text = extract_text_from_pdf(uploaded_file)
    ...instead of rewriting this logic every time.
    """
    
    reader = PdfReader(file)
    # PdfReader opens the PDF — like opening a book
    
    extracted_text = ""
    # Empty string — we'll keep adding pages to this
    
    for page_number, page in enumerate(reader.pages):
        # enumerate() gives us both the index (0,1,2...)
        # and the page object at the same time
        
        text = page.extract_text()
        # This reads the text layer of one page
        
        if text:  
            # Some pages are images (scanned PDFs) — no text layer
            # This check skips blank/image-only pages
            
            extracted_text += f"\n--- Page {page_number + 1} ---\n"
            # We add a page label so we know where content came from
            # +1 because humans count from 1, computers from 0
            
            extracted_text += text
    
    return extracted_text
    # Send the full text back to whoever called this function


def get_pdf_info(file):
    """
    Returns basic info about the PDF — page count, title etc.
    Useful to show the user their file was read correctly.
    """
    reader = PdfReader(file)
    
    info = {
        "pages": len(reader.pages),
        # len() counts items in a list — here it counts pages
        
        "title": reader.metadata.title if reader.metadata else "Unknown"
        # metadata is like a PDF's ID card — has title, author etc.
        # 'if reader.metadata' checks it exists before reading it
        # avoids crash if PDF has no metadata
    }
    
    return info
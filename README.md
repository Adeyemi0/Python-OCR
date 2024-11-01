# Document Scanner with OCR

## Overview

This project is a console-based document scanner application that uses Optical Character Recognition (OCR) to extract text and tables from images and PDF files. It leverages the PaddleOCR and PPStructure libraries for accurate text recognition and structured data extraction, making it useful for automating data entry and document analysis tasks.

## Features

- **Text Extraction**: Extracts text from images and PDF documents using OCR.
- **Table Recognition**: Detects and extracts tables from scanned documents and images.
- **Image Preprocessing**: Enhances image quality for better OCR performance through various preprocessing techniques.
- **Output Export**: Saves extracted data into an Excel file for easy sharing and further analysis.

## Requirements

- Python 3.8
- `opencv-python`
- `numpy`
- `paddleocr`
- `ppstructure`
- `pdfplumber`
- `pandas`
- `Pillow`

## Installation

1. Clone the repository:

   ```bash
   git clone https://github.com/Adeyemi0/Python-OCR.git
   cd document-scanner
   ```
2. Install the required packages:
   ```bash
   pip install opencv-python numpy paddleocr ppstructure pdfplumber pandas Pillow
   ```
## Usage
```bash
python Python-OCR.py
 ```


import streamlit as st
import os
import cv2
import numpy as np
from paddleocr import PPStructure, draw_structure_result, save_structure_res
import pdfplumber
import pandas as pd
from fpdf import FPDF
from io import BytesIO
import tempfile
import concurrent.futures
import re

# Initialize PaddleOCR with the PPStructure module and English language model
ocr = PPStructure(recovery=True, use_angle_cls=True, lang='en')

# Preprocess images for better OCR performance
def preprocess_image(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image_resized = cv2.resize(gray, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    contrast_enhanced = clahe.apply(image_resized)
    binary = cv2.adaptiveThreshold(contrast_enhanced, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
    denoised = cv2.fastNlMeansDenoising(binary, h=10)
    denoised = cv2.medianBlur(denoised, 3)
    kernel = np.ones((3, 3), np.uint8)
    morph = cv2.morphologyEx(denoised, cv2.MORPH_CLOSE, kernel)
    kernel_sharpening = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
    sharpened = cv2.filter2D(morph, -1, kernel_sharpening)
    deskewed = deskew_image(sharpened)
    return deskewed

# Perform OCR with bounding box processing using PPStructure
def perform_ocr_with_bounding_boxes(image, image_path, output_dir, min_confidence=0.6):
    # Get text detection and recognition results from PPStructure
    ocr_results = ocr(image)  
    results = []
    
    # Save the structured OCR results for better presentation
    save_structure_res(ocr_results, output_dir, os.path.basename(image_path).split('.')[0])

    # Visualize the extracted table structure on the image
    image_with_structure = draw_structure_result(image, ocr_results, font_path=None)

    # Display the result image
    st.image(image_with_structure, caption="Extracted Table Structure", use_column_width=True)
    
    for item in ocr_results:
        for elem in item['layout']['elements']:
            text = elem['text']
            bbox = elem['bbox']
            confidence = elem.get('confidence', 1.0)  # Default to 1.0 if confidence is not present
            if confidence > min_confidence:
                # Append bounding box coordinates along with the detected text
                results.append({'text': text, 'bbox': bbox, 'confidence': confidence})

    return results

# Function to process bounding boxes and organize data spatially
def process_bounding_boxes(ocr_data):
    ocr_data_sorted = sorted(ocr_data, key=lambda x: (x['bbox'][1], x['bbox'][0]))  # Sort by y, then x
    organized_data = []
    for data in ocr_data_sorted:
        bbox = data['bbox']
        text = data['text']
        organized_data.append({'text': text, 'bbox': bbox})
    
    return organized_data

# Deskew image to correct text angle
def deskew_image(image):
    coords = np.column_stack(np.where(image > 0))
    angle = cv2.minAreaRect(coords)[-1]

    if angle < -45:
        angle = -(90 + angle)
    else:
        angle = -angle

    (h, w) = image.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
    return rotated

# Load and process files (images or PDFs)
def load_and_extract(files, output_dir):
    extracted_data = []  # Use a list instead of a set
    with concurrent.futures.ThreadPoolExecutor() as executor:
        future_to_file = {executor.submit(process_file, file, output_dir): file for file in files}
        for future in concurrent.futures.as_completed(future_to_file):
            result = future.result()
            if result:
                extracted_data.extend(result)  # Use extend() to add results to the list
    return extracted_data

# Process individual file
def process_file(file, output_dir):
    file_type = file.type
    extracted_data = []
    with tempfile.NamedTemporaryFile(delete=False) as temp_file:
        temp_file.write(file.read())
        temp_file.flush()
        if file_type == "application/pdf":
            extracted_data.extend(process_pdf(temp_file.name, output_dir))
        elif file_type.startswith("image/"):
            extracted_data.extend(process_image(temp_file.name, output_dir))
        else:
            st.warning(f"Unsupported file type: {file_type}")
    return extracted_data

# Process PDF for text and images
def process_pdf(pdf_path, output_dir):
    extracted_data = []
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            text = page.extract_text()
            if text:
                extracted_data.append({'text': clean_text(text), 'bbox': None})
            for img in page.images:
                cropped_image = extract_image_from_pdf(page, img)
                if cropped_image is not None:
                    preprocessed_img = preprocess_image(cropped_image)
                    ocr_data = perform_ocr_with_bounding_boxes(preprocessed_img, pdf_path, output_dir)
                    processed_data = process_bounding_boxes(ocr_data)
                    extracted_data.extend(processed_data)
    return extracted_data

# Process image files for OCR
def process_image(image_path, output_dir):
    image = cv2.imread(image_path)
    if image is None:
        st.error(f"Error loading image from path: {image_path}")
        return []
    preprocessed_img = preprocess_image(image)
    ocr_data = perform_ocr_with_bounding_boxes(preprocessed_img, image_path, output_dir)
    processed_data = process_bounding_boxes(ocr_data)
    return processed_data

# Main Streamlit app
def main():
    st.title("Document Scanner with OCR")
    st.write("Upload images or PDFs, extract data, and download results in Excel or PDF.")
    uploaded_files = st.file_uploader("Upload Documents (Images or PDFs)", accept_multiple_files=True)
    output_dir = "ocr_output"  # Directory to save OCR results
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    if uploaded_files:
        st.write("Processing documents...")
        extracted_data = load_and_extract(uploaded_files, output_dir)
        final_output = "\n".join([d['text'] for d in extracted_data])
        if final_output:
            st.write("Final Extracted Data:")
            st.text(final_output)
        else:
            st.write("No data extracted.")
        output_format = st.selectbox("Select output format", ["Excel", "PDF"])
        if st.button("Download"):
            if output_format == "Excel":
                excel_data = export_to_excel(extracted_data)
                st.download_button("Download Excel", excel_data, file_name="ocr_results.xlsx")
            elif output_format == "PDF":
                pdf_data = export_to_pdf(extracted_data)
                st.download_button("Download PDF", pdf_data, file_name="ocr_results.pdf")

if __name__ == "__main__":
    main()

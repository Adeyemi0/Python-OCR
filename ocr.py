import streamlit as st
import os
import cv2
import numpy as np
from paddleocr import PaddleOCR, PPStructure, draw_structure_result, save_structure_res
import pdfplumber
import pandas as pd
from io import BytesIO
import tempfile
import concurrent.futures
import re
from PIL import Image

# Initialize PaddleOCR with the English language model
ocr = PaddleOCR(use_angle_cls=True, lang='en')

# Initialize the table recognition engine
table_engine = PPStructure(show_log=True)
save_folder = './output'  # Folder to save the results

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
    kernel_sharpening = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]] )
    sharpened = cv2.filter2D(morph, -1, kernel_sharpening)
    deskewed = deskew_image(sharpened)
    return deskewed

# Perform OCR with structure detection
def perform_ocr(image, min_confidence=0.6):
    results = ocr.ocr(image, cls=True)
    extracted_data = []

    for result in results:
        for line in result:
            if line[1][1] > min_confidence:  # Confidence check
                text = line[1][0]
                bbox = line[0]
                extracted_data.append({'text': text, 'bbox': bbox, 'confidence': line[1][1]})

    return extracted_data

# Perform table recognition
def perform_table_recognition(image, img_path):
    result = table_engine(image)  # Recognize table structure
    save_structure_res(result, save_folder, os.path.basename(img_path).split('.')[0])  # Save results

    table_data = []
    for line in result:
        line.pop('img')  # Remove image data
        table_data.append(line)  # Append structured result

    return table_data

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
def load_and_extract(files):
    extracted_data = []  # Use a list instead of a set

    with concurrent.futures.ThreadPoolExecutor() as executor:
        future_to_file = {executor.submit(process_file, file): file for file in files}
        for future in concurrent.futures.as_completed(future_to_file):
            result = future.result()
            if result:
                extracted_data.extend(result)  # Use extend() to add results to the list

    return extracted_data

# Process individual file
def process_file(file):
    file_type = file.type
    extracted_data = []

    with tempfile.NamedTemporaryFile(delete=False) as temp_file:
        temp_file.write(file.read())
        temp_file.flush()

        if file_type == "application/pdf":
            extracted_data.extend(process_pdf(temp_file.name))
        elif file_type.startswith("image/"):
            extracted_data.extend(process_image(temp_file.name))
        else:
            st.warning(f"Unsupported file type: {file_type}")

    return extracted_data

# Process PDF for text and images
def process_pdf(pdf_path):
    extracted_data = []

    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            text = page.extract_text()
            if text:
                extracted_data.append({'text': clean_text(text), 'bbox': None})  # No bounding box for plain text

            for img in page.images:
                cropped_image = extract_image_from_pdf(page, img)
                if cropped_image is not None:
                    preprocessed_img = preprocess_image(cropped_image)
                    ocr_data = perform_ocr(preprocessed_img)  # Use the updated function
                    processed_data = process_bounding_boxes(ocr_data)  # Call the bounding box processing function
                    extracted_data.extend(processed_data)

                    # Perform table recognition
                    table_data = perform_table_recognition(cropped_image, pdf_path)
                    extracted_data.extend(table_data)  # Add table data to extracted data

    return extracted_data

# Extract and preprocess image from a PDF page
def extract_image_from_pdf(page, img):
    bbox = (img["x0"], img["top"], img["x1"], img["bottom"])
    cropped_image = page.within_bbox(bbox).to_image().original
    return cv2.cvtColor(np.array(cropped_image), cv2.COLOR_RGB2BGR) if cropped_image is not None else None

# Process image files for OCR
def process_image(image_path):
    image = cv2.imread(image_path)
    
    if image is None:
        st.error(f"Error loading image from path: {image_path}")
        return []

    preprocessed_img = preprocess_image(image)
    ocr_data = perform_ocr(preprocessed_img)  # Use the updated function
    processed_data = process_bounding_boxes(ocr_data)  # Call the bounding box processing function

    # Perform table recognition
    table_data = perform_table_recognition(image, image_path)
    processed_data.extend(table_data)  # Add table data to processed data

    return processed_data

# Clean extracted text
def clean_text(text):
    text = re.sub(r'\s+', ' ', text)
    text = text.strip()
    return text

# Export extracted data to Excel
def export_to_excel(data):
    output = BytesIO()
    df = pd.DataFrame(data)  # This will automatically use the 'text' and 'bbox' keys from the data
    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
        df.to_excel(writer, index=False, sheet_name='OCR Results')
    return output.getvalue()

# Process bounding boxes from OCR results
def process_bounding_boxes(ocr_data):
    processed_data = []
    for entry in ocr_data:
        text = entry['text']
        bbox = entry['bbox']  # Assuming bbox is in the format required for display
        confidence = entry['confidence']

        processed_data.append({
            'text': text,
            'bbox': bbox,
            'confidence': confidence
        })
    return processed_data

# Main Streamlit app
def main():
    st.title("Document Scanner with OCR")
    st.write("Upload your images or PDF files for OCR and table recognition.")

    uploaded_files = st.file_uploader("Choose files", type=["png", "jpg", "jpeg", "pdf"], accept_multiple_files=True)

    if uploaded_files and st.button("Process"):
        with st.spinner("Processing..."):
            extracted_data = load_and_extract(uploaded_files)

        # Display extracted data
        for data in extracted_data:
            st.write(data)

        # Export option for Excel
        excel_data = export_to_excel(extracted_data)
        st.download_button("Download Excel", excel_data, "extracted_data.xlsx", "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

if __name__ == "__main__":
    main()

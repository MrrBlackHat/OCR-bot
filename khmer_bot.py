import os
import logging
import sys
import re
import subprocess
from telegram import Update
from telegram.ext import Application, CommandHandler, MessageHandler, filters, ContextTypes, ExtBot 
from PIL import Image
import pytesseract
import io
from pdf2image import convert_from_bytes
import cv2
import numpy as np

# --- Configuration and Setup ---

# Configure logging
LOG_FILE = 'universal_khmer_bot.log'
logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO,
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(LOG_FILE, mode='a') 
    ]
)
logger = logging.getLogger(__name__)

# Configure Tesseract for Windows (Crucial path configuration kept)
TESSERACT_PATH = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
try:
    pytesseract.pytesseract.tesseract_cmd = TESSERACT_PATH
    logger.info(f"Tesseract path successfully set to: {TESSERACT_PATH}") 
except Exception:
    logger.warning(f"Could not set tesseract_cmd to {TESSERACT_PATH}. Ensure Tesseract is in system PATH.")

# Universal Khmer Character Set (Kept same)
KHMER_CHAR_SET = "កខគឃងចឆជឈញដឋឌឍណតថទធនបផពភមយរលវឝឞសហឡអឣឤឥឦឧឨឩឪឫឬឭឮឯឰឱឲឳ្ៈាេឹីុូួើឿៀេែៃោៅំុំះ៉៊៌៍៎៏័៑ៗ៕៖។០១២៣៤៥៦៧៨៩"
ENGLISH_CHAR_SET = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ"
NUMBERS_PUNCTUATION_SYMBOLS = r" .,!?;:-+()[]{}<>/=MDSDtpFANOVA0123456789"
WHITELIST_CHARS = ENGLISH_CHAR_SET + KHMER_CHAR_SET + NUMBERS_PUNCTUATION_SYMBOLS

# Constants for better readability
MAX_PDF_PAGES = 5 
MAX_MESSAGE_LENGTH = 4096 # Telegram message limit

class UniversalKhmerTextExtractorBot:
    def __init__(self, token: str):
        self.token = token
        self.application = Application.builder().token(token).build()
        
        # --- Add handlers (Fixed: handlers must match existing methods) ---
        self.application.add_handler(CommandHandler("start", self.start_command))
        self.application.add_handler(CommandHandler("help", self.help_command))
        self.application.add_handler(CommandHandler("tips", self.tips_command))
        self.application.add_handler(CommandHandler("languages", self.languages_command))
        self.application.add_handler(MessageHandler(filters.PHOTO, self.handle_image))
        self.application.add_handler(MessageHandler(filters.Document.PDF, self.handle_document)) 
        self.application.add_handler(MessageHandler(filters.Document.ALL & ~filters.Document.PDF, self.handle_unsupported_document)) # Reject non-PDF documents
        
        self.universal_correction_patterns = self._initialize_universal_corrections()
        
        logger.info("Universal Khmer Text Extractor Bot initialized")
    
    def _initialize_universal_corrections(self):
        """Initialize universal correction patterns, focusing on common mixed-script errors."""
        return {
            # Academic/Statistical Symbols and common OCR errors
            '= $03ន': 'SQ3R', '= 1.|8': 'M=1.78', '»1=': 'M=', 'ស=!': 'M=',
            'ស20030កំ': 'SPSS', 'ដ»06!': 'Version', 'ពន ៤-$សាព្ង(63 !-3!': 'One-way ANOVA',
            
            # IMPROVED: Targeted Latin character confusion fixes
            'l': '1', 'O': '0', 'o': '0', '|': '1', 'i': '1',
            '?': '.', '១': '1', '០': '0',
            
            # Punctuation/Spacing fixes (retained for robustness)
            ' .': '.', ' ,': ',', ' ;': ';', ' :': ':',
        }
        
    # --- Image Processing & Enhancement ---

    def enhance_image_universal(self, image: Image.Image) -> Image.Image:
        """
        Universal image preprocessing for mixed Khmer-English content.
        Includes Adaptive Resizing, Tesseract OSD for Deskewing, Noise Reduction, and
        Morphological operations optimized for Khmer sub-scripts.
        """
        try:
            img_cv = cv2.cvtColor(np.array(image.convert('RGB')), cv2.COLOR_RGB2BGR)
            gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
            
            # 1. Image Resizing (Adaptive)
            height, width = gray.shape
            scale_factor = 1.0
            if width < 1200:
                scale_factor = 1200 / width
            
            if scale_factor != 1.0:
                new_width = int(width * scale_factor)
                new_height = int(height * scale_factor)
                interp = cv2.INTER_CUBIC if scale_factor > 1 else cv2.INTER_AREA
                gray = cv2.resize(gray, (new_width, new_height), interpolation=interp)
                
            # 2. Orientation/Skew Correction (Tesseract OSD Pass)
            osd_data = pytesseract.image_to_osd(Image.fromarray(gray))
            angle_match = re.search(r"Rotate:\s*(\d+)", osd_data)
            
            if angle_match:
                angle = int(angle_match.group(1))
                if angle != 0:
                    logger.info(f"Deskewing by {angle} degrees detected by Tesseract OSD.")
                    (h, w) = gray.shape[:2]
                    center = (w // 2, h // 2)
                    M = cv2.getRotationMatrix2D(center, angle, 1.0)
                    gray = cv2.warpAffine(gray, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)

            # 3. Noise Reduction and Contrast
            denoised = cv2.fastNlMeansDenoising(gray, None, 30, 7, 21)
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            contrast_enhanced = clahe.apply(denoised)
            
            # 4. Adaptive Thresholding 
            thresh = cv2.adaptiveThreshold(
                contrast_enhanced, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                cv2.THRESH_BINARY, 15, 10
            )
            
            # 5. Robust Morphological Closing (Connects broken Khmer characters horizontally)
            kernel_close = np.ones((1, 3), np.uint8) # 1 row, 3 columns
            cleaned = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel_close)
            
            return Image.fromarray(cleaned)
            
        except Exception as e:
            logger.error(f"Universal image enhancement failed. Skipping enhancement: {e}")
            return image

    # --- Core Text Extraction ---

    def extract_text_universal(self, image: Image.Image) -> str:
        """
        Runs multiple OCR configurations (Multi-Pass OCR) and selects the
        highest-scoring output.
        """
        try:
            processed_image = self.enhance_image_universal(image)
            
            universal_configs = [
                # 1. English Only: Highest Priority pass for perfect Latin text
                {'config': r'--oem 3 --psm 6 -l eng', 'name': 'English Only PSM 6', 'weight': 1.2}, 
                # 2. Academic/Technical: Best mixed config with a whitelist
                {'config': r'--oem 3 --psm 6 -l khm+eng -c tessedit_char_whitelist=' + WHITELIST_CHARS, 
                 'name': 'Academic Whitelist', 'weight': 1.1},
                # 3. Primary Mixed: Standard PSM 6 
                {'config': r'--oem 3 --psm 6 -l khm+eng', 'name': 'Mixed PSM 6', 'weight': 1.0},
                # 4. Secondary Mixed: PSM 3 
                {'config': r'--oem 3 --psm 3 -l khm+eng', 'name': 'Mixed PSM 3', 'weight': 0.95},
                # 5. Fallback: Khmer only 
                {'config': r'--oem 3 --psm 6 -l khm', 'name': 'Khmer Only', 'weight': 0.9},
            ]
            
            all_results = []
            
            for config in universal_configs:
                try:
                    text = pytesseract.image_to_string(processed_image, config=config['config'])
                    text = text.strip()
                    
                    if text and len(text) > 10:
                        score = self._score_universal_text_quality(text) * config['weight']
                        all_results.append({
                            'text': text,
                            'score': score,
                            'config': config['name'],
                            'language_mix': self._detect_language_mix(text)
                        })
                except Exception as e:
                    logger.warning(f"OCR config {config['name']} failed: {e}")
                    
            if not all_results:
                return "No readable text detected. Please try with a clearer image."
            
            best_result = max(all_results, key=lambda x: x['score'])
            logger.info(f"Best OCR: {best_result['config']} | Score: {best_result['score']:.2f} | Language: {best_result['language_mix']}")
            
            corrected_text = self._apply_universal_correction(best_result['text'])
            
            return corrected_text
            
        except Exception as e:
            logger.error(f"Error in universal text extraction: {e}")
            return f"OCR processing error: {str(e)}"

    def _detect_language_mix(self, text: str) -> str:
        """Detect the language mix in the text."""
        khmer_count = sum(1 for char in text if char in KHMER_CHAR_SET)
        english_count = sum(1 for char in text if char in ENGLISH_CHAR_SET)
        total_letters = khmer_count + english_count
        
        if total_letters == 0:
            return "Unknown"
        
        khmer_ratio = khmer_count / total_letters
        english_ratio = english_count / total_letters
        
        if khmer_ratio > 0.8:
            return "Mostly Khmer"
        elif english_ratio > 0.8:
            return "Mostly English"
        else:
            return "Mixed Khmer-English"

    def _score_universal_text_quality(self, text: str) -> float:
        """Score text quality based on character validity and language diversity."""
        if not text:
            return 0.0
        
        all_valid_chars = WHITELIST_CHARS + "\n\t"
        
        # 1. Valid Character Ratio 
        total_chars = len(text)
        valid_chars = sum(1 for char in text if char in all_valid_chars)
        valid_ratio = valid_chars / total_chars if total_chars > 0 else 0
        
        # 2. Language Diversity Score (Retained bonus for mixed content)
        khmer_count = sum(1 for char in text if char in KHMER_CHAR_SET)
        english_count = sum(1 for char in text if char in ENGLISH_CHAR_SET)
        
        language_diversity = 0
        if khmer_count > 0 and english_count > 0:
            language_diversity = 0.4  
        
        # 3. Word Count/Density Score 
        words = text.split()
        word_score = min(len(words) / 30, 1.0) * 0.2
        
        final_score = (valid_ratio * 0.4) + language_diversity + word_score
        
        return final_score

    # --- Correction and Formatting ---

    def _apply_universal_correction(self, text: str) -> str:
        """Apply predefined and regex-based universal corrections."""
        if not text:
            return ""
        
        # Step 1: Apply predefined corrections
        corrected = text
        for wrong, right in self.universal_correction_patterns.items():
            corrected = corrected.replace(wrong, right)
        
        # Step 2: Fix common mixed-language patterns (Regex)
        corrected = self._fix_mixed_language_patterns(corrected)
        
        # Step 3: Normalize spacing and formatting
        corrected = self._normalize_universal_formatting(corrected)
        
        return corrected

    def _fix_mixed_language_patterns(self, text: str) -> str:
        """Fix common mixed Khmer-English OCR errors using RegEx."""
        
        # 1. Standardize SQ3R and common academic terms
        text = re.sub(r'\b(SQ3R|S03R|S03រ|SQ3រ)\b', 'SQ3R', text, flags=re.IGNORECASE)
        text = re.sub(r'\b(SPSS|SPS|ស20030)\b', 'SPSS', text)
        text = re.sub(r'\b(ANOVA|ANVOA|សាព្ង)\b', 'ANOVA', text)
        
        # 2. Fix M=SD= statistical string errors
        text = re.sub(r'([M|m]=[\s]*[\d.]+)\s*\(\s*(S|s|$)D[\s]*=[\s]*([\d.]+)\s*\)', r'\1 (SD=\3)', text)
        
        # 3. New English-specific character fixes (i.e., '1' misread as 'l' in numerical context)
        text = re.sub(r'([M|S|D|m|s|d|=])\s*l\s*(\d)', r'\1\2', text) 
        
        return text

    def _normalize_universal_formatting(self, text: str) -> str:
        """Normalize formatting for all text types (spacing, line breaks, punctuation)."""
        # Normalize spaces
        text = re.sub(r'[ \t]+', ' ', text)
        
        # Normalize double line breaks
        text = re.sub(r'\n[ \t]*\n', '\n\n', text)
        
        # Fix punctuation spacing: No space before punctuation
        text = re.sub(r'\s+([.,!?;:])', r'\1', text)
        text = re.sub(r'\s+([។៕៖ៗ])', r'\1', text) # Khmer punctuation
        
        # Fix punctuation spacing: Ensure space after period/comma/etc. if followed by a letter
        text = re.sub(r'([.,!?;:])([A-Za-zក-ឳ])', r'\1 \2', text)
        
        # Remove leading/trailing whitespace
        text = text.strip()
        
        return text

    # --- Document and Handler Methods ---

    def extract_text_from_pdf(self, pdf_bytes: bytes) -> str:
        """Extract text from PDF by converting pages to images and running universal OCR."""
        try:
            images = convert_from_bytes(pdf_bytes, dpi=300, first_page=1, last_page=MAX_PDF_PAGES)
            
            extracted_text = []
            
            for i, image in enumerate(images):
                logger.info(f"Processing PDF page {i+1}/{len(images)}")
                
                page_text = self.extract_text_universal(image)
                if page_text and "No readable text" not in page_text:
                    extracted_text.append(f"--- Page {i+1} ---\n{page_text}")
            
            if extracted_text:
                result = "\n\n".join(extracted_text)
                
                # Check if the PDF was truncated (only check if we got MAX_PDF_PAGES)
                if len(images) == MAX_PDF_PAGES:
                    # Quick check to see if the PDF has more pages than we processed
                    try:
                        total_pages = len(convert_from_bytes(pdf_bytes, first_page=1, last_page=9999, fmt='ppm', thread_count=1))
                        if total_pages > MAX_PDF_PAGES:
                            result += f"\n\n... and more pages were skipped (limit {MAX_PDF_PAGES} pages for speed)."
                    except Exception as e:
                        logger.warning(f"Could not reliably count total PDF pages: {e}")

                return result
            else:
                return "No readable text found in the PDF."
            
        except Exception as e:
            logger.error(f"Error processing PDF: {e}")
            return f"Error processing PDF: {str(e)}"
            
    async def _send_long_message(self, update: Update, text: str, header: str):
        """Helper function to split and send long messages up to 4096 chars."""
        if len(text) > MAX_MESSAGE_LENGTH:
            chunks = [text[i:i+MAX_MESSAGE_LENGTH] for i in range(0, len(text), MAX_MESSAGE_LENGTH)]
            
            # Send the first chunk with the main header
            await update.message.reply_text(header + chunks[0])
            
            # Send subsequent chunks as replies to the first, indicating parts
            for i, chunk in enumerate(chunks[1:]):
                await update.message.reply_text(f"📝 **(Part {i+2})**:\n\n" + chunk)
        else:
            await update.message.reply_text(header + text)


    async def handle_image(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle incoming images."""
        try:
            await update.message.reply_chat_action(action="typing")
            
            photo_file = await update.message.photo[-1].get_file()
            photo_bytes = await photo_file.download_as_bytearray()
            image = Image.open(io.BytesIO(photo_bytes))
            
            await update.message.reply_text("រង់ចាំបន្តិចសិនណាកូនប្រុស កូនស្រី😁...")
            extracted_text = self.extract_text_universal(image)
            
            if extracted_text and "No readable text" not in extracted_text:
                
                await self._send_long_message(
                    update, 
                    extracted_text, 
                    "📝 **Extracted Text:**\n\n"
                )
                
                await update.message.reply_text("អាចមានខុសបន្តិចបន្តួច ពិនិត្យហើយកែតម្រូវសិនណាកូនប៉ាៗ💋")
            else:
                await update.message.reply_text(
                    "❌ No readable text detected.\n\n"
                    "💡 Try with:\n"
                    "• Higher resolution image\n"
                    "• Better lighting\n"
                    "• Clearer text\n"
                    "• See /tips for more help"
                )
        except Exception as e:
            logger.error(f"Error handling image: {e}")
            await update.message.reply_text("❌ Processing error. Please try again.")

    async def handle_document(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle PDF documents."""
        try:
            document = update.message.document
            
            await update.message.reply_chat_action(action="typing")
            
            pdf_file = await document.get_file()
            pdf_bytes = await pdf_file.download_as_bytearray()
            
            await update.message.reply_text("🌍 Processing PDF with universal OCR...")
            extracted_text = self.extract_text_from_pdf(pdf_bytes)
            
            if extracted_text and "No readable text" not in extracted_text:
                
                await self._send_long_message(
                    update, 
                    extracted_text, 
                    "📝 **Extracted Text:**\n\n"
                )
                
                await update.message.reply_text("✅ PDF processing completed! 📄")
            else:
                await update.message.reply_text("❌ No readable text found in PDF.")
                
        except Exception as e:
            logger.error(f"Error handling document: {e}")
            await update.message.reply_text("❌ Error processing PDF.")

    async def handle_unsupported_document(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle incoming documents that are NOT PDF."""
        await update.message.reply_text(
            "⚠️ **Unsupported Document Type**\n"
            "I only process **PDF** files and **Images** (photos). "
            "Please send your document as an image or a PDF."
        )

    # --- Utility Command Handlers (THESE WERE MISSING, NOW ADDED BACK) ---

    async def start_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Send welcome message when the command /start is issued."""
        welcome_text = """
🌍 **Universal Khmer Text Extractor Bot**

🤖 **Supports ALL types of Khmer content:**
• 📚 Academic papers (Khmer + English)
• 📄 Official documents  
• 📱 Social media text

**Perfect for:** Mixed Khmer-English content, research papers, and documents.

**Commands:**
/start - Show this message
/help - Detailed instructions
/tips - Tips for best results
/languages - Show supported languages

**Just send me any image or PDF with Khmer text!**
        """
        await update.message.reply_text(welcome_text)
    
    async def help_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Send help message when the command /help is issued."""
        help_text = """
📖 **Universal Khmer OCR - How It Works**

This bot uses advanced image pre-processing (OpenCV) and multiple Tesseract configurations to achieve high accuracy on mixed Khmer and English content.

**Technical Features:**
• **Adaptive Pre-processing:** Skew correction (using Tesseract OSD), noise reduction, and advanced contrast enhancement.
• **Weighted OCR:** Runs multiple Tesseract modes and selects the best result based on a quality score.
• **Universal Correction:** Auto-fixes common mixed-script OCR errors (e.g., statistical symbols).

**Best Practices (See /tips for more):**
• High resolution images (1200px+ width)
• Clear, readable text (not handwritten)
• Good lighting and contrast
        """
        await update.message.reply_text(help_text)
        
    async def tips_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Send tips for best OCR results."""
        tips_text = """
🎯 **Tips for Perfect OCR Results:**

**Image Quality:**
• **Resolution:** 1200px+ width recommended
• **Lighting:** Bright, even, no shadows
• **Focus:** Sharp and clear text

**Content Characteristics:**
• **Mixed Content:** Works best with standard fonts.
• **Khmer Script:** Complex characters (subscripts) require high resolution.

**Avoid:**
• Blurry or very low-resolution images
• Handwritten text (recognition is very poor)
• Text on complex, noisy, or patterned backgrounds
        """
        await update.message.reply_text(tips_text)
    
    async def languages_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Show supported languages and capabilities."""
        try:
            # Use Tesseract executable path if set, otherwise rely on PATH
            tess_cmd = [pytesseract.pytesseract.tesseract_cmd, '--list-langs'] if pytesseract.pytesseract.tesseract_cmd else ['tesseract', '--list-langs']
            result = subprocess.run(tess_cmd, capture_output=True, text=True, timeout=10)
            languages = [lang for lang in result.stdout.split('\n') if lang.strip() and not lang.startswith('List')]
            
            langs_text = f"""
🌐 **Supported Languages:** {len(languages)}

**Main Languages:**
• **Khmer (khm)** - Primary, highly-optimized support.
• **English (eng)** - Full support. 
• **Mixed khm+eng** - Optimized for academic and formal documents.

**Full List:** {', '.join(languages[:15])}...
            """
            await update.message.reply_text(langs_text)
        except Exception as e:
            await update.message.reply_text(f"❌ Error checking languages. Is Tesseract installed and configured? Error: {e}")

    # --- Run Function ---

    def run(self):
        """Start the bot."""
        logger.info("Universal Khmer Text Extractor Bot is starting...")
        print("🌍 Universal Khmer Text Extractor Bot is running!")
        print("🎯 Supports: Khmer, English, Mixed content, Academic, Documents, Social media")
        print("📱 Send /start to your bot in Telegram")
        print("⏹️ Press Ctrl+C to stop")
        
        self.application.run_polling(allowed_updates=Update.ALL_TYPES)

def main():
    """Main function to run the bot."""
    print("🤖 Universal Khmer Text Extractor Bot")
    print("=" * 60)
    print("🌐 Universal Features:")
    print("  • Mixed Khmer-English text extraction")
    print("  • All content types supported")
    print("  • Automatic language detection")
    print("  • Universal error correction")
    print("=" * 60)
    
    # Use environment variable first, then prompt
    BOT_TOKEN = os.getenv('TELEGRAM_BOT_TOKEN')
    if not BOT_TOKEN:
        print("\n🔑 Enter your Telegram Bot Token:")
        BOT_TOKEN = input("Bot Token: ").strip()
    
    if not BOT_TOKEN:
        print("❌ No token provided. Exiting.")
        return
    
    bot = UniversalKhmerTextExtractorBot(BOT_TOKEN)
    
    try:
        bot.run()
    except KeyboardInterrupt:
        print("\n👋 Bot stopped")

if __name__ == "__main__":
    main()

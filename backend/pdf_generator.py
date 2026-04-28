"""
PDF Report Generator for Brain Tumor Detection Predictions
Generates professional medical prediction reports
"""

from io import BytesIO
from datetime import datetime
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import mm
from reportlab.lib import colors
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, HRFlowable, Table, TableStyle, Image as RLImage
from reportlab.lib.enums import TA_LEFT, TA_CENTER, TA_RIGHT

# ── Colors ────────────────────────────────────────────────────────────────────
BLACK       = colors.HexColor("#0B1220")
WHITE       = colors.HexColor("#FFFFFF")
TEAL        = colors.HexColor("#2dd4bf")
SKY         = colors.HexColor("#38bdf8")
AMBER       = colors.HexColor("#fbbf24")
GREY_TEXT   = colors.HexColor("#64748b")
GREY_LIGHT  = colors.HexColor("#f1f5f9")
BORDER_LINE = colors.HexColor("#e2e8f0")

# ── Confidence Level Colors ───────────────────────────────────────────────────
CONF_LOW    = colors.HexColor("#ef4444")   # Red
CONF_MED    = colors.HexColor("#f59e0b")   # Amber
CONF_HIGH   = colors.HexColor("#10b981")   # Green


class PredictionReport:
    """Generate medical prediction reports as PDF"""

    def __init__(self):
        self.WIDTH, self.HEIGHT = A4
        self.MARGIN = 15 * mm

    def generate(self, prediction_data: dict) -> bytes:
        """
        Generate PDF report from prediction data
        
        Args:
            prediction_data: Dict with keys:
                - predicted_class: str (e.g., "glioma")
                - confidence: float (0-1)
                - all_predictions: dict (class -> confidence)
                - filename: str
                - timestamp: str (optional)
                - model_accuracy: float (optional)
        
        Returns:
            bytes: PDF content
        """
        pdf_buffer = BytesIO()
        
        doc = SimpleDocTemplate(
            pdf_buffer,
            pagesize=A4,
            leftMargin=self.MARGIN,
            rightMargin=self.MARGIN,
            topMargin=self.MARGIN,
            bottomMargin=self.MARGIN,
        )

        # Build styles
        styles = self._build_styles()
        
        # Build story
        story = []
        story.extend(self._header(styles))
        story.append(self._spacer(10))
        story.extend(self._prediction_result(prediction_data, styles))
        story.append(self._spacer(10))
        story.extend(self._confidence_breakdown(prediction_data, styles))
        story.append(self._spacer(10))
        story.extend(self._report_info(prediction_data, styles))
        
        # Build PDF
        doc.build(story)
        pdf_buffer.seek(0)
        return pdf_buffer.getvalue()

    def _build_styles(self) -> dict:
        """Build all paragraph styles"""
        styles = getSampleStyleSheet()
        
        return {
            'title': ParagraphStyle(
                'Title',
                fontName='Helvetica-Bold',
                fontSize=24,
                textColor=BLACK,
                spaceAfter=4,
                alignment=TA_CENTER
            ),
            'subtitle': ParagraphStyle(
                'Subtitle',
                fontName='Helvetica',
                fontSize=11,
                textColor=GREY_TEXT,
                spaceAfter=12,
                alignment=TA_CENTER
            ),
            'section_title': ParagraphStyle(
                'SectionTitle',
                fontName='Helvetica-Bold',
                fontSize=13,
                textColor=BLACK,
                spaceAfter=8,
                spaceBefore=4
            ),
            'label': ParagraphStyle(
                'Label',
                fontName='Helvetica-Bold',
                fontSize=10,
                textColor=GREY_TEXT,
                spaceAfter=2
            ),
            'value': ParagraphStyle(
                'Value',
                fontName='Helvetica',
                fontSize=11,
                textColor=BLACK,
                spaceAfter=6
            ),
            'body': ParagraphStyle(
                'Body',
                fontName='Helvetica',
                fontSize=10,
                textColor=BLACK,
                spaceAfter=6,
                leading=14
            ),
            'small': ParagraphStyle(
                'Small',
                fontName='Helvetica',
                fontSize=9,
                textColor=GREY_TEXT,
                spaceAfter=4
            ),
            'prediction_class': ParagraphStyle(
                'PredictionClass',
                fontName='Helvetica-Bold',
                fontSize=16,
                textColor=TEAL,
                spaceAfter=4
            ),
            'confidence': ParagraphStyle(
                'Confidence',
                fontName='Helvetica-Bold',
                fontSize=14,
                textColor=BLACK,
                spaceAfter=4
            )
        }

    def _spacer(self, height: float):
        """Create a spacer"""
        return Spacer(1, height)

    def _hr(self, space_before=4, space_after=8):
        """Create horizontal rule"""
        return HRFlowable(
            width="100%",
            thickness=0.5,
            color=BORDER_LINE,
            spaceAfter=space_after,
            spaceBefore=space_before
        )

    def _header(self, styles: dict) -> list:
        """Build report header"""
        story = []
        story.append(Paragraph("Brain Tumor Classification Report", styles['title']))
        story.append(self._hr(space_before=4, space_after=12))
        return story

    def _prediction_result(self, data: dict, styles: dict) -> list:
        """Build main prediction result section"""
        story = []
        
        predicted_class = data.get('predicted_class', 'Unknown').upper()
        confidence = data.get('confidence', 0.0)
        
        # Determine confidence color
        if confidence >= 0.8:
            conf_color = CONF_HIGH
            conf_level = "HIGH"
        elif confidence >= 0.6:
            conf_color = CONF_MED
            conf_level = "MEDIUM"
        else:
            conf_color = CONF_LOW
            conf_level = "LOW"
        
        # Classification result
        story.append(Paragraph("Classification Result", styles['section_title']))
        
        # Create result box
        result_data = [
            [
                Paragraph(f"<b>{predicted_class}</b>", styles['prediction_class']),
                Paragraph(
                    f"<font color='#{conf_color.hexval()}'><b>{conf_level}</b></font><br/>"
                    f"<font size='14'><b>{confidence*100:.1f}%</b> confidence</font>",
                    styles['confidence']
                )
            ]
        ]
        
        result_table = Table(result_data, colWidths=[100*mm, 60*mm])
        result_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, -1), GREY_LIGHT),
            ('BORDER', (0, 0), (-1, -1), 1, BORDER_LINE),
            ('PADDING', (0, 0), (-1, -1), 12),
            ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
            ('ALIGN', (0, 0), (0, -1), 'LEFT'),
            ('ALIGN', (1, 0), (-1, -1), 'CENTER'),
        ]))
        
        story.append(result_table)
        story.append(self._spacer(8))
        
        # Add interpretation
        interpretation = self._get_class_interpretation(predicted_class)
        story.append(Paragraph(f"<font color='#{GREY_TEXT.hexval()}'><i>{interpretation}</i></font>", styles['small']))
        
        return story

    def _confidence_breakdown(self, data: dict, styles: dict) -> list:
        """Build confidence breakdown table"""
        story = []
        
        story.append(self._spacer(4))
        story.append(Paragraph("Prediction Confidence Breakdown", styles['section_title']))
        
        all_predictions = data.get('all_predictions', {})
        
        # Sort by confidence descending
        sorted_preds = sorted(
            all_predictions.items(),
            key=lambda x: x[1],
            reverse=True
        )
        
        # Create table data
        table_data = [['Classification', 'Confidence', 'Progress']]
        
        for class_name, conf in sorted_preds:
            # Create progress bar as ASCII representation
            bar_length = int(conf * 20)  # 20 character scale
            bar = '█' * bar_length + '░' * (20 - bar_length)
            
            table_data.append([
                Paragraph(class_name.title(), styles['value']),
                Paragraph(f"{conf*100:.1f}%", styles['value']),
                Paragraph(f"<font face='Courier' size='9'>{bar}</font>", styles['small'])
            ])
        
        table = Table(table_data, colWidths=[50*mm, 30*mm, 80*mm])
        table.setStyle(TableStyle([
            # Header
            ('BACKGROUND', (0, 0), (-1, 0), TEAL),
            ('TEXT', (0, 0), (-1, 0), 'color', WHITE),
            ('FONT', (0, 0), (-1, 0), 'Helvetica-Bold', 10),
            ('PADDING', (0, 0), (-1, 0), 8),
            
            # Rows
            ('BACKGROUND', (0, 1), (-1, -1), GREY_LIGHT),
            ('GRID', (0, 0), (-1, -1), 0.5, BORDER_LINE),
            ('PADDING', (0, 1), (-1, -1), 10),
            ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
            ('ALIGN', (1, 1), (1, -1), 'CENTER'),
            
            # Alternate row colors
            ('BACKGROUND', (0, 2), (-1, 2), WHITE),
            ('BACKGROUND', (0, 4), (-1, 4), WHITE),
        ]))
        
        story.append(table)
        return story

    def _report_info(self, data: dict, styles: dict) -> list:
        """Build report information footer"""
        story = []
        
        story.append(self._hr(space_before=12, space_after=8))
        story.append(Paragraph("Report Information", styles['section_title']))
        
        # File info
        filename = data.get('filename', 'unknown')
        timestamp = data.get('timestamp', datetime.now().isoformat())
        model_accuracy = data.get('model_accuracy', 0.78)
        
        info_data = [
            ['File Name', filename],
            ['Analysis Time', timestamp],
            ['Model Accuracy', f"{model_accuracy*100:.1f}%"],
            ['System', 'Federated Learning - Brain Tumor Detection'],
        ]
        
        info_table = Table(info_data, colWidths=[50*mm, 100*mm])
        info_table.setStyle(TableStyle([
            ('GRID', (0, 0), (-1, -1), 0.5, BORDER_LINE),
            ('BACKGROUND', (0, 0), (0, -1), GREY_LIGHT),
            ('PADDING', (0, 0), (-1, -1), 8),
            ('FONT', (0, 0), (0, -1), 'Helvetica-Bold', 9),
            ('VALIGN', (0, 0), (-1, -1), 'TOP'),
        ]))
        
        story.append(info_table)
        
        story.append(self._spacer(12))
        story.append(Paragraph(
            "<font size='8' color='#999999'>"
            "This report is generated automatically and should be reviewed by a qualified medical professional. "
            "This is not a substitute for professional medical advice, diagnosis, or treatment."
            "</font>",
            styles['small']
        ))
        
        return story

    def _get_class_interpretation(self, predicted_class: str) -> str:
        """Get interpretation text for predicted class"""
        interpretations = {
            'GLIOMA': 'Glioma detected - A type of brain tumor originating from glial cells.',
            'MENINGIOMA': 'Meningioma detected - A tumor arising from the meninges surrounding the brain.',
            'PITUITARY': 'Pituitary tumor detected - A tumor of the pituitary gland.',
            'NOTUMOR': 'No tumor detected - The scan appears to show normal brain tissue.'
        }
        return interpretations.get(predicted_class, 'Classification complete.')

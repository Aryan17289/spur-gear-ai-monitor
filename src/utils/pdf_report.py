"""PDF report generation utilities"""
import io
from datetime import datetime
from reportlab.lib.pagesizes import A4
from reportlab.lib import colors as rl_colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import cm
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image as RLImage

def make_table(story, data, widths):
    """Create a styled table for PDF report"""
    t = Table(data, colWidths=widths)
    t.setStyle(TableStyle([
        ("BACKGROUND",     (0,0),(-1,0),  rl_colors.HexColor("#0B1628")),
        ("TEXTCOLOR",      (0,0),(-1,0),  rl_colors.white),
        ("FONTNAME",       (0,0),(-1,0),  "Helvetica-Bold"),
        ("FONTSIZE",       (0,0),(-1,-1), 10),
        ("ROWBACKGROUNDS", (0,1),(-1,-1), [rl_colors.HexColor("#f8fafc"), rl_colors.white]),
        ("GRID",           (0,0),(-1,-1), 0.5, rl_colors.HexColor("#e2e8f0")),
        ("PADDING",        (0,0),(-1,-1), 8),
        ("FONTNAME",       (0,1),(-1,-1), "Helvetica"),
    ]))
    story.append(t)

def build_pdf_report(gear_data, prediction_data, shap_fig=None):
    """
    Build comprehensive PDF report
    
    Args:
        gear_data: Dict with gear parameters
        prediction_data: Dict with prediction results
        shap_fig: Optional matplotlib figure for SHAP chart
    """
    buf = io.BytesIO()
    doc = SimpleDocTemplate(buf, pagesize=A4,
                           leftMargin=2*cm, rightMargin=2*cm,
                           topMargin=2*cm, bottomMargin=2*cm)
    
    # Styles
    ss = getSampleStyleSheet()
    title_style = ParagraphStyle("title", parent=ss["Heading1"], 
                                fontSize=18, spaceAfter=4,
                                textColor=rl_colors.HexColor("#0B1628"))
    sub_style = ParagraphStyle("sub", parent=ss["Normal"], 
                              fontSize=10, spaceAfter=16,
                              textColor=rl_colors.HexColor("#5A6A80"))
    body_style = ParagraphStyle("body", parent=ss["Normal"], 
                               fontSize=10, leading=15,
                               textColor=rl_colors.HexColor("#182030"))
    h2_style = ParagraphStyle("h2", parent=ss["Heading2"], 
                             fontSize=13, spaceBefore=16, spaceAfter=6,
                             textColor=rl_colors.HexColor("#0B1628"))
    
    story = []
    
    # Header
    story.append(Paragraph("Spur Gear AI Failure Report", title_style))
    story.append(Paragraph(
        f"Generated: {datetime.now().strftime('%d %B %Y  %H:%M')}  |  "
        f"Gear: {gear_data['gear_type']}", sub_style))
    story.append(Table([[""]], colWidths=[17*cm],
        style=TableStyle([("LINEBELOW",(0,0),(-1,-1),1,rl_colors.HexColor("#e2e8f0"))])))
    story.append(Spacer(1, 10))
    
    # Prediction Summary
    story.append(Paragraph("Prediction Summary", h2_style))
    make_table(story, [
        ["Parameter", "Value"],
        ["Prediction", "FAILURE DETECTED" if prediction_data['prediction']==1 else "NO FAILURE DETECTED"],
        ["Failure Probability", f"{prediction_data['probability_pct']:.1f}%"],
        ["Risk Level", prediction_data['risk_label']],
        ["Gear Type", gear_data['gear_type']],
        ["Health Score", f"{prediction_data['health_score']*100:.1f}%  ({prediction_data['rul_label']})"],
        ["RUL (cycles)", f"{prediction_data['rul_cycles']:,.0f}"],
        ["RUL (time)", f"{prediction_data['rul_hours']:.1f} hrs"],
    ], [6*cm, 11*cm])
    story.append(Spacer(1, 16))
    
    # Operational Parameters
    story.append(Paragraph("Operational Parameters", h2_style))
    param_rows = [["Parameter", "Value", "Unit"]]
    for key, value in gear_data['parameters'].items():
        param_rows.append([key, str(value['value']), value['unit']])
    make_table(story, param_rows, [6*cm, 6*cm, 5*cm])
    story.append(Spacer(1, 16))
    
    # Add SHAP chart if provided
    if shap_fig:
        story.append(Paragraph("SHAP Feature Importance", h2_style))
        ibuf = io.BytesIO()
        shap_fig.savefig(ibuf, format="png", dpi=150, bbox_inches="tight")
        ibuf.seek(0)
        story.append(RLImage(ibuf, width=15*cm, height=7*cm))
        story.append(Spacer(1, 16))
    
    # Build PDF
    doc.build(story)
    buf.seek(0)
    return buf.read()

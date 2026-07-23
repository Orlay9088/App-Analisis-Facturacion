import os
import io
import json
import shutil
import tempfile
import numpy as np
from pathlib import Path
from dotenv import load_dotenv
from fastapi import FastAPI, UploadFile, File, HTTPException, Header, Query
from fastapi.responses import HTMLResponse, FileResponse, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from typing import Optional

from analyzer import process_excel, build_vendedor_prompt, get_table_data


def to_serializable(obj):
    if isinstance(obj, dict):
        return {k: to_serializable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [to_serializable(v) for v in obj]
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj


GEMINI_MODELS = ["gemini-3.1-flash-lite", "gemini-3-flash-preview", "gemini-3.5-flash"]
CLAUDE_MODELS = ["claude-sonnet-4-20250514", "claude-3-5-haiku-20241022"]
OPENAI_MODELS = ["gpt-4o", "gpt-4o-mini"]


def _call_gemini(prompt: str, api_key: str) -> str:
    import google.generativeai as genai
    genai.configure(api_key=api_key)
    last_error = None
    for model_name in GEMINI_MODELS:
        try:
            model = genai.GenerativeModel(model_name)
            response = model.generate_content(prompt)
            return response.text
        except Exception as e:
            last_error = e
            err_str = str(e).lower()
            if "429" in err_str or "quota" in err_str or "rate" in err_str:
                continue
            raise e
    raise Exception(f"Todos los modelos Gemini fallaron. Ultimo error: {last_error}")


def _call_claude(prompt: str, api_key: str) -> str:
    import time
    import anthropic
    client = anthropic.Anthropic(api_key=api_key)
    last_error = None
    for model_name in CLAUDE_MODELS:
        for attempt in range(3):
            try:
                response = client.messages.create(
                    model=model_name,
                    max_tokens=8192,
                    messages=[{"role": "user", "content": prompt}],
                )
                return response.content[0].text
            except Exception as e:
                last_error = e
                err_str = str(e).lower()
                if "429" in err_str or "rate" in err_str or "overloaded" in err_str:
                    if attempt < 2:
                        time.sleep(5 * (attempt + 1))
                        continue
                    else:
                        break
                else:
                    raise e
    raise Exception(f"Todos los modelos Claude fallaron. Ultimo error: {last_error}")


def _call_openai(prompt: str, api_key: str) -> str:
    import time
    from openai import OpenAI
    client = OpenAI(api_key=api_key)
    last_error = None
    for model_name in OPENAI_MODELS:
        for attempt in range(3):
            try:
                response = client.chat.completions.create(
                    model=model_name,
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=8192,
                )
                return response.choices[0].message.content
            except Exception as e:
                last_error = e
                err_str = str(e).lower()
                if "429" in err_str or "rate" in err_str or "quota" in err_str:
                    if attempt < 2:
                        time.sleep(5 * (attempt + 1))
                        continue
                    else:
                        break
                else:
                    raise e
    raise Exception(f"Todos los modelos OpenAI fallaron. Ultimo error: {last_error}")


def _call_ai(prompt: str, api_key: str, provider: str = "gemini") -> str:
    if provider == "claude":
        return _call_claude(prompt, api_key)
    elif provider == "openai":
        return _call_openai(prompt, api_key)
    else:
        return _call_gemini(prompt, api_key)


CHART_COLORS = ['#4F46E5', '#7C3AED', '#2563EB', '#3B82F6', '#8B5CF6',
                '#A78BFA', '#C4B5FD', '#DDD6FE', '#6366F1', '#818CF8']


def _verify_ai_key(api_key: str, provider: str = "gemini") -> dict:
    models_tried = []
    try:
        if provider == "claude":
            import anthropic
            client = anthropic.Anthropic(api_key=api_key)
            response = client.messages.create(
                model=CLAUDE_MODELS[0],
                max_tokens=10,
                messages=[{"role": "user", "content": "Responde solo: OK"}],
            )
            return {"success": True, "model": CLAUDE_MODELS[0], "message": "Conexion exitosa con " + CLAUDE_MODELS[0]}
        elif provider == "openai":
            from openai import OpenAI
            client = OpenAI(api_key=api_key)
            response = client.chat.completions.create(
                model=OPENAI_MODELS[0],
                messages=[{"role": "user", "content": "Responde solo: OK"}],
                max_tokens=10,
            )
            return {"success": True, "model": OPENAI_MODELS[0], "message": "Conexion exitosa con " + OPENAI_MODELS[0]}
        else:
            import google.generativeai as genai
            genai.configure(api_key=api_key)
            last_err = ""
            for model_name in GEMINI_MODELS:
                models_tried.append(model_name)
                try:
                    model = genai.GenerativeModel(model_name)
                    response = model.generate_content("Responde solo: OK")
                    return {"success": True, "model": model_name, "message": "Conexion exitosa con " + model_name}
                except Exception as e:
                    last_err = str(e)
                    continue
            raise Exception(last_err or "Todos los modelos fallaron")
    except Exception as e:
        err_str = str(e).lower()
        if "429" in err_str or "quota" in err_str or "rate" in err_str:
            tried = ", ".join(models_tried) if models_tried else provider
            raise HTTPException(status_code=429, detail=f"Cuota agotada. Modelos intentados: {tried}. Espera o usa otra API key.")
        if "invalid" in err_str or "unauthorized" in err_str or "401" in err_str:
            raise HTTPException(status_code=400, detail="API key invalida para " + provider)
        raise HTTPException(status_code=400, detail=f"Error verificando {provider}: {str(e)}")

load_dotenv()

app = FastAPI(title="Analisis de Facturacion", version="1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

UPLOAD_DIR = Path("uploads")
UPLOAD_DIR.mkdir(exist_ok=True)

current_data = {}


@app.get("/", response_class=HTMLResponse)
async def serve_index():
    html_path = Path("index.html")
    if html_path.exists():
        return HTMLResponse(content=html_path.read_text(encoding="utf-8"))
    return HTMLResponse(content="<h1>index.html no encontrado</h1>", status_code=404)


@app.post("/verify-key")
async def verify_key(
    x_api_key: Optional[str] = Header(None),
    x_provider: Optional[str] = Header(None),
):
    api_key = x_api_key or ""
    provider = x_provider or "gemini"
    if not api_key:
        raise HTTPException(status_code=400, detail="No se proporciono API key.")
    return _verify_ai_key(api_key, provider)


@app.post("/upload")
async def upload_excel(file: UploadFile = File(...)):
    if not file.filename:
        raise HTTPException(status_code=400, detail="No se envio ningun archivo.")

    if not file.filename.endswith((".xlsx", ".xls")):
        raise HTTPException(status_code=400, detail="Formato no valido. Solo se permiten archivos Excel (.xlsx)")

    filepath = UPLOAD_DIR / file.filename
    try:
        with open(filepath, "wb") as f:
            shutil.copyfileobj(file.file, f)
    except PermissionError:
        raise HTTPException(status_code=500, detail="No se pudo guardar el archivo.")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error al guardar el archivo: {str(e)}")

    if filepath.stat().st_size == 0:
        filepath.unlink(missing_ok=True)
        raise HTTPException(status_code=400, detail="El archivo esta vacio.")

    try:
        result = process_excel(str(filepath))
        current_data["_df"] = result.pop("_df", None)
        current_data["result"] = to_serializable(result)
        current_data["filename"] = file.filename
        current_data["filepath"] = str(filepath)
    except ValueError as e:
        filepath.unlink(missing_ok=True)
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        filepath.unlink(missing_ok=True)
        raise HTTPException(status_code=500, detail=f"Error al procesar el archivo: {str(e)}")

    return {
        "success": True,
        "filename": file.filename,
        "total_rows": result["total_rows"],
        "total_unfiltered_rows": result.get("total_unfiltered_rows", result["total_rows"]),
        "dimensions": len(result["active_dimensions"]),
        "date_range": result.get("date_range", {}),
    }


@app.post("/refilter")
async def refilter(filters: str = Query("{}")):
    if not current_data.get("filepath"):
        raise HTTPException(status_code=400, detail="No hay archivo cargado.")
    filepath = current_data["filepath"]
    if not os.path.exists(filepath):
        raise HTTPException(status_code=400, detail="El archivo original ya no existe.")

    try:
        filter_dict = json.loads(filters) if filters else {}
    except json.JSONDecodeError:
        filter_dict = {}

    try:
        result = process_excel(str(filepath), filter_overrides=filter_dict)
        current_data["_df"] = result.pop("_df", None)
        current_data["result"] = to_serializable(result)
        return {
            "success": True,
            "filter_overrides": result["filter_overrides"],
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error al re-filtrar: {str(e)}")


@app.get("/data")
async def get_data():
    if not current_data.get("result"):
        raise HTTPException(status_code=400, detail="No hay datos cargados.")
    return current_data["result"]


@app.get("/table-data")
async def get_table_data_endpoint(
    page: int = Query(1),
    page_size: int = Query(50),
    sort_column: Optional[str] = Query(None),
    sort_ascending: bool = Query(True),
    search: Optional[str] = Query(None),
):
    if current_data.get("_df") is None:
        raise HTTPException(status_code=400, detail="No hay datos cargados.")
    try:
        result = get_table_data(
            current_data["_df"],
            page=page,
            page_size=page_size,
            sort_column=sort_column,
            sort_ascending=sort_ascending,
            search=search,
        )
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error al obtener datos: {str(e)}")


@app.post("/analyze/{dimension_key}/{entity_name}")
async def analyze_entity(
    dimension_key: str,
    entity_name: str,
    x_api_key: Optional[str] = Header(None),
    x_provider: Optional[str] = Header(None),
    filters: Optional[str] = Query("{}"),
):
    if not current_data.get("result"):
        raise HTTPException(status_code=400, detail="No hay datos cargados.")

    api_key = x_api_key or os.getenv("GEMINI_API_KEY", "")
    provider = x_provider or "gemini"
    if not api_key:
        raise HTTPException(status_code=400, detail="No se proporciono API key.")

    result = current_data["result"]
    active_dims = result["active_dimensions"]

    if dimension_key not in active_dims:
        raise HTTPException(status_code=400, detail=f"Dimension '{dimension_key}' no encontrada.")

    dim_data = active_dims[dimension_key]
    entity_data = None
    for m in dim_data["metrics"]:
        if m["nombre"].strip().upper() == entity_name.strip().upper():
            entity_data = m
            break

    if not entity_data:
        raise HTTPException(status_code=404, detail=f"Entidad '{entity_name}' no encontrada en dimension '{dimension_key}'.")

    try:
        filter_dict = json.loads(filters) if filters else {}
    except json.JSONDecodeError:
        filter_dict = {}

    prompt = build_vendedor_prompt(entity_data, dim_data["summary"], filter_dict)

    try:
        informe = _call_ai(prompt, api_key, provider)
    except Exception as e:
        err_str = str(e).lower()
        if "429" in err_str or "quota" in err_str or "rate" in err_str:
            raise HTTPException(status_code=429, detail="Cuota agotada.")
        raise HTTPException(status_code=500, detail=f"Error con {provider}: {str(e)}")

    cache_key = f"report_{dimension_key}_{entity_name.strip().upper()}"
    current_data[cache_key] = informe

    return {
        "dimension": dimension_key,
        "entity": entity_name,
        "informe": informe,
        "metricas": entity_data,
    }


@app.get("/download/{dimension_key}/{entity_name}/word")
async def download_word(
    dimension_key: str,
    entity_name: str,
    filters: Optional[str] = Query("{}"),
):
    if not current_data.get("result"):
        raise HTTPException(status_code=400, detail="No hay datos cargados.")

    result = current_data["result"]
    active_dims = result["active_dimensions"]

    if dimension_key not in active_dims:
        raise HTTPException(status_code=404, detail="Dimension no encontrada.")

    entity_data = None
    for m in active_dims[dimension_key]["metrics"]:
        if m["nombre"].strip().upper() == entity_name.strip().upper():
            entity_data = m
            break

    if not entity_data:
        raise HTTPException(status_code=404, detail="Entidad no encontrada.")

    cache_key = f"report_{dimension_key}_{entity_name.strip().upper()}"
    informe = current_data.get(cache_key, "")
    dim_label = active_dims[dimension_key]["label"]

    try:
        filter_dict = json.loads(filters) if filters else {}
    except json.JSONDecodeError:
        filter_dict = {}

    from docx import Document
    from docx.shared import Inches, Pt, RGBColor
    from docx.enum.text import WD_ALIGN_PARAGRAPH
    from docx.enum.table import WD_TABLE_ALIGNMENT

    doc = Document()

    style = doc.styles['Normal']
    font = style.font
    font.name = 'Calibri'
    font.size = Pt(11)

    title = doc.add_heading('', level=0)
    run = title.add_run(f'Informe de Facturacion - {entity_name}')
    run.font.color.rgb = RGBColor(79, 70, 229)
    run.font.size = Pt(22)

    subtitle = doc.add_paragraph()
    subtitle.alignment = WD_ALIGN_PARAGRAPH.CENTER
    subtitle_text = f'Dimension: {dim_label}'
    if filter_dict:
        subtitle_text += '  |  Filtros: ' + ', '.join(f'{k}={v}' for k, v in filter_dict.items())
    run = subtitle.add_run(subtitle_text)
    run.font.size = Pt(10)
    run.font.color.rgb = RGBColor(100, 116, 139)

    doc.add_paragraph()
    doc.add_heading('Metricas Principales', level=1)
    table = doc.add_table(rows=12, cols=2, style='Light Shading Accent 1')
    table.alignment = WD_TABLE_ALIGNMENT.CENTER

    metrics_data = [
        ('Valor Neto', f"${entity_data['valor_neto']:,.0f}"),
        ('Valor Subtotal Local', f"${entity_data['valor_subtotal']:,.0f}"),
        ('Cantidad Total', f"{entity_data['cantidad']:,.0f} unidades"),
        ('Costo Total', f"${entity_data['costo_total']:,.0f}"),
        ('Costo CIF', f"${entity_data['costo_cif']:,.0f}"),
        ('Utilidad Total', f"${entity_data['utilidad_total']:,.0f}"),
        ('Margen Promedio', f"{entity_data['margen_promedio']:.1f}%"),
        ('Margen Total', f"{entity_data['margen_total']:.1f}%"),
        ('Descuentos', f"${entity_data['valor_descuentos']:,.0f}"),
        ('Utilidad Promedio', f"${entity_data['utilidad_promedio']:,.0f}"),
        ('Documentos Unicos', str(entity_data['documentos_unicos'])),
        ('Total Registros', str(entity_data['total_registros'])),
    ]

    for i, (label, value) in enumerate(metrics_data):
        row = table.rows[i]
        row.cells[0].text = label
        row.cells[1].text = value
        for cell in row.cells:
            for paragraph in cell.paragraphs:
                for run in paragraph.runs:
                    run.font.size = Pt(10)

    if entity_data.get("top_items"):
        doc.add_paragraph()
        doc.add_heading('Top Items/Productos', level=2)
        items_table = doc.add_table(rows=len(entity_data["top_items"]) + 1, cols=4, style='Light List Accent 1')
        for idx, h in enumerate(['Item', 'Valor Neto', 'Cantidad', 'Docs']):
            items_table.rows[0].cells[idx].text = h
            for run in items_table.rows[0].cells[idx].paragraphs[0].runs:
                run.bold = True
        for i, item in enumerate(entity_data["top_items"]):
            row = items_table.rows[i + 1]
            row.cells[0].text = item["item"]
            row.cells[1].text = f"${item['valor_neto']:,.0f}"
            row.cells[2].text = f"{item['cantidad']:,.0f}"
            row.cells[3].text = str(item["registros"])

    if entity_data.get("top_clientes"):
        doc.add_paragraph()
        doc.add_heading('Top Clientes', level=2)
        cl_table = doc.add_table(rows=len(entity_data["top_clientes"]) + 1, cols=4, style='Light List Accent 1')
        for idx, h in enumerate(['Cliente', 'Valor Neto', 'Cantidad', 'Docs']):
            cl_table.rows[0].cells[idx].text = h
            for run in cl_table.rows[0].cells[idx].paragraphs[0].runs:
                run.bold = True
        for i, cl in enumerate(entity_data["top_clientes"]):
            row = cl_table.rows[i + 1]
            row.cells[0].text = cl["cliente"]
            row.cells[1].text = f"${cl['valor_neto']:,.0f}"
            row.cells[2].text = f"{cl['cantidad']:,.0f}"
            row.cells[3].text = str(cl["registros"])

    if entity_data.get("top_ciudades"):
        doc.add_paragraph()
        doc.add_heading('Top Ciudades', level=2)
        ci_table = doc.add_table(rows=len(entity_data["top_ciudades"]) + 1, cols=4, style='Light List Accent 1')
        for idx, h in enumerate(['Ciudad', 'Valor Neto', 'Cantidad', 'Docs']):
            ci_table.rows[0].cells[idx].text = h
            for run in ci_table.rows[0].cells[idx].paragraphs[0].runs:
                run.bold = True
        for i, ci in enumerate(entity_data["top_ciudades"]):
            row = ci_table.rows[i + 1]
            row.cells[0].text = ci["ciudad"]
            row.cells[1].text = f"${ci['valor_neto']:,.0f}"
            row.cells[2].text = f"{ci['cantidad']:,.0f}"
            row.cells[3].text = str(ci["registros"])

    if informe:
        doc.add_page_break()
        doc.add_heading('Informe de IA', level=1)

        lines = informe.split('\n')
        i = 0
        while i < len(lines):
            stripped = lines[i].strip()

            if stripped.startswith('### '):
                doc.add_heading(stripped[4:], level=2)
            elif stripped.startswith('## '):
                doc.add_heading(stripped[3:], level=1)
            elif stripped.startswith('# '):
                doc.add_heading(stripped[2:], level=0)
            elif stripped.startswith('|') and '|' in stripped[1:]:
                table_rows = []
                while i < len(lines) and lines[i].strip().startswith('|'):
                    row_line = lines[i].strip()
                    cells = [c.strip() for c in row_line.split('|')[1:-1]]
                    is_sep = all(c.replace('-', '').replace(':', '').strip() == '' for c in cells)
                    if not is_sep:
                        table_rows.append(cells)
                    i += 1
                i -= 1

                if table_rows:
                    num_cols = max(len(r) for r in table_rows)
                    t = doc.add_table(rows=len(table_rows), cols=num_cols, style='Light Shading Accent 1')
                    t.alignment = WD_TABLE_ALIGNMENT.CENTER
                    for ri, row_data in enumerate(table_rows):
                        for ci, cell_text in enumerate(row_data):
                            if ci < num_cols:
                                t.rows[ri].cells[ci].text = cell_text
                                for paragraph in t.rows[ri].cells[ci].paragraphs:
                                    for run in paragraph.runs:
                                        run.font.size = Pt(10)
                                        if ri == 0:
                                            run.bold = True
            elif stripped.startswith('- '):
                doc.add_paragraph(stripped[2:], style='List Bullet')
            elif stripped:
                p = doc.add_paragraph()
                parts = stripped.split('**')
                for j, part in enumerate(parts):
                    run = p.add_run(part)
                    run.font.size = Pt(11)
                    if j % 2 == 1:
                        run.bold = True

            i += 1

    buffer = io.BytesIO()
    doc.save(buffer)
    buffer.seek(0)

    safe_name = entity_name.replace(' ', '_').replace('.', '').replace('/', '_')
    return StreamingResponse(
        buffer,
        media_type='application/vnd.openxmlformats-officedocument.wordprocessingml.document',
        headers={'Content-Disposition': f'attachment; filename="Informe_{safe_name}.docx"'}
    )


@app.get("/download/{dimension_key}/{entity_name}/excel")
async def download_excel(
    dimension_key: str,
    entity_name: str,
    filters: Optional[str] = Query("{}"),
):
    if not current_data.get("result"):
        raise HTTPException(status_code=400, detail="No hay datos cargados.")

    result = current_data["result"]
    active_dims = result["active_dimensions"]

    if dimension_key not in active_dims:
        raise HTTPException(status_code=404, detail="Dimension no encontrada.")

    entity_data = None
    for m in active_dims[dimension_key]["metrics"]:
        if m["nombre"].strip().upper() == entity_name.strip().upper():
            entity_data = m
            break

    if not entity_data:
        raise HTTPException(status_code=404, detail="Entidad no encontrada.")

    import openpyxl
    from openpyxl.styles import Font, PatternFill, Alignment, Border, Side

    wb = openpyxl.Workbook()

    header_font = Font(name='Calibri', bold=True, size=12, color='FFFFFF')
    header_fill = PatternFill(start_color='4F46E5', end_color='4F46E5', fill_type='solid')
    title_font = Font(name='Calibri', bold=True, size=16, color='4F46E5')
    subtitle_font = Font(name='Calibri', bold=True, size=10, color='64748B')
    metric_label_font = Font(name='Calibri', bold=True, size=11)
    metric_value_font = Font(name='Calibri', size=11)
    thin_border = Border(
        left=Side(style='thin'), right=Side(style='thin'),
        top=Side(style='thin'), bottom=Side(style='thin')
    )

    ws = wb.active
    ws.title = 'Metricas'

    ws.merge_cells('A1:D1')
    ws['A1'] = f'Informe de Facturacion - {entity_name}'
    ws['A1'].font = title_font
    ws['A1'].alignment = Alignment(horizontal='center')

    dim_label = active_dims[dimension_key]["label"]
    ws.merge_cells('A2:D2')
    ws['A2'] = f'Dimension: {dim_label}'
    ws['A2'].font = subtitle_font
    ws['A2'].alignment = Alignment(horizontal='center')

    ws['A4'] = 'Metrica'
    ws['B4'] = 'Valor'
    ws['A4'].font = header_font
    ws['A4'].fill = header_fill
    ws['B4'].font = header_font
    ws['B4'].fill = header_fill
    ws['A4'].border = thin_border
    ws['B4'].border = thin_border

    metrics_data = [
        ('Valor Neto', entity_data['valor_neto']),
        ('Valor Subtotal Local', entity_data['valor_subtotal']),
        ('Cantidad Total', entity_data['cantidad']),
        ('Costo Total', entity_data['costo_total']),
        ('Costo CIF', entity_data['costo_cif']),
        ('Utilidad Total', entity_data['utilidad_total']),
        ('Margen Promedio (%)', entity_data['margen_promedio']),
        ('Margen Total (%)', entity_data['margen_total']),
        ('Descuentos Totales', entity_data['valor_descuentos']),
        ('Utilidad Promedio', entity_data['utilidad_promedio']),
        ('Costo Uni. Promedio', entity_data['costo_uni_promedio']),
        ('Precio Unit. Promedio', entity_data['precio_unit_promedio']),
        ('Documentos Unicos', entity_data['documentos_unicos']),
        ('Total Registros', entity_data['total_registros']),
    ]

    for i, (label, value) in enumerate(metrics_data):
        row = i + 5
        ws.cell(row=row, column=1, value=label).font = metric_label_font
        cell = ws.cell(row=row, column=2, value=value)
        cell.font = metric_value_font
        cell.number_format = '#,##0'
        ws.cell(row=row, column=1).border = thin_border
        ws.cell(row=row, column=2).border = thin_border

    ws.column_dimensions['A'].width = 30
    ws.column_dimensions['B'].width = 20

    if entity_data.get("top_items"):
        ws2 = wb.create_sheet('Top Items')
        headers2 = ['Item', 'Valor Neto', 'Cantidad', 'Docs']
        for j, h in enumerate(headers2):
            cell = ws2.cell(row=1, column=j + 1, value=h)
            cell.font = header_font
            cell.fill = header_fill
        for i, item in enumerate(entity_data["top_items"]):
            ws2.cell(row=i + 2, column=1, value=item["item"])
            ws2.cell(row=i + 2, column=2, value=item["valor_neto"])
            ws2.cell(row=i + 2, column=3, value=item["cantidad"])
            ws2.cell(row=i + 2, column=4, value=item["registros"])
        ws2.column_dimensions['A'].width = 50
        ws2.column_dimensions['B'].width = 18
        ws2.column_dimensions['C'].width = 12
        ws2.column_dimensions['D'].width = 10

    if entity_data.get("top_clientes"):
        ws3 = wb.create_sheet('Top Clientes')
        headers3 = ['Cliente', 'Valor Neto', 'Cantidad', 'Docs']
        for j, h in enumerate(headers3):
            cell = ws3.cell(row=1, column=j + 1, value=h)
            cell.font = header_font
            cell.fill = header_fill
        for i, cl in enumerate(entity_data["top_clientes"]):
            ws3.cell(row=i + 2, column=1, value=cl["cliente"])
            ws3.cell(row=i + 2, column=2, value=cl["valor_neto"])
            ws3.cell(row=i + 2, column=3, value=cl["cantidad"])
            ws3.cell(row=i + 2, column=4, value=cl["registros"])
        ws3.column_dimensions['A'].width = 50
        ws3.column_dimensions['B'].width = 18
        ws3.column_dimensions['C'].width = 12
        ws3.column_dimensions['D'].width = 10

    if entity_data.get("top_ciudades"):
        ws4 = wb.create_sheet('Top Ciudades')
        headers4 = ['Ciudad', 'Valor Neto', 'Cantidad', 'Docs']
        for j, h in enumerate(headers4):
            cell = ws4.cell(row=1, column=j + 1, value=h)
            cell.font = header_font
            cell.fill = header_fill
        for i, ci in enumerate(entity_data["top_ciudades"]):
            ws4.cell(row=i + 2, column=1, value=ci["ciudad"])
            ws4.cell(row=i + 2, column=2, value=ci["valor_neto"])
            ws4.cell(row=i + 2, column=3, value=ci["cantidad"])
            ws4.cell(row=i + 2, column=4, value=ci["registros"])
        ws4.column_dimensions['A'].width = 30
        ws4.column_dimensions['B'].width = 18
        ws4.column_dimensions['C'].width = 12
        ws4.column_dimensions['D'].width = 10

    if entity_data.get("estado_dist"):
        ws5 = wb.create_sheet('Estados')
        headers5 = ['Estado', 'Docs', 'Valor Neto', 'Cantidad']
        for j, h in enumerate(headers5):
            cell = ws5.cell(row=1, column=j + 1, value=h)
            cell.font = header_font
            cell.fill = header_fill
        for i, (estado, d) in enumerate(entity_data["estado_dist"].items()):
            ws5.cell(row=i + 2, column=1, value=estado)
            ws5.cell(row=i + 2, column=2, value=d["registros"])
            ws5.cell(row=i + 2, column=3, value=d["valor_neto"])
            ws5.cell(row=i + 2, column=4, value=d["cantidad"])
        ws5.column_dimensions['A'].width = 25
        ws5.column_dimensions['B'].width = 10
        ws5.column_dimensions['C'].width = 18
        ws5.column_dimensions['D'].width = 12

    cache_key = f"report_{dimension_key}_{entity_name.strip().upper()}"
    informe = current_data.get(cache_key, "")
    if informe:
        ws_report = wb.create_sheet('Informe IA')
        lines = informe.split('\n')
        row_num = 1
        i = 0
        while i < len(lines):
            stripped = lines[i].strip()
            if stripped.startswith('# '):
                ws_report.cell(row=row_num, column=1, value=stripped[2:]).font = Font(name='Calibri', bold=True, size=16, color='4F46E5')
                ws_report.merge_cells(f'A{row_num}:F{row_num}')
                row_num += 1
            elif stripped.startswith('## '):
                ws_report.cell(row=row_num, column=1, value=stripped[3:]).font = Font(name='Calibri', bold=True, size=13, color='4F46E5')
                ws_report.merge_cells(f'A{row_num}:F{row_num}')
                row_num += 1
            elif stripped.startswith('### '):
                ws_report.cell(row=row_num, column=1, value=stripped[4:]).font = Font(name='Calibri', bold=True, size=11, color='334155')
                ws_report.merge_cells(f'A{row_num}:F{row_num}')
                row_num += 1
            elif stripped.startswith('|') and '|' in stripped[1:]:
                table_rows = []
                while i < len(lines) and lines[i].strip().startswith('|'):
                    row_line = lines[i].strip()
                    cells = [c.strip() for c in row_line.split('|')[1:-1]]
                    is_sep = all(c.replace('-', '').replace(':', '').strip() == '' for c in cells)
                    if not is_sep:
                        table_rows.append(cells)
                    i += 1
                i -= 1
                if table_rows:
                    num_cols = max(len(r) for r in table_rows)
                    for ri, row_data in enumerate(table_rows):
                        for ci, cell_text in enumerate(row_data):
                            if ci < num_cols:
                                cell = ws_report.cell(row=row_num, column=ci + 1, value=cell_text)
                                if ri == 0:
                                    cell.font = header_font
                                    cell.fill = header_fill
                                else:
                                    cell.font = Font(name='Calibri', size=10)
                                cell.border = thin_border
                                cell.alignment = Alignment(wrap_text=True)
                        row_num += 1
                    row_num += 1
            elif stripped.startswith('- '):
                ws_report.cell(row=row_num, column=1, value='  *  ' + stripped[2:]).font = Font(name='Calibri', size=10)
                row_num += 1
            elif stripped:
                ws_report.cell(row=row_num, column=1, value=stripped).font = Font(name='Calibri', size=10)
                row_num += 1
            i += 1

        ws_report.column_dimensions['A'].width = 30
        ws_report.column_dimensions['B'].width = 20
        ws_report.column_dimensions['C'].width = 20

    buffer = io.BytesIO()
    wb.save(buffer)
    buffer.seek(0)

    safe_name = entity_name.replace(' ', '_').replace('.', '').replace('/', '_')
    return StreamingResponse(
        buffer,
        media_type='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
        headers={'Content-Disposition': f'attachment; filename="Informe_{safe_name}.xlsx"'}
    )


if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8002))
    uvicorn.run(app, host="0.0.0.0", port=port)

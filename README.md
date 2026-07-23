# Analisis de Facturacion

Dashboard web para analizar facturacion por dimension (vendedor, linea, grupo, etc.), generar informes con IA y exportar reportes en Word/Excel.

## Ejecutar en local

```bash
pip install -r requirements.txt
```

Crear archivo `.env` con tu API key de Gemini:
```
GEMINI_API_KEY=tu_api_key_aqui
```

Ejecutar:
```bash
uvicorn app:app --reload --port 8000
```

Abrir http://127.0.0.1:8000

## Deploy en Render

1. Crear cuenta en [render.com](https://render.com)
2. Conectar el repositorio de GitHub
3. Configurar la variable de entorno `GEMINI_API_KEY` en el dashboard de Render
4. Deploy automatico al hacer push

## Funcionalidades

- Carga archivos Excel de facturacion (43 columnas)
- Filtros dinamicos: Canal Distribucion, Linea, Grupo, Sub-Linea, Estado, Categoria, Canal, Vendedor, Ciudad, Pais, etc.
- Dimensiones de analisis: Vendedor, Linea, Grupo, Sub-Linea, Canal Distribucion, Unidad Negocio, Categoria, Canal, Ciudad, Pais, Cliente, Estado, Tipo Cliente
- Metricas: Valor neto, cantidad, costo, utilidad, margen, descuentos, documentos
- Generacion de informes con Gemini/Claude/ChatGPT AI
- Exportacion a Word (.docx) y Excel (.xlsx)

FROM python:3.11-slim

WORKDIR /app
COPY pyproject.toml README.md ./
COPY src ./src
COPY fixtures ./fixtures
RUN pip install --no-cache-dir -e ".[pg]"

EXPOSE 8000
CMD ["uvicorn", "edgar_analyst.api.main:app", "--host", "0.0.0.0", "--port", "8000"]

FROM python:3.11-slim-bullseye

# Install ngspice for PySpice native backend
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        ngspice libngspice0 libngspice0-dev \
    && rm -rf /var/lib/apt/lists/*

ENV SPICE_BACKEND=ngspice

# Create user for Hugging Face Spaces
RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH

WORKDIR $HOME/app

# Suppress "Note: No compatibility mode selected!" from newer ngspice 
# which breaks PySpice's subprocess parser
RUN echo "set compatmacs" > $HOME/.spiceinit

# Install Python dependencies
COPY --chown=user:user requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy project files
COPY --chown=user:user . .

# Create sim output directory
RUN mkdir -p sim_output

EXPOSE 7860

# Health check
HEALTHCHECK --interval=30s --timeout=3s --start-period=5s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:7860/health')" || exit 1

CMD ["uvicorn", "server.app:app", "--host", "0.0.0.0", "--port", "7860"]

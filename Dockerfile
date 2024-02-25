FROM python:3.10-slim

RUN apt-get -qq update \
    && apt-get -qq -y install awscli tree vim curl \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get -qq -y clean

ENV \
  PYTHONUNBUFFERED=1 \ 
  PYTHONDONTWRITEBYTECODE=1 \
  DEBIAN_FRONTEND=noninteractive \
  PIP_NO_CACHE_DIR=off \
  VIRTUAL_ENV=/app/.venv
ENV \
  PATH="${VIRTUAL_ENV}/bin:${PATH}" \
  POETRY_VERSION=1.7.1 \
  POETRY_NO_INTERACTION=1 \
  POETRY_VIRTUALENVS_CREATE=false
   
RUN pip install -U pip poetry==${POETRY_VERSION}

RUN mkdir app 

RUN python3 -m venv ${VIRTUAL_ENV} \
    && . ${VIRTUAL_ENV}/bin/activate

WORKDIR /app

COPY . ./

RUN poetry install --without dev

EXPOSE 8501

HEALTHCHECK CMD curl --fail https://localhost:8501/_store/health

ENTRYPOINT ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
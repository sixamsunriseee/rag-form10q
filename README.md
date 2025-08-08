# RAG System for FORM-10Q SEC files
Repository serves as RAG (Retrieval Augmented Generation) QA system.
Deliverable is a composed microservice of [FastAPI](https://github.com/fastapi/fastapi) endpoint
and [Qdrant](https://qdrant.tech/) database. Application architecture is referenced [here](docs/architecutre.md).

## Installation

### Development version:
- Clone repository:
`git clone https://github.com/sixamsunriseee/rag-form10q.git`

- Sync virtual environment with [uv](https://github.com/astral-sh/uv):
`uv sync`

### Docker:
- Build image from root directory: 
`docker build -t IMAGE_NAME .`

- Run docker image:
`docker run -d -p 8000:8000 IMAGE_NAME`
## 

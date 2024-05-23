from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import os
import tempfile
from uuid import uuid4
from dotenv import load_dotenv
import os

load_dotenv()  # .envファイルから環境変数を読み込む

API_KEY = os.getenv("COHERE_API_KEY")

# llama_index関連のインポート
from llama_index.core import Settings, VectorStoreIndex, SimpleDirectoryReader, PromptTemplate
from llama_index.llms.cohere import Cohere
from llama_index.embeddings.cohere import CohereEmbedding
from llama_index.postprocessor.cohere_rerank import CohereRerank

from pydantic import BaseModel

class QueryRequest(BaseModel):
    prompt: str
    file_key: str

app = FastAPI()

# フロントエンドのオリジンを指定
origins = [
    "http://localhost:3000",  # Reactアプリが動作しているオリジン
    # 必要に応じて他のオリジンも追加
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # すべてのオリジンを許可
    allow_credentials=True,
    allow_methods=["GET", "POST"],  # 許可するHTTPメソッド
    allow_headers=["Content-Type"],  # 許可するヘッダー
)

# セッションとファイルキャッシュの管理
session_state = {
    "id": str(uuid4()),
    "file_cache": {}
}

def setup_llama_index(file_path: str, file_key: str):
    loader = SimpleDirectoryReader(
        input_dir=file_path,
        required_exts=[".pdf"],
        recursive=True
    )
    docs = loader.load_data()

    # LLM & embedding modelのセットアップ
    llm = Cohere(api_key=API_KEY, model="command-r-plus")
    embed_model = CohereEmbedding(
        cohere_api_key=API_KEY,
        model_name="embed-english-v3.0",
        input_type="search_query",
    )
    cohere_rerank = CohereRerank(
        model='rerank-english-v3.0',
        api_key=API_KEY,
    )

    # ドキュメントからインデックスを作成
    Settings.embed_model = embed_model
    index = VectorStoreIndex.from_documents(docs, show_progress=True)

    # クエリエンジンの作成
    Settings.llm = llm
    query_engine = index.as_query_engine(streaming=True, node_postprocessors=[cohere_rerank])
    # ====== Customise prompt template ======
    qa_prompt_tmpl_str = (
        "Context information is below.\n"
        "---------------------\n"
        "{context_str}\n"
        "---------------------\n"
        "Given the context information above I want you to think step by step to answer the query in a crisp manner, in case you don't know the answer say 'I don't know!'.\n"
        "Query: {query_str}\n"
        "Answer: "
    )
    qa_prompt_tmpl = PromptTemplate(qa_prompt_tmpl_str)

    query_engine.update_prompts(
        {"response_synthesizer:text_qa_template": qa_prompt_tmpl}
    )

    session_state['file_cache'][file_key] = query_engine

    return query_engine

@app.post("/upload/")
async def upload_file(file: UploadFile = File(...) ):
    try:
        with tempfile.TemporaryDirectory() as temp_dir:
            file_path = os.path.join(temp_dir, file.filename)
            
            with open(file_path, "wb") as f:
                f.write(await file.read())
            
            file_key = f"{session_state['id']}-{file.filename}"
            print(file_key)
            
            if file_key not in session_state['file_cache']:
                query_engine = setup_llama_index(temp_dir, file_key)
                session_state['file_cache'][file_key] = query_engine
            return {"id": session_state['id']}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/query/")
async def query(request: QueryRequest):
    try:
        print(request.prompt)
        query_engine = session_state['file_cache'].get(request.file_key)
        if not query_engine:
            raise HTTPException(status_code=404, detail="Query engine not found.")
        
        response = query_engine.query(request.prompt)

        response = str(response)
        return {"message": response}
    except Exception as e:
        print(e)
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)


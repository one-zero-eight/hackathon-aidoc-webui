import requests
import logging
import ftfy
import pymupdf4llm

from langchain_community.document_loaders import (
    BSHTMLLoader,
    CSVLoader,
    Docx2txtLoader,
    OutlookMessageLoader,
    TextLoader,
    UnstructuredEPubLoader,
    UnstructuredExcelLoader,
    UnstructuredMarkdownLoader,
    UnstructuredPowerPointLoader,
    UnstructuredRSTLoader,
    UnstructuredXMLLoader,
)
from langchain_core.documents import Document
from openai import OpenAI

from open_webui.config import OPENAI_API_BASE_URL, OPENAI_API_KEY
from open_webui.env import SRC_LOG_LEVELS

log = logging.getLogger(__name__)
log.setLevel(SRC_LOG_LEVELS["RAG"])

known_source_ext = [
    "go",
    "py",
    "java",
    "sh",
    "bat",
    "ps1",
    "cmd",
    "js",
    "ts",
    "css",
    "cpp",
    "hpp",
    "h",
    "c",
    "cs",
    "sql",
    "log",
    "ini",
    "pl",
    "pm",
    "r",
    "dart",
    "dockerfile",
    "env",
    "php",
    "hs",
    "hsc",
    "lua",
    "nginxconf",
    "conf",
    "m",
    "mm",
    "plsql",
    "perl",
    "rb",
    "rs",
    "db2",
    "scala",
    "bash",
    "swift",
    "vue",
    "svelte",
    "msg",
    "ex",
    "exs",
    "erl",
    "tsx",
    "jsx",
    "hs",
    "lhs",
]


class TikaLoader:
    def __init__(self, url, file_path, mime_type=None):
        self.url = url
        self.file_path = file_path
        self.mime_type = mime_type

    def load(self) -> list[Document]:
        with open(self.file_path, "rb") as f:
            data = f.read()

        if self.mime_type is not None:
            headers = {"Content-Type": self.mime_type}
        else:
            headers = {}

        endpoint = self.url
        if not endpoint.endswith("/"):
            endpoint += "/"
        endpoint += "tika/text"

        r = requests.put(endpoint, data=data, headers=headers)

        if r.ok:
            raw_metadata = r.json()
            text = raw_metadata.get("X-TIKA:content", "<No text content found>")

            if "Content-Type" in raw_metadata:
                headers["Content-Type"] = raw_metadata["Content-Type"]

            log.info("Tika extracted text: %s", text)

            return [Document(page_content=text, metadata=headers)]
        else:
            raise Exception(f"Error calling Tika: {r.reason}")


class Pdf4LlmLoader:
    """
    title: Make charts out of your data v2
    author: Iqbal Maulana
    author_url: https://github.com/iqballx?tab=repositories
    author_linkedin: https://www.linkedin.com/in/iqbaalm/
    funding_url: https://github.com/open-webui
    version: 2.0.0
    """

    def __init__(self, file_path, extract_images: bool = False, mime_type=None):
        self.file_path = file_path
        self.mime_type = mime_type
        self.extract_images = extract_images
        self.openai = OpenAI(api_key=OPENAI_API_KEY, base_url=OPENAI_API_BASE_URL)

    def load(self) -> list[Document]:
        # noinspection PyTypeChecker
        _: list[dict] = pymupdf4llm.to_markdown(self.file_path, page_chunks=True, graphics_limit=1000,
                                                write_images=self.extract_images, show_progress=False)

        return [Document(
            page_content=page["text"], metadata={
                **page["metadata"],
                "toc_items": page["toc_items"],
                "tables": page["tables"],
                "images": page["images"],
                "graphics": page["graphics"],
                "words": page["words"],
            }
        ) for page in _]


class Loader:
    def __init__(self, engine: str = "", **kwargs):
        self.engine = engine
        self.kwargs = kwargs

    def load(
            self, filename: str, file_content_type: str, file_path: str
    ) -> list[Document]:
        loader = self._get_loader(filename, file_content_type, file_path)
        docs = loader.load()

        return [
            Document(
                page_content=ftfy.fix_text(doc.page_content), metadata=doc.metadata
            )
            for doc in docs
        ]

    def _get_loader(self, filename: str, file_content_type: str, file_path: str):
        file_ext = filename.split(".")[-1].lower()

        if self.engine == "tika" and self.kwargs.get("TIKA_SERVER_URL"):
            if file_ext in known_source_ext or (
                    file_content_type and file_content_type.find("text/") >= 0
            ):
                loader = TextLoader(file_path, autodetect_encoding=True)
            else:
                loader = TikaLoader(
                    url=self.kwargs.get("TIKA_SERVER_URL"),
                    file_path=file_path,
                    mime_type=file_content_type,
                )
        else:
            if file_ext == "pdf":
                loader = Pdf4LlmLoader(
                    file_path, extract_images=self.kwargs.get("PDF_EXTRACT_IMAGES")
                )
            elif file_ext == "csv":
                loader = CSVLoader(file_path)
            elif file_ext == "rst":
                loader = UnstructuredRSTLoader(file_path, mode="elements")
            elif file_ext == "xml":
                loader = UnstructuredXMLLoader(file_path)
            elif file_ext in ["htm", "html"]:
                loader = BSHTMLLoader(file_path, open_encoding="unicode_escape")
            elif file_ext == "md":
                loader = UnstructuredMarkdownLoader(file_path)
            elif file_content_type == "application/epub+zip":
                loader = UnstructuredEPubLoader(file_path)
            elif (
                    file_content_type
                    == "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
                    or file_ext == "docx"
            ):
                loader = Docx2txtLoader(file_path)
            elif file_content_type in [
                "application/vnd.ms-excel",
                "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            ] or file_ext in ["xls", "xlsx"]:
                loader = UnstructuredExcelLoader(file_path)
            elif file_content_type in [
                "application/vnd.ms-powerpoint",
                "application/vnd.openxmlformats-officedocument.presentationml.presentation",
            ] or file_ext in ["ppt", "pptx"]:
                loader = UnstructuredPowerPointLoader(file_path)
            elif file_ext == "msg":
                loader = OutlookMessageLoader(file_path)
            elif file_ext in known_source_ext or (
                    file_content_type and file_content_type.find("text/") >= 0
            ):
                loader = TextLoader(file_path, autodetect_encoding=True)
            else:
                loader = TextLoader(file_path, autodetect_encoding=True)

        return loader

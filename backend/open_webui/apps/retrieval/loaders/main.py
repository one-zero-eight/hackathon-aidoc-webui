import json
import logging
import re
from io import StringIO

import camelot
import ftfy
import pandas as pd
import pymupdf
import pymupdf4llm
import requests
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
from tables_extraction.main import extract_tables

from open_webui.apps.webui.models.files import Files
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


SYSTEM_PROMPT_CONVERT_TO_TABLES = """
Objective:
Your goal is to read the tables in markdown format and return the data in CSV format.
"""

USER_PROMPT_CONVERT_TO_TABLES = """
Giving this markdown input: {Query}.
Generate tables in CSV format separated by "\\n\\n".
"""


class Pdf4LlmLoader:
    def __init__(self, file_path, extract_images: bool = False):
        self.file_path = file_path
        self.extract_images = extract_images

    @staticmethod
    def format_cell(x):
        if isinstance(x, str):
            return re.sub(r'\s{2,}', ' ', x.replace("\n", "")).strip()
        else:
            return x

    @staticmethod
    def merge_dfs(dfs):
        merged = []
        last = None
        was_numerated = False

        for i, df in enumerate(dfs):
            # Convert first column to integer list, removing any trailing periods in strings
            first_col_as_ints = (df.iloc[:, 0]
                                 .astype(str)
                                 .str
                                 .replace(r'\.$', '', regex=True)
                                 .astype(int, errors="ignore")
                                 .tolist())

            # Check if the current DataFrame's first column is sequentially numbered
            is_numerated = first_col_as_ints == list(range(1, len(first_col_as_ints) + 1))

            # For the first DataFrame, set the initial numbering status
            if i == 0:
                was_numerated = is_numerated
                last = first_col_as_ints[-1] if is_numerated else None
                merged.append(df)
                continue

            # Check if current DataFrame continues from the previous one
            if was_numerated and first_col_as_ints == list(range(last + 1, last + len(first_col_as_ints) + 1)):
                # Merge with the last DataFrame in `merged`
                merged[-1] = pd.concat([merged[-1], df], ignore_index=True)
                last = first_col_as_ints[-1]
            else:
                # Start a new group if not sequential
                merged.append(df)
                was_numerated = is_numerated
                last = first_col_as_ints[-1] if is_numerated else None
        output = []

        for df in merged:
            df = df.replace("", float("NaN"))
            df = df.dropna(how="all", axis=1)
            output.append(df)

        return merged

    def dfs_to_csvs_pipeline(self, dfs):
        _dfs = []
        for df in dfs:
            first_row = df.iloc[0, :].tolist()

            # Check if the first row matches [1, 2, 3, ..., N] format, even if they are strings
            if first_row == list(map(str, range(1, len(first_row) + 1))) or first_row == list(
                    range(1, len(first_row) + 1)):
                df = df.drop(index=0)  # Drop the first row if it matches the format

            df = df.replace("", float("NaN"))
            df = df.dropna(how="all")
            df = df.map(self.format_cell)
            _dfs.append(df)

        _dfs = self.merge_dfs(_dfs)

        tables_csvs = []
        for df in _dfs:
            csv_io = StringIO()
            df.to_csv(csv_io, index=False, header=False)
            csv_io.seek(0)
            tables_csvs.append(csv_io.read())
        return tables_csvs

    def load(self) -> list[Document]:
        is_empty_or_scan = True
        with pymupdf.open(self.file_path) as doc:
            for page in doc:
                if page.get_text():
                    is_empty_or_scan = False
                    break

        if is_empty_or_scan:
            tables_dfs, texts = extract_tables(self.file_path)
            tables_csvs = self.dfs_to_csvs_pipeline(filter(lambda x: x is not None, tables_dfs))
            docs = [Document(page_content=text) for text in texts if text]
        else:
            # noinspection PyTypeChecker
            pages: list[dict] = pymupdf4llm.to_markdown(self.file_path, page_chunks=True, graphics_limit=1000,
                                                        write_images=self.extract_images, show_progress=False)
            tables = camelot.read_pdf(self.file_path, pages="all", backend="poppler")
            tables_csvs = self.dfs_to_csvs_pipeline(tables._tables)
            docs = [
                Document(page_content=page["text"], metadata=page["metadata"] or {}) for page in pages
                if page["text"]
            ]
        file = Files.get_file_by_path(self.file_path)
        meta = file.meta or {}
        meta["csvs"] = json.dumps(tables_csvs, ensure_ascii=False)
        Files.update_file_metadata_by_id(file.id, meta=meta)

        return docs


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
                loader = Pdf4LlmLoader(file_path)
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

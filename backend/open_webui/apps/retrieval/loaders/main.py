import logging
import math
from io import StringIO
from typing import Tuple, Union

import cv2
import ftfy
import numpy as np
import pymupdf4llm
import requests
from deskew import determine_skew
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
from pandas import DataFrame

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


class Deskewer:
    def rotate(
            self,
            gray_scaled_image: np.ndarray,
            angle: float,
            background: Union[int, Tuple[int, int, int]]
    ) -> np.ndarray:
        old_width, old_height = gray_scaled_image.shape[:2]
        angle_radian = math.radians(angle)
        width = abs(np.sin(angle_radian) * old_height) + abs(np.cos(angle_radian) * old_width)
        height = abs(np.sin(angle_radian) * old_width) + abs(np.cos(angle_radian) * old_height)

        image_center = tuple(np.array(gray_scaled_image.shape[1::-1]) / 2)
        rot_mat = cv2.getRotationMatrix2D(image_center, angle, 1.0)
        rot_mat[1, 2] += (width - old_width) / 2
        rot_mat[0, 2] += (height - old_height) / 2
        return cv2.warpAffine(gray_scaled_image, rot_mat, (int(round(height)), int(round(width))),
                              borderValue=background)

    def deskew(self, image: np.ndarray) -> np.ndarray:
        grayscale = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        angle = determine_skew(grayscale)
        rotated = self.rotate(image, angle, (255, 255, 255))
        return rotated


DESKEWER = Deskewer()

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

    def load(self) -> list[Document]:
        # noinspection PyTypeChecker
        pages: list[dict] = pymupdf4llm.to_markdown(self.file_path, page_chunks=True, graphics_limit=1000,
                                                    write_images=self.extract_images, show_progress=False)

        file = Files.get_file_by_path(self.file_path)
        print("!!!!", file)
        df = DataFrame(
            data=[
                [1, "Выручка, млрд. руб.", "Строка «Выручка» консолидированных отчетов о прибылях и убытках", 281.6,
                 332.2],
                [2, "Операционная прибыль до вычета износа основных средств и амортизации нематериальных активов ("
                    "OIBDA), млрд. руб.", "Сумма строк «Операционная прибыль» и «Амортизация основных средств и "
                                          "нематериальных активов» консолидированных отчетов о прибылях и убытках",
                 118.7, 124.5],
                [3, "Рентабельность по OIBDA (OIBDA margin), %", "Отношение показателя OIBDA к выручке", "42,2%",
                 "37,5%"],
                [4, "Скорректированная OIBDA, млрд. руб.",
                 "Сумма операционной прибыли до вычета износа основных средств и амортизации нематериальных активов ("
                 "OIBDA) и строки «Резерв под обесценение внеоборотных активов» консолидированных отчетов о прибылях "
                 "и убытках",
                 118.7,
                 124.5]
            ],
            headers=["Nп/п", "Наименование", "Методика расчета показателя", "6 мес. 2023", "6 мес. 2024"])
        meta = file.meta or {}
        csv_io = StringIO()
        df.to_csv(csv_io)
        meta["csvs"] = [csv_io.read()]
        Files.update_file_metadata_by_id(file.id, meta=meta)

        return [
            Document(page_content=page["text"], metadata=page["metadata"]) for page in pages
        ]


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

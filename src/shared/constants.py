"""Application-wide constants."""

from __future__ import annotations

import os
import sys
from pathlib import Path

# --- Application identity ---
APP_NAME: str = "OCR Studio"
APP_VERSION: str = "1.0.0"
APP_ORGANIZATION: str = "OCRStudio"
APP_ID: str = "com.ocrstudio.app"

# --- File system locations ---
if sys.platform == "win32":
    _LOCALAPPDATA = Path(os.environ.get("LOCALAPPDATA", Path.home() / "AppData" / "Local"))
    USER_DATA_DIR: Path = _LOCALAPPDATA / "OCRStudio"
else:
    # Fallback for development on non-Windows
    USER_DATA_DIR = Path.home() / ".ocrstudio"

CONFIG_DIR: Path = USER_DATA_DIR / "config"
PROFILES_DIR: Path = USER_DATA_DIR / "profiles"
TEMP_DIR: Path = USER_DATA_DIR / "temp"
LOGS_DIR: Path = USER_DATA_DIR / "logs"
RECOVERY_DIR: Path = USER_DATA_DIR / "recovery"
# Persistent OCR result cache: same input PDF + same profile skips the
# pipeline entirely on the second run. Pruned lazily on overflow.
OCR_CACHE_DIR: Path = USER_DATA_DIR / "ocr-cache"
OCR_CACHE_MAX_BYTES: int = 2 * 1024 * 1024 * 1024  # 2 GB budget, LRU-evicted

# --- Bundled resources (relative to app root) ---
def get_app_root() -> Path:
    """Return the directory that holds bundled resources.

    In a source checkout this is the project root. Under a PyInstaller
    build it must resolve to the location of the ``--add-data`` payload:

    * PyInstaller 6.x ``--onedir`` lays out the app as::

          OCRStudio/
              OCRStudio.exe
              _internal/
                  resources/...
                  profiles/...

      i.e. data assets live in the ``_internal`` subdirectory, NOT next
      to the executable. ``sys._MEIPASS`` is set to that ``_internal``
      directory by the bootloader.

    * PyInstaller ``--onefile`` extracts everything to a temporary
      directory and likewise sets ``sys._MEIPASS``.

    * Older ``--onedir`` (PyInstaller < 6.0) placed data next to the
      executable; we keep that fallback so downgrades stay working.
    """
    if getattr(sys, "frozen", False):
        meipass = getattr(sys, "_MEIPASS", None)
        if meipass:
            return Path(meipass)
        return Path(sys.executable).parent
    return Path(__file__).resolve().parent.parent.parent


APP_ROOT: Path = get_app_root()
RESOURCES_DIR: Path = APP_ROOT / "resources"
TESSDATA_DIR: Path = RESOURCES_DIR / "tessdata"
TESSERACT_BIN_DIR: Path = RESOURCES_DIR / "tesseract"
ICONS_DIR: Path = RESOURCES_DIR / "icons"
STYLES_DIR: Path = RESOURCES_DIR / "styles"
BUNDLED_PROFILES_DIR: Path = APP_ROOT / "profiles"
# Optional bundled HTR model weights. Populated by the CI installer
# pipeline (Download GOT-OCR 2.0 weights step) so end-users don't need
# to fetch ~580 MB from HuggingFace on first launch. Empty in source
# checkouts — ModelManager treats it as an alternate read-only lookup
# root if the user-writable copy doesn't have everything yet.
BUNDLED_MODELS_DIR: Path = RESOURCES_DIR / "models"

# --- Tesseract ---
TESSERACT_VERSION: str = "5.5.0"
TESSERACT_EXE_NAME: str = "tesseract.exe" if sys.platform == "win32" else "tesseract"

# --- OCR defaults ---
DEFAULT_LANGUAGE: str = "rus+eng"
DEFAULT_DPI: int = 300
MIN_DPI: int = 150
MAX_DPI: int = 600
DPI_CHOICES: tuple[int, ...] = (150, 200, 300, 400, 600)
DEFAULT_CONFIDENCE_THRESHOLD: float = 60.0
DEFAULT_TESSERACT_TIMEOUT_SEC: int = 120

# --- Parallel processing ---
DEFAULT_PARALLEL_WORKERS: int = 2
MIN_PARALLEL_WORKERS: int = 1
MAX_PARALLEL_WORKERS: int = 4

# --- Logging ---
LOG_FILE_NAME: str = "ocr-studio.log"
LOG_MAX_BYTES: int = 10 * 1024 * 1024  # 10 MB
LOG_BACKUP_COUNT: int = 3

# --- UI ---
UI_THUMBNAIL_SIZE: int = 160
UI_PREVIEW_UPDATE_DEBOUNCE_MS: int = 200
UI_PAGE_NUM_FORMAT: str = "--- Page {num} ---"
UI_AUTOSAVE_INTERVAL_PAGES: int = 10

# --- Preprocessing defaults ---
DEFAULT_CLAHE_CLIP: float = 2.0
DEFAULT_CLAHE_TILE: int = 8
DEFAULT_ADAPTIVE_BLOCK_SIZE: int = 31
DEFAULT_ADAPTIVE_C: int = 10
DEFAULT_SAUVOLA_WINDOW: int = 25
DEFAULT_SAUVOLA_K: float = 0.2
DEFAULT_MEDIAN_KSIZE: int = 3
DEFAULT_GAUSSIAN_SIGMA: float = 1.0
DEFAULT_NLM_H: int = 7
DEFAULT_MORPH_KSIZE: int = 3

# --- Preprocessing ranges (for UI sliders) ---
ADAPTIVE_BLOCK_SIZE_RANGE: tuple[int, int] = (3, 99)
ADAPTIVE_C_RANGE: tuple[int, int] = (-20, 20)
CLAHE_CLIP_RANGE: tuple[float, float] = (1.0, 10.0)
CLAHE_TILE_RANGE: tuple[int, int] = (4, 32)
MEDIAN_KSIZE_CHOICES: tuple[int, ...] = (3, 5, 7)
NLM_H_RANGE: tuple[int, int] = (3, 21)
DESKEW_MANUAL_RANGE: tuple[float, float] = (-45.0, 45.0)

# --- Colors (dark theme) ---
COLOR_BG_MAIN: str = "#1E1E2E"
COLOR_BG_PANEL: str = "#252536"
COLOR_BG_CONTROL: str = "#2D2D44"
COLOR_ACCENT: str = "#7C3AED"
COLOR_ACCENT_ALT: str = "#3B82F6"
COLOR_TEXT_PRIMARY: str = "#E4E4E7"
COLOR_TEXT_SECONDARY: str = "#A1A1AA"
COLOR_BORDER: str = "#3F3F5C"
COLOR_SUCCESS: str = "#22C55E"
COLOR_ERROR: str = "#EF4444"
COLOR_WARNING: str = "#F59E0B"

# --- File filters ---
PDF_FILE_FILTER: str = "PDF Files (*.pdf)"
IMPORT_FILE_FILTERS: str = "PDF Files (*.pdf);;All Files (*.*)"
EXPORT_PDF_FILTER: str = "PDF Files (*.pdf)"
EXPORT_TXT_FILTER: str = "Text Files (*.txt)"
EXPORT_DOCX_FILTER: str = "Word Documents (*.docx)"


def ensure_user_dirs() -> None:
    """Create all user-level directories if they don't exist."""
    for directory in (
        USER_DATA_DIR,
        CONFIG_DIR,
        PROFILES_DIR,
        TEMP_DIR,
        LOGS_DIR,
        RECOVERY_DIR,
        OCR_CACHE_DIR,
    ):
        directory.mkdir(parents=True, exist_ok=True)

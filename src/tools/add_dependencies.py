"""
Tool for installing Python dependencies at runtime.
"""

import subprocess
import sys
from typing import List

from langchain_core.tools import tool

from ..config import get_settings
from ..logging_config import get_logger

logger = get_logger(__name__)


# Common safe packages that can be installed
ALLOWED_PACKAGES = {
    # Data processing
    "pandas", "numpy", "scipy", "polars",
    # Visualization
    "matplotlib", "seaborn", "plotly", "altair",
    # File handling
    "openpyxl", "xlrd", "PyPDF2", "pdfplumber", "python-docx",
    "Pillow", "pillow",
    # Web/API
    "requests", "httpx", "beautifulsoup4", "lxml",
    # Data formats
    "pyyaml", "toml", "json5", "xmltodict", "html5lib", "markdown",
    # Statistics/ML
    "scikit-learn", "statsmodels", "sympy",
    # Geo
    "geopy", "shapely", "geopandas",
    # Network
    "networkx",
    # String/text processing
    "fuzzywuzzy", "python-Levenshtein", "regex", "chardet",
    # Date/time
    "python-dateutil", "pytz", "dateparser",
    # Validation/parsing
    "phonenumbers", "pycountry", "babel", "validators",
    # Crypto
    "cryptography",
    # Misc utilities
    "tqdm", "tabulate",
}


@tool
def add_dependencies(dependencies: List[str]) -> str:
    """
    Install Python packages into the environment.

    Use this tool when you need packages that aren't already available.
    The packages will be installed using pip and will be available for
    subsequent code execution.

    IMPORTANT:
    - Only commonly used data science packages are allowed
    - Installation may take a few seconds
    - If a package is not allowed, you'll get an error
    - Already installed packages will be skipped

    Parameters
    ----------
    dependencies : List[str]
        A list of Python package names to install.
        Example: ["pandas", "matplotlib", "scikit-learn"]

    Returns
    -------
    str
        A message indicating success or failure of the installation.
    """
    settings = get_settings()
    timeout = settings.code_execution_timeout_seconds
    
    if not dependencies:
        return "No dependencies specified."
    
    # Validate packages
    invalid_packages = []
    valid_packages = []
    
    for pkg in dependencies:
        # Normalize package name (lowercase, strip version specifiers)
        pkg_name = pkg.lower().split("==")[0].split(">=")[0].split("<=")[0].strip()
        
        if pkg_name in ALLOWED_PACKAGES or pkg_name.replace("-", "_") in ALLOWED_PACKAGES:
            valid_packages.append(pkg)
        else:
            invalid_packages.append(pkg)
    
    if invalid_packages:
        logger.warning(
            "blocked_packages",
            packages=invalid_packages,
        )
        return (
            f"The following packages are not allowed: {', '.join(invalid_packages)}. "
            f"Only common data science packages can be installed. "
            f"Valid packages requested: {', '.join(valid_packages) if valid_packages else 'none'}"
        )
    
    if not valid_packages:
        return "No valid packages to install."
    
    logger.info(
        "installing_packages",
        packages=valid_packages,
    )
    
    try:
        # Try using uv first (faster)
        try:
            result = subprocess.run(
                ["uv", "pip", "install"] + valid_packages,
                capture_output=True,
                text=True,
                timeout=timeout,
            )
            
            if result.returncode == 0:
                logger.info(
                    "packages_installed",
                    packages=valid_packages,
                    method="uv",
                )
                return f"Successfully installed: {', '.join(valid_packages)}"
            
            # If uv failed, fall back to pip
            raise FileNotFoundError("uv failed")
            
        except FileNotFoundError:
            # uv not available, use pip
            result = subprocess.run(
                [sys.executable, "-m", "pip", "install", "--quiet"] + valid_packages,
                capture_output=True,
                text=True,
                timeout=timeout,
            )
        
        if result.returncode == 0:
            logger.info(
                "packages_installed",
                packages=valid_packages,
                method="pip",
            )
            return f"Successfully installed: {', '.join(valid_packages)}"
        else:
            logger.error(
                "package_install_failed",
                packages=valid_packages,
                stderr=result.stderr,
            )
            return (
                f"Installation failed for: {', '.join(valid_packages)}\n"
                f"Error: {result.stderr[:500] if result.stderr else 'Unknown error'}"
            )
            
    except subprocess.TimeoutExpired:
        logger.error(
            "package_install_timeout",
            packages=valid_packages,
            timeout=timeout,
        )
        return f"Installation timed out after {timeout} seconds."
    except Exception as e:
        logger.error(
            "package_install_error",
            packages=valid_packages,
            error=str(e),
        )
        return f"Failed to install packages: {str(e)}"

"""
API versioning support for the API Gateway.
"""
from typing import Dict, Optional, List, Any
from fastapi import Request, HTTPException, status
from enum import Enum
import re
import logging

logger = logging.getLogger(__name__)


class VersioningStrategy(Enum):
    """API versioning strategies."""
    URL_PATH = "url_path"          # /api/v1/users
    HEADER = "header"              # Accept: application/vnd.api.v1+json
    QUERY_PARAM = "query_param"    # /api/users?version=1
    CONTENT_TYPE = "content_type"  # Content-Type: application/vnd.api.v1+json


class APIVersion:
    """Represents an API version."""
    
    def __init__(
        self,
        major: int,
        minor: int = 0,
        patch: int = 0,
        pre_release: Optional[str] = None
    ):
        self.major = major
        self.minor = minor
        self.patch = patch
        self.pre_release = pre_release
    
    @classmethod
    def from_string(cls, version_str: str) -> "APIVersion":
        """Parse version from string (e.g., 'v1.2.3', '1.0', 'v2.0-beta')."""
        # Remove 'v' prefix if present
        version_str = version_str.lstrip('v')
        
        # Handle pre-release versions
        pre_release = None
        if '-' in version_str:
            version_str, pre_release = version_str.split('-', 1)
        
        # Parse version parts
        parts = version_str.split('.')
        major = int(parts[0]) if len(parts) > 0 else 1
        minor = int(parts[1]) if len(parts) > 1 else 0
        patch = int(parts[2]) if len(parts) > 2 else 0
        
        return cls(major, minor, patch, pre_release)
    
    def __str__(self) -> str:
        """String representation of version."""
        version = f"{self.major}.{self.minor}.{self.patch}"
        if self.pre_release:
            version += f"-{self.pre_release}"
        return version
    
    def __eq__(self, other) -> bool:
        """Check version equality."""
        if not isinstance(other, APIVersion):
            return False
        return (
            self.major == other.major and
            self.minor == other.minor and
            self.patch == other.patch and
            self.pre_release == other.pre_release
        )
    
    def __lt__(self, other) -> bool:
        """Check if this version is less than other."""
        if not isinstance(other, APIVersion):
            return NotImplemented
        
        if self.major != other.major:
            return self.major < other.major
        if self.minor != other.minor:
            return self.minor < other.minor
        if self.patch != other.patch:
            return self.patch < other.patch
        
        # Handle pre-release versions
        if self.pre_release and not other.pre_release:
            return True
        if not self.pre_release and other.pre_release:
            return False
        if self.pre_release and other.pre_release:
            return self.pre_release < other.pre_release
        
        return False
    
    def __le__(self, other) -> bool:
        """Check if this version is less than or equal to other."""
        return self == other or self < other
    
    def __gt__(self, other) -> bool:
        """Check if this version is greater than other."""
        return not self <= other
    
    def __ge__(self, other) -> bool:
        """Check if this version is greater than or equal to other."""
        return not self < other
    
    def is_compatible(self, other: "APIVersion") -> bool:
        """Check if versions are compatible (same major version)."""
        return self.major == other.major


class VersionExtractor:
    """Extracts API version from requests using different strategies."""
    
    def __init__(self, strategies: List[VersioningStrategy] = None):
        self.strategies = strategies or [
            VersioningStrategy.URL_PATH,
            VersioningStrategy.HEADER,
            VersioningStrategy.QUERY_PARAM
        ]
    
    def extract_version(self, request: Request) -> Optional[APIVersion]:
        """Extract version from request using configured strategies."""
        
        for strategy in self.strategies:
            version = None
            
            if strategy == VersioningStrategy.URL_PATH:
                version = self._extract_from_url_path(request)
            elif strategy == VersioningStrategy.HEADER:
                version = self._extract_from_header(request)
            elif strategy == VersioningStrategy.QUERY_PARAM:
                version = self._extract_from_query_param(request)
            elif strategy == VersioningStrategy.CONTENT_TYPE:
                version = self._extract_from_content_type(request)
            
            if version:
                return version
        
        return None
    
    def _extract_from_url_path(self, request: Request) -> Optional[APIVersion]:
        """Extract version from URL path (e.g., /api/v1/users)."""
        path = request.url.path
        
        # Match patterns like /api/v1/, /api/v2.1/, etc.
        match = re.search(r'/api/v(\d+(?:\.\d+)*(?:-[a-zA-Z0-9]+)?)', path)
        if match:
            version_str = match.group(1)
            try:
                return APIVersion.from_string(version_str)
            except ValueError:
                logger.warning(f"Invalid version in URL path: {version_str}")
        
        return None
    
    def _extract_from_header(self, request: Request) -> Optional[APIVersion]:
        """Extract version from Accept header."""
        accept_header = request.headers.get("Accept", "")
        
        # Match patterns like application/vnd.api.v1+json
        match = re.search(r'application/vnd\.api\.v(\d+(?:\.\d+)*(?:-[a-zA-Z0-9]+)?)', accept_header)
        if match:
            version_str = match.group(1)
            try:
                return APIVersion.from_string(version_str)
            except ValueError:
                logger.warning(f"Invalid version in Accept header: {version_str}")
        
        # Also check for custom version header
        version_header = request.headers.get("API-Version")
        if version_header:
            try:
                return APIVersion.from_string(version_header)
            except ValueError:
                logger.warning(f"Invalid version in API-Version header: {version_header}")
        
        return None
    
    def _extract_from_query_param(self, request: Request) -> Optional[APIVersion]:
        """Extract version from query parameter."""
        version_param = request.query_params.get("version") or request.query_params.get("v")
        if version_param:
            try:
                return APIVersion.from_string(version_param)
            except ValueError:
                logger.warning(f"Invalid version in query parameter: {version_param}")
        
        return None
    
    def _extract_from_content_type(self, request: Request) -> Optional[APIVersion]:
        """Extract version from Content-Type header."""
        content_type = request.headers.get("Content-Type", "")
        
        # Match patterns like application/vnd.api.v1+json
        match = re.search(r'application/vnd\.api\.v(\d+(?:\.\d+)*(?:-[a-zA-Z0-9]+)?)', content_type)
        if match:
            version_str = match.group(1)
            try:
                return APIVersion.from_string(version_str)
            except ValueError:
                logger.warning(f"Invalid version in Content-Type header: {version_str}")
        
        return None


class VersionManager:
    """Manages API versions and compatibility."""
    
    def __init__(self):
        self.supported_versions = [
            APIVersion(1, 0, 0),  # v1.0.0
            APIVersion(2, 0, 0),  # v2.0.0 (future)
        ]
        self.default_version = APIVersion(1, 0, 0)
        self.deprecated_versions = []
        
        # Version-specific routing rules
        self.version_routes = {
            "1.0.0": {
                "users": "/api/v1/users",
                "tenants": "/api/v1/tenants",
                "data": "/api/v1/data",
                "ml": "/api/v1/ml"
            },
            "2.0.0": {
                "users": "/api/v2/users",
                "tenants": "/api/v2/tenants", 
                "data": "/api/v2/data",
                "ml": "/api/v2/ml"
            }
        }
    
    def get_version_for_request(self, request: Request) -> APIVersion:
        """Get the API version to use for a request."""
        extractor = VersionExtractor()
        requested_version = extractor.extract_version(request)
        
        if requested_version:
            # Check if requested version is supported
            if self.is_version_supported(requested_version):
                return requested_version
            else:
                # Find closest compatible version
                compatible_version = self.find_compatible_version(requested_version)
                if compatible_version:
                    return compatible_version
                else:
                    raise HTTPException(
                        status_code=status.HTTP_400_BAD_REQUEST,
                        detail=f"API version {requested_version} is not supported"
                    )
        
        # Return default version if none specified
        return self.default_version
    
    def is_version_supported(self, version: APIVersion) -> bool:
        """Check if a version is supported."""
        return version in self.supported_versions
    
    def find_compatible_version(self, requested_version: APIVersion) -> Optional[APIVersion]:
        """Find a compatible version for the requested version."""
        compatible_versions = [
            v for v in self.supported_versions
            if v.is_compatible(requested_version) and v >= requested_version
        ]
        
        if compatible_versions:
            # Return the lowest compatible version
            return min(compatible_versions)
        
        return None
    
    def get_route_for_version(self, version: APIVersion, service: str) -> Optional[str]:
        """Get the route pattern for a service in a specific version."""
        version_key = str(version)
        version_routes = self.version_routes.get(version_key, {})
        return version_routes.get(service)
    
    def add_version(self, version: APIVersion, routes: Dict[str, str]):
        """Add a new API version."""
        if version not in self.supported_versions:
            self.supported_versions.append(version)
            self.supported_versions.sort()
        
        self.version_routes[str(version)] = routes
    
    def deprecate_version(self, version: APIVersion, sunset_date: Optional[str] = None):
        """Mark a version as deprecated."""
        if version not in self.deprecated_versions:
            self.deprecated_versions.append({
                "version": version,
                "sunset_date": sunset_date
            })
    
    def get_version_info(self) -> Dict[str, Any]:
        """Get information about all API versions."""
        return {
            "supported_versions": [str(v) for v in self.supported_versions],
            "default_version": str(self.default_version),
            "deprecated_versions": [
                {
                    "version": str(item["version"]),
                    "sunset_date": item["sunset_date"]
                }
                for item in self.deprecated_versions
            ]
        }


# Global version manager instance
version_manager = VersionManager()


# Utility functions
def get_api_version(request: Request) -> APIVersion:
    """Get API version for request."""
    return version_manager.get_version_for_request(request)


def add_version_headers(response, version: APIVersion):
    """Add version information to response headers."""
    response.headers["API-Version"] = str(version)
    response.headers["API-Supported-Versions"] = ",".join(
        str(v) for v in version_manager.supported_versions
    )
    
    # Add deprecation warning if version is deprecated
    for deprecated in version_manager.deprecated_versions:
        if deprecated["version"] == version:
            response.headers["Deprecation"] = "true"
            if deprecated["sunset_date"]:
                response.headers["Sunset"] = deprecated["sunset_date"]
            break


def transform_path_for_version(path: str, version: APIVersion) -> str:
    """Transform request path based on API version."""
    # Remove version from path if present
    path_without_version = re.sub(r'/api/v\d+(?:\.\d+)*', '/api', path)
    
    # Add version-specific prefix
    if version.major == 1:
        return path_without_version.replace('/api', '/api/v1', 1)
    elif version.major == 2:
        return path_without_version.replace('/api', '/api/v2', 1)
    
    return path
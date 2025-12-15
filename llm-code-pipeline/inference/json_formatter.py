"""
JSON Formatter - Handles JSON extraction and validation for LLM outputs.
"""

import json
import re
import logging
from typing import Optional, Any, Union
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)


class JSONExtractionMode(Enum):
    """Modes for JSON extraction from text."""
    STRICT = "strict"  # Only accept valid JSON
    LENIENT = "lenient"  # Try to fix common issues
    TAGGED = "tagged"  # Extract from <json_output> tags


@dataclass
class JSONExtractionResult:
    """Result of JSON extraction."""
    success: bool
    data: Optional[Any]
    raw_text: str
    error: Optional[str] = None
    extraction_mode: str = "strict"


class JSONFormatter:
    """
    Handles JSON extraction, validation, and formatting for LLM outputs.

    Supports multiple extraction modes:
    - Strict: Only accepts valid JSON
    - Lenient: Attempts to fix common JSON issues
    - Tagged: Extracts JSON from <json_output> tags
    """

    # Regex patterns for JSON extraction
    JSON_BLOCK_PATTERN = re.compile(
        r'```(?:json)?\s*([\s\S]*?)```',
        re.IGNORECASE
    )

    JSON_TAGGED_PATTERN = re.compile(
        r'<json_output>\s*([\s\S]*?)\s*</json_output>',
        re.IGNORECASE
    )

    JSON_OBJECT_PATTERN = re.compile(
        r'\{[\s\S]*\}',
        re.MULTILINE
    )

    JSON_ARRAY_PATTERN = re.compile(
        r'\[[\s\S]*\]',
        re.MULTILINE
    )

    def __init__(self, default_mode: JSONExtractionMode = JSONExtractionMode.LENIENT):
        """
        Initialize JSON formatter.

        Args:
            default_mode: Default extraction mode
        """
        self.default_mode = default_mode

    def extract_json(
        self,
        text: str,
        mode: Optional[JSONExtractionMode] = None,
        schema: Optional[dict] = None
    ) -> JSONExtractionResult:
        """
        Extract JSON from LLM output text.

        Args:
            text: Raw LLM output text
            mode: Extraction mode (uses default if not specified)
            schema: Optional JSON schema for validation

        Returns:
            JSONExtractionResult with parsed data or error
        """
        mode = mode or self.default_mode

        if mode == JSONExtractionMode.TAGGED:
            return self._extract_tagged(text, schema)
        elif mode == JSONExtractionMode.STRICT:
            return self._extract_strict(text, schema)
        else:
            return self._extract_lenient(text, schema)

    def _extract_tagged(
        self,
        text: str,
        schema: Optional[dict] = None
    ) -> JSONExtractionResult:
        """Extract JSON from <json_output> tags."""
        match = self.JSON_TAGGED_PATTERN.search(text)

        if match:
            json_str = match.group(1).strip()
            return self._parse_and_validate(
                json_str, text, schema, "tagged"
            )

        return JSONExtractionResult(
            success=False,
            data=None,
            raw_text=text,
            error="No <json_output> tags found",
            extraction_mode="tagged"
        )

    def _extract_strict(
        self,
        text: str,
        schema: Optional[dict] = None
    ) -> JSONExtractionResult:
        """Extract JSON in strict mode - no modifications."""
        # Try direct parse first
        try:
            data = json.loads(text.strip())
            return self._validate_against_schema(
                data, text, schema, "strict"
            )
        except json.JSONDecodeError:
            pass

        # Try extracting from code blocks
        match = self.JSON_BLOCK_PATTERN.search(text)
        if match:
            json_str = match.group(1).strip()
            try:
                data = json.loads(json_str)
                return self._validate_against_schema(
                    data, text, schema, "strict"
                )
            except json.JSONDecodeError as e:
                return JSONExtractionResult(
                    success=False,
                    data=None,
                    raw_text=text,
                    error=f"Invalid JSON in code block: {e}",
                    extraction_mode="strict"
                )

        # Try finding JSON object or array
        for pattern in [self.JSON_OBJECT_PATTERN, self.JSON_ARRAY_PATTERN]:
            match = pattern.search(text)
            if match:
                json_str = match.group(0)
                try:
                    data = json.loads(json_str)
                    return self._validate_against_schema(
                        data, text, schema, "strict"
                    )
                except json.JSONDecodeError:
                    continue

        return JSONExtractionResult(
            success=False,
            data=None,
            raw_text=text,
            error="No valid JSON found in text",
            extraction_mode="strict"
        )

    def _extract_lenient(
        self,
        text: str,
        schema: Optional[dict] = None
    ) -> JSONExtractionResult:
        """Extract JSON with automatic fixing of common issues."""
        # First try strict extraction
        result = self._extract_strict(text, schema)
        if result.success:
            result.extraction_mode = "lenient"
            return result

        # Try to find and fix JSON
        json_str = self._find_json_candidate(text)
        if json_str:
            fixed_json = self._fix_common_issues(json_str)
            return self._parse_and_validate(
                fixed_json, text, schema, "lenient"
            )

        return JSONExtractionResult(
            success=False,
            data=None,
            raw_text=text,
            error="Could not extract or fix JSON",
            extraction_mode="lenient"
        )

    def _find_json_candidate(self, text: str) -> Optional[str]:
        """Find the best JSON candidate in text."""
        # Check code blocks first
        match = self.JSON_BLOCK_PATTERN.search(text)
        if match:
            return match.group(1).strip()

        # Check tagged output
        match = self.JSON_TAGGED_PATTERN.search(text)
        if match:
            return match.group(1).strip()

        # Find object or array patterns
        for pattern in [self.JSON_OBJECT_PATTERN, self.JSON_ARRAY_PATTERN]:
            matches = pattern.findall(text)
            if matches:
                # Return the longest match (likely the most complete)
                return max(matches, key=len)

        return None

    def _fix_common_issues(self, json_str: str) -> str:
        """Fix common JSON formatting issues from LLM output."""
        fixed = json_str

        # Remove trailing commas before } or ]
        fixed = re.sub(r',\s*([}\]])', r'\1', fixed)

        # Fix single quotes to double quotes (but not in strings)
        # This is a simple heuristic - may not work for all cases
        fixed = re.sub(r"(?<!\\)'([^']*)'(?=\s*:)", r'"\1"', fixed)
        fixed = re.sub(r":\s*'([^']*)'", r': "\1"', fixed)

        # Remove comments
        fixed = re.sub(r'//[^\n]*\n', '\n', fixed)
        fixed = re.sub(r'/\*[\s\S]*?\*/', '', fixed)

        # Fix unquoted keys
        fixed = re.sub(
            r'([{,]\s*)([a-zA-Z_][a-zA-Z0-9_]*)\s*:',
            r'\1"\2":',
            fixed
        )

        # Fix Python-style booleans and None
        fixed = re.sub(r'\bTrue\b', 'true', fixed)
        fixed = re.sub(r'\bFalse\b', 'false', fixed)
        fixed = re.sub(r'\bNone\b', 'null', fixed)

        # Remove control characters
        fixed = re.sub(r'[\x00-\x1f\x7f-\x9f]', '', fixed)

        return fixed

    def _parse_and_validate(
        self,
        json_str: str,
        original_text: str,
        schema: Optional[dict],
        mode: str
    ) -> JSONExtractionResult:
        """Parse JSON string and optionally validate against schema."""
        try:
            data = json.loads(json_str)
            return self._validate_against_schema(
                data, original_text, schema, mode
            )
        except json.JSONDecodeError as e:
            return JSONExtractionResult(
                success=False,
                data=None,
                raw_text=original_text,
                error=f"JSON parse error: {e}",
                extraction_mode=mode
            )

    def _validate_against_schema(
        self,
        data: Any,
        original_text: str,
        schema: Optional[dict],
        mode: str
    ) -> JSONExtractionResult:
        """Validate parsed JSON against schema if provided."""
        if schema is None:
            return JSONExtractionResult(
                success=True,
                data=data,
                raw_text=original_text,
                extraction_mode=mode
            )

        try:
            import jsonschema
            jsonschema.validate(data, schema)
            return JSONExtractionResult(
                success=True,
                data=data,
                raw_text=original_text,
                extraction_mode=mode
            )
        except ImportError:
            logger.warning("jsonschema not installed, skipping validation")
            return JSONExtractionResult(
                success=True,
                data=data,
                raw_text=original_text,
                extraction_mode=mode
            )
        except jsonschema.ValidationError as e:
            return JSONExtractionResult(
                success=False,
                data=data,
                raw_text=original_text,
                error=f"Schema validation failed: {e.message}",
                extraction_mode=mode
            )

    def format_for_output(
        self,
        data: Any,
        wrap_in_tags: bool = False,
        indent: int = 2
    ) -> str:
        """
        Format data as JSON string for output.

        Args:
            data: Data to format
            wrap_in_tags: Wrap output in <json_output> tags
            indent: Indentation level

        Returns:
            Formatted JSON string
        """
        json_str = json.dumps(data, indent=indent, ensure_ascii=False)

        if wrap_in_tags:
            return f"<json_output>\n{json_str}\n</json_output>"

        return json_str

    def create_json_prompt_suffix(
        self,
        schema: Optional[dict] = None,
        use_tags: bool = True
    ) -> str:
        """
        Create a prompt suffix instructing JSON output format.

        Args:
            schema: Optional JSON schema to include
            use_tags: Use <json_output> tag format

        Returns:
            Prompt suffix string
        """
        if use_tags:
            suffix = "\n\nRespond with valid JSON wrapped in <json_output></json_output> tags."
        else:
            suffix = "\n\nRespond with valid JSON only, no other text."

        if schema:
            schema_str = json.dumps(schema, indent=2)
            suffix += f"\n\nUse this JSON schema:\n```json\n{schema_str}\n```"

        return suffix


# Pre-configured formatters
strict_formatter = JSONFormatter(JSONExtractionMode.STRICT)
lenient_formatter = JSONFormatter(JSONExtractionMode.LENIENT)
tagged_formatter = JSONFormatter(JSONExtractionMode.TAGGED)

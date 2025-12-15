"""
Response format schemas for structured outputs.
"""

from typing import Optional, Any, Literal
from pydantic import BaseModel, Field


class JSONSchema(BaseModel):
    """JSON Schema definition for structured outputs."""
    name: str = Field(..., description="Name of the schema")
    description: Optional[str] = Field(None, description="Schema description")
    schema_: dict = Field(..., alias="schema", description="JSON Schema definition")
    strict: bool = Field(default=False, description="Enforce strict schema adherence")

    class Config:
        populate_by_name = True
        json_schema_extra = {
            "example": {
                "name": "CodeResponse",
                "description": "Response containing converted code",
                "schema": {
                    "type": "object",
                    "properties": {
                        "convertedCode": {"type": "string"},
                        "language": {"type": "string"},
                        "explanation": {"type": "string"}
                    },
                    "required": ["convertedCode", "language"]
                },
                "strict": True
            }
        }


class ResponseFormat(BaseModel):
    """
    Response format specification.

    Supports:
    - text: Plain text response
    - json_object: Valid JSON object
    - json_schema: JSON adhering to a specific schema
    """
    type: Literal["text", "json_object", "json_schema"] = Field(
        default="text",
        description="Response format type"
    )
    json_schema: Optional[JSONSchema] = Field(
        default=None,
        description="JSON schema for structured output"
    )

    class Config:
        json_schema_extra = {
            "examples": [
                {"type": "text"},
                {"type": "json_object"},
                {
                    "type": "json_schema",
                    "json_schema": {
                        "name": "CodeConversion",
                        "schema": {
                            "type": "object",
                            "properties": {
                                "code": {"type": "string"},
                                "language": {"type": "string"}
                            },
                            "required": ["code"]
                        }
                    }
                }
            ]
        }


# Pre-defined response formats for common use cases
class CodeConversionSchema(BaseModel):
    """Schema for code conversion responses."""
    converted_code: str = Field(..., description="The converted source code")
    source_language: str = Field(..., description="Original programming language")
    target_language: str = Field(..., description="Target programming language")
    explanation: Optional[str] = Field(None, description="Explanation of conversion")
    warnings: Optional[list[str]] = Field(None, description="Conversion warnings")


class CodeGenerationSchema(BaseModel):
    """Schema for code generation responses."""
    code: str = Field(..., description="Generated code")
    language: str = Field(..., description="Programming language")
    description: Optional[str] = Field(None, description="Code description")
    dependencies: Optional[list[str]] = Field(None, description="Required dependencies")


class CodeRefactoringSchema(BaseModel):
    """Schema for code refactoring responses."""
    refactored_code: str = Field(..., description="Refactored code")
    changes: list[str] = Field(..., description="List of changes made")
    improvements: Optional[list[str]] = Field(None, description="Improvements made")


# Helper function to create response format with schema
def create_json_response_format(
    schema_name: str,
    properties: dict[str, dict],
    required: list[str],
    description: Optional[str] = None,
    strict: bool = False
) -> ResponseFormat:
    """
    Create a ResponseFormat with JSON schema.

    Args:
        schema_name: Name for the schema
        properties: JSON Schema properties
        required: List of required property names
        description: Schema description
        strict: Enforce strict adherence

    Returns:
        ResponseFormat configured for JSON schema output
    """
    return ResponseFormat(
        type="json_schema",
        json_schema=JSONSchema(
            name=schema_name,
            description=description,
            schema={
                "type": "object",
                "properties": properties,
                "required": required
            },
            strict=strict
        )
    )


# Pre-built response formats
CODE_CONVERSION_FORMAT = create_json_response_format(
    schema_name="CodeConversion",
    properties={
        "convertedCode": {"type": "string", "description": "Converted source code"},
        "sourceLanguage": {"type": "string", "description": "Original language"},
        "targetLanguage": {"type": "string", "description": "Target language"},
        "explanation": {"type": "string", "description": "Conversion explanation"}
    },
    required=["convertedCode", "targetLanguage"],
    description="Code conversion response format"
)

CODE_GENERATION_FORMAT = create_json_response_format(
    schema_name="CodeGeneration",
    properties={
        "code": {"type": "string", "description": "Generated code"},
        "language": {"type": "string", "description": "Programming language"},
        "description": {"type": "string", "description": "Code description"}
    },
    required=["code", "language"],
    description="Code generation response format"
)

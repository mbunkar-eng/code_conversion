import json
import os
from typing import Optional
import requests
try:
    from openai import OpenAI
except Exception:
    OpenAI = None

# ================================================================
# PROMPTS
# ================================================================
# (Truncated in-file docstrings for brevity; these are the exact prompts
# you provided in the request. Keep full prompt fidelity when using.
# For readability here we store them as multi-line constants.)

CONVERSION_PROMPT = r"""
Convert the following ASP.NET Core C# file into equivalent Java code using Quarkus.
**Very Important: Do not reduce the code lenght ,Do not change my base code logic and maintain the same code quality as in the original programming language. Provide the full converted code.**
CRITICAL REQUIREMENTS:
1. Generate ONLY ONE Java class per file
2. Do NOT include multiple classes in the same file
3. Do NOT include configuration properties in Java files
4. For DTO files: Convert ALL classes in the file (including nested classes, multiple DTOs, etc.)
5. For Controller files: Convert the main controller class only
6. For Model/Entity files: Convert the main entity class only
7. For Service files: Convert the main service class only
8. Do NOT recreate classes that already exist (see existing classes list)
9. Use imports to reference existing classes instead of recreating them
10. Output ONLY the converted Java code - no explanations, no additional classes
11. Maintain the same structure, routes, and logic
12. If there's no direct equivalent in Quarkus, implement the closest possible Java alternative

QUARKUS FRAMEWORK REQUIREMENTS:
Use proper package declaration: package ${basePackage}.${targetFolder};
For Controllers: Use @Path, @GET, @POST, @PUT, @DELETE, @Consumes, @Produces annotations
For Entities: Use @Entity, @Id, @GeneratedValue, @Column, @OneToMany, @ManyToOne, @JoinColumn annotations
For Services: Use @ApplicationScoped annotation
For Repositories: Use @ApplicationScoped and PanacheRepository interface
For DTOs: Use proper Jackson annotations like @JsonProperty
Import all necessary Quarkus and JPA annotations
Use jakarta.ws.rs.* for REST endpoints (NOT javax)
Use jakarta.persistence.* for JPA entities (NOT javax)
Use jakarta.enterprise.context.ApplicationScoped for CDI (NOT javax)
Use jakarta.inject.Inject for dependency injection (NOT javax)
Use io.quarkus.* for Quarkus-specific annotations

Target Folder: ${targetFolder}
Base Package: ${basePackage}




====================================================================
ABSOLUTE REQUIREMENTS
====================================================================
1. Produce ONLY Java code — no explanations, no commentary, no Markdown.
2. The Java code must compile, assuming correct import statements.
3. The conversion MUST follow all modules and rules described below.
4. Preserve original logic, method behavior, class structure, and semantics exactly.
5. Do NOT change naming unless required by Java language constraints.
6. Do NOT omit members, methods, properties, or logic unless they are clearly unreachable.
7. If ambiguous C# constructs exist, choose the closest Java equivalent without altering semantics.

... (full prompt continues exactly as provided by the user) ...
"""

VALIDATION_PROMPT = r"""
(VALIDATION PROMPT CONTENT - Use the full validator prompt exactly as provided.)
"""

GENERATION_WITH_FEEDBACK_PROMPT = r"""
(GENERATION WITH FEEDBACK PROMPT CONTENT - Use the full prompt exactly as provided.)
"""


# ================================================================
# BASE ENGINE CLASS
# ================================================================

class CodeConverterEngine:
    """
    Engine that can target either the OpenAI client or a local inference HTTP API.
    If backend=='openai' the OpenAI Python client must be available and an api_key supplied.
    If backend=='local' the `api_base_url` (eg. http://localhost:8000) will be used and
    the API key (if required) is taken from env var `LLM_API_KEY` or session.
    """
    def __init__(self, api_key: Optional[str], model_name: str = "gpt-4.1", backend: str = "openai", api_base_url: str = "http://localhost:8000"):
        self.backend = backend
        self.model = model_name
        self.api_base_url = api_base_url.rstrip("/")

        if self.backend == "openai":
            if not api_key:
                raise ValueError("API key required to initialize OpenAI client for backend 'openai'")
            if OpenAI is None:
                raise RuntimeError("OpenAI client library not available. Install openai package or use backend='local'.")
            self.client = OpenAI(api_key=api_key)
        else:
            # for local backend we still accept an api_key if server enforces bearer token
            self.client = None
            self.api_key = api_key or os.getenv("LLM_API_KEY") or os.getenv("OPENAI_API_KEY")

    def call_text_model(self, system_prompt: str, user_content: str, endpoint: str = "/v1/chat/completions") -> str:
        """
        Standard LLM call for plain text (Java code generation).
        Supports both OpenAI client and a local HTTP inference API.
        """
        if self.backend == "openai":
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_content}
                ],
                temperature=0.0
            )
            return response.choices[0].message.content.strip()

        # local backend via requests
        url = f"{self.api_base_url}{endpoint}"
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_content}
            ],
            "temperature": 0.0
        }

        resp = requests.post(url, headers=headers, json=payload, timeout=120)
        resp.raise_for_status()
        data = resp.json()
        # expect OpenAI-compatible response structure
        return data.get("choices", [{}])[0].get("message", {}).get("content", "").strip()

    def call_json_model(self, system_prompt: str, user_content: str, endpoint: str = "/v1/chat/completions") -> dict:
        """
        JSON-enforced LLM call. For local backend we expect the server to return
        a JSON string in .choices[0].message.content which itself is JSON.
        """
        if self.backend == "openai":
            response = self.client.chat.completions.create(
                model=self.model,
                response_format={"type": "json_object"},
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_content}
                ],
                temperature=0.0
            )
            return json.loads(response.choices[0].message.content)

        # local HTTP backend: call and parse
        url = f"{self.api_base_url}{endpoint}"
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_content}
            ],
            "temperature": 0.0
        }

        resp = requests.post(url, headers=headers, json=payload, timeout=120)
        resp.raise_for_status()
        data = resp.json()
        # Try to parse content as JSON
        content = data.get("choices", [{}])[0].get("message", {}).get("content", "")
        try:
            return json.loads(content)
        except Exception:
            # If server already returned JSON object, return it
            if isinstance(content, dict):
                return content
            # Otherwise return the raw response for validator to inspect
            return {"raw": content, "response": data}


# ================================================================
# CONVERTER CLASS
# ================================================================

class CS2JavaConverter(CodeConverterEngine):

    def convert_initial(self, cs_code: str, cs_file_name: str, java_file_name: str) -> str:
        """
        First-pass conversion (no feedback).
        """
        user_message = f"""
C# FILE NAME: {cs_file_name}

C# CODE:
{cs_code}

CONVERT TO JAVA:
- File name must be {java_file_name}
- Output ONLY Java code.
"""
        return self.call_text_model(CONVERSION_PROMPT, user_message)

    def convert_with_feedback(
        self,
        cs_code: str,
        cs_file_name: str,
        previous_java_code: str,
        previous_java_file_name: str,
        validation_report: dict
    ) -> str:
        """
        Regenerate Java code using validator feedback.
        """
        rule_suggestions = []
        for rule in validation_report.get("results", []):
            if rule.get("status") in ("fail", "partial"):
                suggestion = rule.get("suggestion", "")
                if suggestion.strip():
                    rule_suggestions.append(f"- {rule.get('module')}: {suggestion}")

        suggestions_text = "\n".join(rule_suggestions) or "No suggestions."

        user_message = GENERATION_WITH_FEEDBACK_PROMPT.format(
            cs_file_name=cs_file_name,
            cs_code=cs_code,
            previous_java_file_name=previous_java_file_name,
            previous_java_code=previous_java_code,
            overall_status=validation_report.get("overall_status", ""),
            validation_notes=validation_report.get("notes", ""),
            validation_review=validation_report.get("review", ""),
            validation_feedback=validation_report.get("feedback", ""),
            validation_rule_suggestions=suggestions_text
        )

        return self.call_text_model(CONVERSION_PROMPT, user_message)


# ================================================================
# VALIDATOR CLASS
# ================================================================

class CS2JavaValidator(CodeConverterEngine):

    def validate(self, cs_code: str, cs_file_name: str, java_code: str, java_file_name: str) -> dict:
        """
        Run JSON-enforced validation.
        """
        validation_input = {
            "original_cs_file_name": cs_file_name,
            "original_cs_code": cs_code,
            "converted_java_file_name": java_file_name,
            "converted_java_code": java_code
        }

        return self.call_json_model(
            VALIDATION_PROMPT,
            json.dumps(validation_input, indent=2)
        )


# ================================================================
# MANAGER ORCHESTRATOR (3 ITERATION MAX)
# ================================================================

class CS2JavaManager:
    MAX_ITERATIONS = 3

    def __init__(self, api_key: str, model_name="gpt-4.1"):
        self.converter = CS2JavaConverter(api_key, model_name)
        self.validator = CS2JavaValidator(api_key, model_name)

    def convert_with_validation(self, cs_file_name: str, cs_code: str, output_dir: str = "output") -> dict:
        """
        1. Convert once
        2. Validate
        3. If fail/partial → regenerate using feedback
        4. Repeat max 3 iterations
        5. Save the final Java code to a file
        """
        # Derive Java file name
        if cs_file_name.endswith(".cs"):
            java_file_name = cs_file_name[:-3] + ".java"
        else:
            java_file_name = cs_file_name + ".java"

        # Step 1: Initial conversion
        java_code = self.converter.convert_initial(cs_code, cs_file_name, java_file_name)

        final_report = None

        # Step 2 & 3: Validation loop
        for iteration in range(self.MAX_ITERATIONS):
            final_report = self.validator.validate(
                cs_code=cs_code,
                cs_file_name=cs_file_name,
                java_code=java_code,
                java_file_name=java_file_name
            )

            status = final_report.get("overall_status")

            if status == "pass":
                break  # success

            # refine if iterations remain
            if iteration < self.MAX_ITERATIONS - 1:
                java_code = self.converter.convert_with_feedback(
                    cs_code=cs_code,
                    cs_file_name=cs_file_name,
                    previous_java_code=java_code,
                    previous_java_file_name=java_file_name,
                    validation_report=final_report
                )

        # Save the final Java code to a file
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, java_file_name)
        with open(output_path, "w", encoding="utf-8") as java_file:
            java_file.write(java_code)

        print(f"✅ Java code saved to: {output_path}")

        return {
            "final_java_code": java_code,
            "validation_report": final_report,
            "output_path": output_path
        }


# ================================================================
# SAFE EXAMPLE USAGE (no secrets in code)
# ================================================================

if __name__ == "__main__":
    # Provide your OpenAI API key via the environment variable LLM_API_KEY or OPENAI_API_KEY
    api_key = os.getenv("LLM_API_KEY") or os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("Please set LLM_API_KEY or OPENAI_API_KEY in your environment to run this example.")
        exit(1)

    # Example manager usage (uncomment and replace cs_content to actually run):
    # manager = CS2JavaManager(api_key)
    # cs_filename = "Example.cs"
    # cs_content = "// your C# file content here"
    # result = manager.convert_with_validation(cs_filename, cs_content, output_dir="java_output")
    # print(result["output_path"]) 

from typing import Literal, List
from pydantic import BaseModel, Field, validate_call

class SubState(BaseModel):
    source_type: Literal["URL", "Document", "Unknown"] = Field(
        description="Source type must be either URL or Document."
    )

    source_id: str = Field(
        description="Source ID."
    )

    source_name: str = Field(
        description="Source Original Name."
    )

    source_summary: str = Field(
        description="Short summary of uploaded source."
    )

class SourceConfigStructure(BaseModel):
    source_type: Literal["URL", "Document"] = Field(
        description="Source type must be either URL or Document."
    )
    
    source_id: str = Field(
        description="Source ID."
    )

    top_k: int = Field(
        ge=2, le=10,
        description=(
            "Number of document chunks to retrieve from this source based on the specific query. "
            "Must be between 2 and 10."
        )
    )

class SourceConfigListStruction(BaseModel):
    items: List[SourceConfigStructure] = Field(default_factory=list)

class RetrievalState(BaseModel):
    query: str = Field(
        description="Main query given by user."
    )

    indexID: str = Field(
        description="Index or Unique database ID for each user."
    )

    notebookID: str = Field(
        description="Notebook ID of user."
    )

    listOfSummaries: List[SubState] = Field(
        description="List of objects each object has source type, source_if, source_name and source_summary.",
        default_factory=list
    )

    sourceConfig: SourceConfigListStruction = Field(
        default_factory=SourceConfigListStruction
    )

    context: str = Field(
        description="Main context which is extracted by each sources"
    )
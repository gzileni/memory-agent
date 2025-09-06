from pydantic import BaseModel, Field
from typing import Optional


class Conversation(BaseModel):
    """
    Represents a single message in the conversation.
    """
    role: str = Field(..., description="Role of the sender, e.g., 'user'")
    content: str = Field(..., description="Message text")


class Triple(BaseModel):
    """
    Store all new facts, preferences, and relationships as triples.
    """
    subject: str
    predicate: str
    object: str
    context: str | None = None


class UserProfile(BaseModel):
    """Represents the full representation of a user."""
    name: Optional[str] = None
    language: Optional[str] = None
    timezone: Optional[str] = None


class Episode(BaseModel):
    """
    Write the episode from the perspective of the agent within it.
    Use the benefit of hindsight to record the memory, saving the agent's
    key internal thought process so it can learn over time.
    """

    observation: str = Field(
        ...,
        description="The context and setup - what happened"
    )
    thoughts: str = Field(
        ...,
        description=(
            "Internal reasoning process and observations of the agent in "
            "the episode that let it arrive at the correct "
            "action and result. \"I ...\""
        ),
    )
    action: str = Field(
        ...,
        description=(
            "What was done, how, and in what format. "
            "(Include whatever is salient to the success of the action). I .."
        ),
    )
    result: str = Field(
        ...,
        description=(
            "Outcome and retrospective. What did you do well? "
            "What could you do better next time? I ..."
        ),
    )
    action: str = Field(
        ...,
        description=(
            "What was done, how, and in what format. "
            "(Include whatever is salient to the success of the action). I .."
        ),
    )
    result: str = Field(
        ...,
        description=(
            "Outcome and retrospective. What did you do well? "
            "What could you do better next time? I ..."
        ),
    )

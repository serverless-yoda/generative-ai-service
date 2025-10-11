from typing import Literal, Annotated
from uuid import uuid4
from datetime import datetime
from pydantic import (
    BaseModel,
    Field,
    HttpUrl,
    IPvAnyAddress,
    PositiveInt,
    validate_call,
    AfterValidator,
    computed_field,
    model_validator,
)

from app.api.core.utils import count_tokens

VoicePresets = Literal['v2/en_speaker_1', 'v2/en_speaker_9']
ImageSize = Annotated[
    tuple[PositiveInt, PositiveInt], "Width and height of an image in pixels"
]
SupportedModels = Literal['tinyLlama']

@validate_call
def is_square_image(value: ImageSize) -> ImageSize:
    if value[0] / value[1] != 1:
        raise ValueError("Only square images are supported")
    if value[0] not in [512, 1024]:
        raise ValueError(f"Invalid output size: {value} - expected 512 or 1024")
    return value

OutputSize = Annotated[ImageSize, AfterValidator(is_square_image)]

class ModelRequest(BaseModel):
    prompt: Annotated[str, Field(min_length=1, max_length=10000)]

class ModelResponse(BaseModel):
    request_id: Annotated[str, Field(default_factory=lambda: uuid4().hex)]
    ip: Annotated[str, IPvAnyAddress] | "None" = "::1"
    content: Annotated[str | None, Field(min_length=0, max_length=10000)]
    created_at: datetime = Field(default_factory=datetime.now)

class TextModelRequest(ModelRequest):
    model: SupportedModels
    temperature: Annotated[float, Field(ge=0.0, le=1.0, default=0.1)]

class TextModelResponse(ModelResponse):
    model: SupportedModels
    price: Annotated[float, Field(ge=0, default=0.01)]
    temperature: Annotated[float, Field(ge=0, le=1.0, default=0.1)]
    #cost: Annotated[float, Field(ge=0.0, le=1.0, default=0.1)] | None = None

    @property
    @computed_field
    def tokens(self) -> int:
        return count_tokens(self.content)

    @property
    @computed_field
    def cost(self) -> float:
        return self.price * self.tokens

class ImageModelRequest(ModelRequest):
    model: SupportedModels
    output_size: OutputSize
    num_inference_steps: int

    @model_validator(mode="after")
    def validate_inference_steps(self) -> 'ImageModelRequest':
        if self.model == "tinysd" and self.num_inference_steps > 2000:
            raise ValueError("TinySD model cannot have more than 2000 inference steps")
        return self

class ImageModelResponse(ModelResponse):
    size: ImageSize
    url: Annotated[str, HttpUrl] | None = None
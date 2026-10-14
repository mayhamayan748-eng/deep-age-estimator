from pydantic import BaseModel, Field


class HealthResponse(BaseModel):
    status: str = Field(description='"ok" when at least one age model and the face detector are loaded, else "degraded"')
    models: dict[str, bool] = Field(description="Which age models are loaded")
    generator: bool
    face_detector: bool
    default_model: str | None


class BBoxSchema(BaseModel):
    x: int
    y: int
    w: int
    h: int


class FacePrediction(BaseModel):
    bbox: BBoxSchema = Field(description="Face crop (detected box + UTKFace-style margins) in original image pixels")
    age: float = Field(description="Predicted age in years")
    crop: str = Field(description="Base64 PNG of the exact 128x128 input the model saw")


class PredictResponse(BaseModel):
    model: str
    image_width: int
    image_height: int
    faces: list[FacePrediction]


class GenerateRequest(BaseModel):
    age: int = Field(ge=1, le=90)
    count: int = Field(default=4, ge=1, le=8)
    seed: int | None = Field(default=None, ge=0, le=2**31 - 1)


class GenerateResponse(BaseModel):
    age: int
    seed: int = Field(description="Seed used; send it back to reproduce the same faces")
    images: list[str] = Field(description="Base64 PNG, 128x128 grayscale")


class ModelSummary(BaseModel):
    name: str
    label: str
    loaded: bool
    input: str
    val_mae: float | None = None
    test_mae: float | None = None
    test_rmse: float | None = None
    mae_by_age_group: dict[str, float] = {}


class ModelInfoResponse(BaseModel):
    dataset: dict
    preprocessing: list[str]
    models: list[ModelSummary]
    default_model: str | None
    gan: dict


class ErrorResponse(BaseModel):
    detail: str

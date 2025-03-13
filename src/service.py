
import bentoml
import numpy as np
from bentoml.io import JSON
from pydantic import BaseModel
import jwt
from datetime import datetime, timedelta
from typing import Optional

# JWT Configuration
SECRET_KEY = "bentoml-cli-8r"
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30

# Data Models
class User(BaseModel):
    username: str
    password: str

class AdmissionInput(BaseModel):
    gre_score: int
    toefl_score: int
    university_rating: int
    sop: float
    lor: float
    cgpa: float
    research: int

class TokenResponse(BaseModel):
    access_token: str
    token_type: str

# Loading BentoML Models
admission_model = bentoml.sklearn.load_model("lopes_admission_model:latest")
scaler = bentoml.sklearn.load_model("lopes_admission_scaler:latest")

svc = bentoml.Service("LOPES_admission_service")

def create_access_token(username: str):
    # Calculate expiration time (30 minutes from now)
    expire = datetime.utcnow() + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    # Encode the JWT with username and expiration
    return jwt.encode({"sub": username, "exp": expire}, SECRET_KEY, algorithm=ALGORITHM)

def get_current_user(auth_header: Optional[str]):
    if not auth_header or not auth_header.startswith("Bearer "):
        raise ValueError("Missing or invalid token")
    try:
        token = auth_header.split(" ")[1]
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        return payload["sub"]
    except jwt.ExpiredSignatureError:
        raise ValueError("Expired token")
    except jwt.PyJWTError:
        raise ValueError("Invalid token")

@svc.api(input=JSON(pydantic_model=User), output=JSON(pydantic_model=TokenResponse))
def login(user: User, ctx: bentoml.Context):
    if user.username == "admin" and user.password == "password":
        return {"access_token": create_access_token(user.username), "token_type": "bearer"}
    else:
        ctx.response.status_code = 401
        return {"message": "Incorrect credentials"}

@svc.api(input=JSON(pydantic_model=AdmissionInput), output=JSON())
def predict(data: AdmissionInput, ctx: bentoml.Context):
    auth_header = ctx.request.headers.get("Authorization")  
    
    try:
        get_current_user(auth_header)
    except ValueError as e:
        ctx.response.status_code = 401
        return {"message": str(e)}

    input_data = np.array([[
        data.gre_score, 
        data.toefl_score, 
        data.university_rating, 
        data.sop, 
        data.lor, 
        data.cgpa, 
        data.research
    ]])

    input_data_scaled = scaler.transform(input_data)

    prediction = admission_model.predict(input_data_scaled)
    
    return {"chance_of_admit": float(prediction[0])}

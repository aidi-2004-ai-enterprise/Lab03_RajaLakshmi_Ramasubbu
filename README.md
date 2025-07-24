# Penguins XGBoost FastAPI Lab

**Author:** Raja Lakshmi Ramasubbu  
**Repo:** lab3_raja_lakshmi_ramasubbu

This project demonstrates an end-to-end machine learning workflow: training an XGBoost classifier on the Seaborn penguins dataset, then serving predictions via FastAPI. The API uses Pydantic for input validation and returns clear errors for invalid requests. All dependencies are managed and reproducible using `uv`.

---

## Features

- End-to-end ML pipeline: data preprocessing, model training, evaluation, and serving
- FastAPI REST API with strong input validation (Pydantic & Enums)
- Robust error handling (invalid input returns HTTP 422/400)
- Logging for key events and prediction requests
- Easy setup and reproducibility using `uv`
- Demo video included

---

## Setup Instructions

1. **Clone this repository**
    ```sh
    git clone https://github.com/aidi-2004-ai-enterprise/lab3_raja_lakshmi_ramasubbu.git
    cd lab3_raja_lakshmi_ramasubbu
    ```

2. **Create a virtual environment and install dependencies**
    ```sh
    uv venv
    uv pip install .
    ```

3. **Train the model**
    ```sh
    python train.py
    ```

4. **Run the FastAPI app**
    ```sh
    uvicorn app.main:app --reload
    ```

5. **Open the API docs at [http://localhost:8000/docs](http://localhost:8000/docs)**

---

## How to Test

- Visit [http://localhost:8000/docs](http://localhost:8000/docs) for an interactive UI.
- Or use curl:
    ```sh
    curl -X POST "http://localhost:8000/predict" -H "Content-Type: application/json" -d '{
      "bill_length_mm": 39.1,
      "bill_depth_mm": 18.7,
      "flipper_length_mm": 181,
      "body_mass_g": 3750,
      "year": 2007,
      "sex": "male",
      "island": "Torgersen"
    }'
    ```

- **Example successful response:**
    ```json
    {
      { "species": "Adelie" }
    }
    ```

- **Example: Invalid field (`sex`):**
    ```json
    {
      "bill_length_mm": 39.1,
      "bill_depth_mm": 18.7,
      "flipper_length_mm": 181,
      "body_mass_g": 3750,
      "year": 2007,
      "sex": "robot",
      "island": "Torgersen"
    }
    ```
    **Returns HTTP 422 Unprocessable Entity with:**
    ```json
    {
      "detail": [
        {
          "type": "enum",
          "loc": ["body", "sex"],
          "msg": "Input should be 'male' or 'female'",
          "input": "robot"
        }
      ]
    }
    ```
- If a request passes basic validation but fails internal checks, the API may return HTTP 400 with an error message.

---

## Error Handling

- **Invalid categorical values** (e.g., wrong `sex` or `island`) are blocked by Pydantic, returning HTTP 422 by default.
- **Internal processing errors** or feature mismatches are handled with HTTP 400 and a clear error message.
- All errors are logged.

---

## Video Demo

See `demo.mp4` in this repo for:
- API requests using `/docs`
- At least one successful prediction
- At least one error example (e.g., invalid input)

---

## Dependencies

- fastapi
- uvicorn
- xgboost
- pandas
- seaborn
- scikit-learn
- joblib

---

## Acknowledgments

- [Seaborn Penguins Dataset](https://github.com/mwaskom/seaborn-data)
- [FastAPI](https://fastapi.tiangolo.com/)
- [XGBoost](https://xgboost.readthedocs.io/)
- [uv - the fast Python package manager](https://github.com/astral-sh/uv)

---

## Notes

- Invalid categorical values are gracefully handled (HTTP 422/400).
- Includes logging for all key events.
- Uses Pydantic for input validation.
- Model and encoders are saved and loaded for consistent predictions.

---

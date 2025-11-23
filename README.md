# ML Pipeline Web Application

A comprehensive web application for deploying machine learning models with both binary and multiclass classification capabilities.

## 🚀 Features

- **Dual Model Support**: Binary and multiclass classification models
- **Web Interface**: User-friendly web form with manual input and file upload
- **REST API**: JSON endpoints for programmatic access
- **Smart Prediction**: Auto-selects appropriate model based on task type
- **Batch Processing**: Upload CSV files for batch predictions
- **Interactive Results**: Detailed prediction results with confidence scores
- **API Documentation**: Built-in API documentation and examples

## 📋 Prerequisites

- Python 3.8+
- Required packages (see requirements.txt)

## 🛠️ Installation & Setup

### Step 1: Prepare Models

1. Navigate to the web app directory:
```bash
cd ml_web_app
```

2. Run the model preparation script:
```bash
python model_preparation.py
```

This will:
- Load and preprocess your data from `fe_outcomes2.csv`
- Train both binary and multiclass models
- Save model files to the `models/` directory
- Generate metadata and feature schema

### Step 2: Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 3: Run the Application

```bash
python app.py
```

The application will start at: http://localhost:5000

## 📁 Project Structure

```
ml_web_app/
├── app.py                 # Main Flask application
├── model_preparation.py   # Model training and preparation
├── requirements.txt       # Python dependencies
├── models/               # Trained model files (generated)
│   ├── binary_model.pkl
│   ├── multiclass_model.pkl
│   └── model_metadata.json
├── templates/            # HTML templates
│   ├── base.html
│   ├── index.html
│   ├── result.html
│   ├── batch_results.html
│   └── error.html
└── static/              # Static files (CSS, JS, images)
```

## 🌐 Web Interface

### Manual Input
- Select prediction type (Binary or Multiclass)
- Fill in company features manually
- Get instant predictions with confidence scores

### File Upload
- Upload CSV files for batch predictions
- Support for up to 100 rows per file
- Download results as CSV

## 📡 API Endpoints

### Binary Classification
```http
POST /predict-binary
Content-Type: application/json

{
  "entity_id": 123,
  "name": 456,
  "category_code": 42,
  // ... other features
}
```

### Multiclass Classification
```http
POST /predict-multiclass
Content-Type: application/json

{
  "entity_id": 123,
  "name": 456,
  "category_code": 42,
  // ... other features
}
```

### Smart Prediction
```http
POST /predict
Content-Type: application/json

{
  "task_type": "binary",  // or "multiclass"
  "entity_id": 123,
  "name": 456,
  "category_code": 42,
  // ... other features
}
```

### API Information
```http
GET /api/info
```

## 📊 Response Format

```json
{
  "prediction": 3,
  "prediction_label": "Status 3",
  "probabilities": {
    "class_1": 0.15,
    "class_2": 0.25,
    "class_3": 0.60
  },
  "confidence": 0.60,
  "model_type": "multiclass"
}
```

## 🧪 Testing the Application

### Using cURL

Binary prediction:
```bash
curl -X POST http://localhost:5000/predict-binary \
  -H "Content-Type: application/json" \
  -d '{"entity_id": 123, "name": 456, "category_code": 42}'
```

Multiclass prediction:
```bash
curl -X POST http://localhost:5000/predict-multiclass \
  -H "Content-Type: application/json" \
  -d '{"entity_id": 123, "name": 456, "category_code": 42}'
```

### Using Python

```python
import requests
import json

# Example prediction
data = {
    "entity_id": 123,
    "name": 456,
    "category_code": 42,
    # Add more features as needed
}

response = requests.post(
    'http://localhost:5000/predict-multiclass',
    headers={'Content-Type': 'application/json'},
    data=json.dumps(data)
)

result = response.json()
print(f"Prediction: {result['prediction']}")
print(f"Confidence: {result['confidence']:.2%}")
```

## 🔧 Configuration

### Environment Variables
- `FLASK_ENV`: Set to 'development' for debug mode
- `FLASK_SECRET_KEY`: Change the secret key for production

### Model Configuration
Models are automatically loaded from the `models/` directory. Ensure these files exist:
- `binary_model.pkl`
- `multiclass_model.pkl`
- `model_metadata.json`

## 🚀 Deployment

### Local Development
```bash
python app.py
```

### Production Deployment

1. **Using Gunicorn** (recommended for production):
```bash
pip install gunicorn
gunicorn -w 4 -b 0.0.0.0:5000 app:app
```

2. **Using Docker**:
```dockerfile
FROM python:3.9-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .
EXPOSE 5000
CMD ["gunicorn", "-w", "4", "-b", "0.0.0.0:5000", "app:app"]
```

3. **Environment Setup for Production**:
- Set `FLASK_ENV=production`
- Use a proper secret key
- Configure reverse proxy (nginx)
- Enable HTTPS
- Set up monitoring and logging

## 🔍 Troubleshooting

### Common Issues

1. **Models not loading**:
   - Ensure `model_preparation.py` ran successfully
   - Check that all model files exist in `models/` directory

2. **Feature mismatch**:
   - Verify input features match training data schema
   - Check `model_metadata.json` for required features

3. **Memory issues**:
   - Reduce batch size for file uploads
   - Consider model optimization for large datasets

4. **Port already in use**:
   - Change port in `app.py`: `app.run(port=5001)`
   - Or kill existing process: `lsof -ti:5000 | xargs kill`

## 📈 Performance Optimization

- **Caching**: Implement Redis for prediction caching
- **Async Processing**: Use Celery for batch processing
- **Load Balancing**: Deploy multiple instances behind load balancer
- **Model Optimization**: Use model compression techniques
- **Database**: Store predictions in database for analytics

## 🛡️ Security Considerations

- Implement input validation and sanitization
- Add rate limiting for API endpoints
- Use HTTPS in production
- Implement authentication for sensitive endpoints
- Regular security updates for dependencies

## 📝 License

This project is licensed under the MIT License.

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## 📞 Support

For issues and questions:
- Check the troubleshooting section
- Review API documentation at `/api/info`
- Open an issue on the repository
